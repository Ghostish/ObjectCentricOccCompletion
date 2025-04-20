import math
import numpy as np
import torch
from mmcv.runner import BaseModule, force_fp32
from torch import nn as nn
import torch.nn.functional as F

from mmdet3d.core.bbox.structures import (
    LiDARInstance3DBoxes,
    rotation_3d_in_axis,
    xywhr2xyxyr,
)
from mmdet3d.models.builder import build_head, build_loss, build_backbone
from mmdet3d.ops import scatter_v2, build_mlp
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from mmdet.models import HEADS

from mmdet3d.models import builder
from mmdet3d.models.roi_heads.bbox_heads import FullySparseBboxHead
from mmdet3d.ops.sst.sst_ops import get_activation_layer
from mmdet3d.ops.occ import occ_ops
from mmdet3d.models.occ.occ_base import OccDecoder


@HEADS.register_module()
class OccAutoEncoder(FullySparseBboxHead):
    def __init__(
        self,
        backbone,
        occ_decoder,
        voxel_size,
        loss_occ_ae=dict(
            type="FocalLoss", reduction="none", use_sigmoid=True, loss_weight=1.0
        ),
        scale_wlh=[1.0, 1.0, 1.0],
        offset_wlh=[0.0, 0.0, 0.0],
        online_sample_size=-1,
        balance_sample=False,
        with_voxelize_centers=False,
        compensate_encoder_coors=False,
        add_train_prob=0.0,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg)
        self.point_encoder = build_backbone(backbone)
        self.occ_decoder = OccDecoder(**occ_decoder)
        self.loss_occ_ae = build_loss(loss_occ_ae)
        self.voxel_size = voxel_size
        self.scale_wlh = scale_wlh
        self.offset_wlh = offset_wlh
        self.online_sample_size = online_sample_size
        self.balance_sample = balance_sample
        self.loss_need_squeeze = (
            loss_occ_ae["type"] == "CrossEntropyLoss" and loss_occ_ae["use_sigmoid"]
        )
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_voxelize_centers = with_voxelize_centers
        self.compensate_encoder_coors = compensate_encoder_coors
        self.add_train_prob = add_train_prob

    def sample_observation(
        self,
        local_xyz,
        rois,
        pts_roi_inds,
        downsample_size=-1,
        balance_sample=False,
    ):
        assert rois.size(1) in (8, 10)
        if not self.compensate_encoder_coors:
            # fixed coordinate system mismatch
            compensate_r = local_xyz.new_tensor((np.pi / 2,))
            local_xyz = rotation_3d_in_axis(
                local_xyz[None, :, :], compensate_r, axis=2
            ).squeeze(0)
        pts_coors = occ_ops.quantize_points(
            local_xyz,
            rois,
            pts_roi_inds,
            self.voxel_size,
            scale_wlh=self.scale_wlh,
            offset_wlh=self.offset_wlh,
        )
        volume_list = occ_ops.generate_dense_voxel_centers(
            rois[:, 4:7],
            self.voxel_size,
            scale_wlh=self.scale_wlh,
            offset_wlh=self.offset_wlh,
            as_volume=True,
        )

        new_label_list = []
        new_local_xyz_list = []
        new_pts_roi_inds_list = []
        # observed_point_mask_list = []

        for i, volume in enumerate(volume_list):
            # new_smp_pts_xyz_local_i = new_smp_pts_xyz_local[ext_pts_roi_inds == i]
            new_labels = torch.zeros_like(volume[..., 0]).long()
            voxel_dims = torch.tensor(volume.shape[0:3]).to(new_labels.device)
            roi_mask = pts_roi_inds == i
            if roi_mask.any():
                # local_xyz_i = local_xyz[roi_mask]
                pts_coors_i = pts_coors[roi_mask]

                # points may fall right on the boundary, remove them
                valid_coor_masks = (
                    (pts_coors_i < voxel_dims[None]) & (pts_coors_i >= 0)
                ).all(dim=1)
                pts_coors_i = pts_coors_i[valid_coor_masks]
                if len(pts_coors_i) > 0:
                    # continue
                    # set the observed location to 1
                    new_labels[
                        pts_coors_i[:, 0],
                        pts_coors_i[:, 1],
                        pts_coors_i[:, 2],
                    ] = 1
                else:
                    # continue
                    pass
            else:
                # continue  # do not consider all empty observation, is this really correct?
                pass

            new_labels = new_labels.view(-1)
            volume = volume.view(-1, 3)
            if balance_sample:
                indexes = torch.arange(len(new_labels), device=new_labels.device)
                pos_indexes = indexes[new_labels == 1]
                neg_indexes = indexes[new_labels == 0]
                num_neg_sample = int(len(pos_indexes) * 1.0)
                if num_neg_sample > 0 and len(neg_indexes) > 0:
                    choices = torch.multinomial(
                        torch.ones_like(neg_indexes, dtype=torch.float32),
                        num_samples=num_neg_sample,
                        replacement=num_neg_sample > len(neg_indexes),
                    )
                    new_labels = torch.cat(
                        [new_labels[pos_indexes], new_labels[neg_indexes[choices]]]
                    )
                    volume = torch.cat(
                        [volume[pos_indexes], volume[neg_indexes[choices]]]
                    )
                elif len(neg_indexes) > 0:
                    new_labels = new_labels[
                        :1
                    ]  # no positive sample, just use one negative sample
                    volume = volume[:1]
                else:
                    continue  # no negative or pos sample , skip this roi
                if len(new_labels) > downsample_size > 0:
                    choices = torch.multinomial(
                        torch.ones_like(new_labels, dtype=torch.float32),
                        num_samples=downsample_size,
                        replacement=False,
                    )
                    new_labels = new_labels[choices]
                    volume = volume[choices]

            elif len(new_labels) > downsample_size > 0:
                sample_weights = torch.ones_like(new_labels, dtype=torch.float32)
                sample_weights[new_labels == 1] = 100
                # sample_weights[(ori_pred_cls.view(-1) == 1) & (new_labels == 1)] = 10
                # num_samples = min(downsample_size, 2 * new_labels.sum())
                num_samples = downsample_size

                choices = torch.multinomial(
                    sample_weights, num_samples=num_samples, replacement=False
                )
                new_labels = new_labels[choices]
                volume = volume[choices]
                # observed_point_mask = observed_point_mask[choices]

            new_label_list.append(new_labels)
            new_local_xyz_list.append(volume)
            # observed_point_mask_list.append(observed_point_mask)
            new_pts_roi_inds_list.append(
                torch.full(
                    (new_labels.numel(),), i, dtype=torch.long, device=new_labels.device
                )
            )
        if len(new_label_list) == 0:
            return (
                local_xyz.new_zeros((0, 3)),
                local_xyz.new_zeros((0,)),
                local_xyz.new_zeros((0,)),
            )
        obs_occ_labels = torch.cat(new_label_list, dim=0)
        smp_pts_xyz_local = torch.cat(new_local_xyz_list, dim=0)
        smp_pts_roi_inds = torch.cat(new_pts_roi_inds_list, dim=0)

        return (
            smp_pts_xyz_local,
            obs_occ_labels,
            smp_pts_roi_inds,
        )

    @force_fp32(apply_to=("pts_features", "rois"))
    def encode(
        self,
        pts_xyz,
        pts_features,
        pts_info,
        roi_inds,
        rois,
        point_encoder=None,
        cat_global_xyz=False,
    ):
        local_xyz = pts_info["local_xyz"]
        if self.compensate_encoder_coors:
            # fixed coordinate system mismatch
            compensate_r = local_xyz.new_tensor((np.pi / 2,))
            local_xyz = rotation_3d_in_axis(
                local_xyz[None, :, :], compensate_r, axis=2
            ).squeeze(0)

        boundary_offset = pts_info["boundary_offset"]
        is_in_margin = pts_info["is_in_margin"]
        if pts_features is not None:
            out_feats = torch.cat(
                [pts_features, boundary_offset, is_in_margin[:, None]], 1
            )
        else:
            out_feats = torch.cat([boundary_offset, is_in_margin[:, None]], 1)
        if cat_global_xyz:
            out_feats = torch.cat([out_feats, pts_xyz], 1)
        if self.with_voxelize_centers:
            assert (
                self.compensate_encoder_coors
            )  # otherwise the voxel centers are not correct
            # quantize the points to voxel centers
            center_coords = occ_ops.quantize_points(
                local_xyz,
                rois,
                roi_inds,
                self.voxel_size,
                self.scale_wlh,
                self.offset_wlh,
                to_center=True,
            )
            # local_xyz,inverse_indices = torch.unique(local_xyz, dim=0, return_inverse=True)
            out_feats = torch.cat([out_feats, center_coords], 1)
        # TODO: use voxel centers as xyzs instead of features？
        if point_encoder is None:
            point_encoder = self.point_encoder
        out_feats, final_cluster_feats, out_coors = point_encoder(
            local_xyz, out_feats, roi_inds
        )

        if self.training and (out_coors == -1).any():
            assert (
                out_coors[0].item() == -1
            ), "This should hold due to sorted=True in torch.unique"

        nonempty_roi_mask = self.get_nonempty_roi_mask(out_coors, len(rois))
        final_cluster_feats = self.align_roi_feature_and_rois(
            final_cluster_feats, out_coors, len(rois)
        )
        return final_cluster_feats, nonempty_roi_mask, local_xyz

    def decode(self, roi_feats, smp_pts_xyz_local, smp_pts_roi_inds):
        occ_preds = self.occ_decoder(roi_feats, smp_pts_xyz_local, smp_pts_roi_inds)
        return occ_preds

    def forward_train_ae(
        self, pts_xyz, pts_features, pts_info, roi_inds, rois, start_add_train=False
    ):
        local_roi_feats, nonempty_roi_mask, local_xyz = self.encode(
            pts_xyz, pts_features, pts_info, roi_inds, rois
        )

        if start_add_train and torch.rand(1).item() < self.add_train_prob:
            # merge two local rois
            shuffle_rois_inds = torch.randperm(len(rois), device=rois.device)
            shuffle_roi_feats = local_roi_feats[shuffle_rois_inds]
            # local_roi_feats = shuffle_roi_feats + local_roi_feats
            local_roi_feats = torch.max(
                torch.stack([local_roi_feats, shuffle_roi_feats], dim=0), dim=0
            )[0]

            new_rois_inds_list = []
            new_local_xyz_list = []
            for i, j in enumerate(shuffle_rois_inds):
                local_xyz_i = local_xyz[roi_inds == i]
                local_xyz_j = local_xyz[roi_inds == j]
                local_xyz_ij = torch.cat([local_xyz_i, local_xyz_j], dim=0)
                pts_roi_ids = torch.full(
                    (len(local_xyz_ij),),
                    i,
                    dtype=torch.long,
                    device=local_xyz_ij.device,
                )
                new_rois_inds_list.append(pts_roi_ids)
                new_local_xyz_list.append(local_xyz_ij)
            new_local_xyz = torch.cat(new_local_xyz_list, dim=0)
            new_rois_inds = torch.cat(new_rois_inds_list, dim=0)
            # we only care the sizes of the rois since the AE operates on local coordinates
            # when merging two rois, we need to select the larger one to be the final roi
            new_rois = rois.clone()
            ori_rois_size = rois[:, 4:7]
            shuffle_rois_size = rois[shuffle_rois_inds][:, 4:7]
            new_rois[:, 4:7] = torch.max(
                torch.stack([ori_rois_size, shuffle_rois_size], dim=0), dim=0
            )[0]
            (
                smp_pts_xyz_local,
                obs_occ_labels,
                smp_pts_roi_inds,
            ) = self.sample_observation(
                new_local_xyz,
                new_rois,
                new_rois_inds,
                downsample_size=self.online_sample_size,
                balance_sample=self.balance_sample,
            )

        else:
            (
                smp_pts_xyz_local,
                obs_occ_labels,
                smp_pts_roi_inds,
            ) = self.sample_observation(
                local_xyz,
                rois,
                roi_inds,
                downsample_size=self.online_sample_size,
                balance_sample=self.balance_sample,
            )

        occ_preds = self.decode(local_roi_feats, smp_pts_xyz_local, smp_pts_roi_inds)
        loss = self.loss(
            occ_preds,
            local_roi_feats,
            smp_pts_xyz_local,
            smp_pts_roi_inds,
            obs_occ_labels,
            nonempty_roi_mask,
        )
        return loss

    @force_fp32(apply_to=("roi_features", "pts_smp"))
    def online_tuning_forward(
        self,
        roi_features,
        pts_smp,
        pts_labels,
        pts_weights,
        pts_roi_inds,
        num_ttt_iter,
        apply_jitter=False,
    ):
        roi_embed = roi_features.clone().detach()
        roi_embed.requires_grad = True
        if pts_weights is None:
            pts_weights = torch.ones_like(pts_labels, dtype=torch.float32)
        train_state = self.training
        with torch.enable_grad():
            optimizer = torch.optim.Adam([roi_embed], lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.1)
            self.eval()
            i = 0
            for param in self.parameters():
                param.requires_grad = False

            while i < num_ttt_iter:
                optimizer.zero_grad()
                occ_preds = self.decode(roi_embed, pts_smp, pts_roi_inds)

                if self.loss_need_squeeze:
                    occ_preds = occ_preds.view(-1)
                else:
                    occ_preds = occ_preds.view(-1, 1)
                if len(occ_preds) > 0:
                    loss_ae = self.loss_occ_ae(
                        occ_preds, pts_labels.view(-1), pts_weights
                    ).mean()  # num_points,cls_dim

                    loss_ae.backward()
                    optimizer.step()
                    scheduler.step()
                i += 1

        self.train(train_state)
        for param in self.parameters():
            param.requires_grad = train_state
        return roi_embed

    def forward_test_ae(self, pts_xyz, pts_features, pts_info, roi_inds, rois):
        local_roi_feats, nonempty_roi_mask, local_xyz = self.encode(
            pts_xyz, pts_features, pts_info, roi_inds, rois
        )
        if self.test_cfg and self.test_cfg.get("online_tuning", False):
            (
                smp_pts_xyz_local,
                obs_occ_labels,
                smp_pts_roi_inds,
            ) = self.sample_observation(
                local_xyz,
                rois,
                roi_inds,
                downsample_size=self.test_cfg.get("downsample_size", -1),
                balance_sample=self.test_cfg.get("balance_sample", False),
            )
            local_roi_feats = self.online_tuning_forward(
                local_roi_feats,
                smp_pts_xyz_local,
                obs_occ_labels,
                None,
                smp_pts_roi_inds,
                self.test_cfg.get("num_iter", 10),
            )
        occs = self.get_occ(local_roi_feats, rois)

        return occs

    def get_occ(
        self,
        local_roi_feats,
        rois,
        transform=True,
    ):
        return self.occ_decoder.get_occ(
            local_roi_feats,
            rois,
            self.voxel_size,
            self.scale_wlh,
            self.offset_wlh,
            # concat_batch=False,
            # local_xyz=smp_pts_xyz_local[obs_occ_labels == 1],
            # local_pts_roi_inds=smp_pts_roi_inds[obs_occ_labels == 1],
            # return_full=False,
            transform=transform,
        )

    def get_roi_occ(self, local_roi_feats, rois, transform=True, return_score=False):
        return self.occ_decoder.get_roi_occ(
            local_roi_feats,
            rois,
            self.voxel_size,
            self.scale_wlh,
            self.offset_wlh,
            transform,
            return_score,
        )

    def loss(
        self,
        occ_preds,
        local_roi_feats,
        smp_pts_xyz_local,
        smp_pts_roi_inds,
        obs_occ_labels,
        nonempty_roi_mask,
    ):
        # The features is extracted from roi pooling in local coordinate system
        # this loss forces the decoded occupancies to be close to the observation
        num_occupied = obs_occ_labels.sum().float()
        num_free = obs_occ_labels.numel() - num_occupied
        per_points_masks = nonempty_roi_mask[smp_pts_roi_inds]
        num_valid_occupied = (
            ((obs_occ_labels == 1) & (per_points_masks == 1)).sum().float()
        )
        num_valid_free = ((obs_occ_labels == 0) & (per_points_masks == 1)).sum().float()
        ret_dict = {}
        if self.loss_need_squeeze:
            occ_preds = occ_preds.view(-1)
        else:
            occ_preds = occ_preds.view(-1, 1)
        assert len(smp_pts_xyz_local) > 0
        loss_ae = self.loss_occ_ae(
            occ_preds,
            obs_occ_labels.view(-1),
            # weights,
        )  # num_points,cls_dim

        assert (smp_pts_roi_inds >= 0).all()
     
        loss_ae = loss_ae.mean()
        if self.loss_need_squeeze:
            pred_cls = (occ_preds.sigmoid() > 0.5).long().view(-1)
        else:
            pred_cls = (occ_preds.sigmoid() < 0.5).long().view(-1)
        num_pred_occupied = pred_cls.sum().float()
        num_pred_free = pred_cls.numel() - num_pred_occupied
        num_gt_occupied = obs_occ_labels.sum().float()
        num_gt_free = obs_occ_labels.numel() - num_gt_occupied
        num_correct_occupied = ((pred_cls == 1) & (obs_occ_labels == 1)).sum().float()
        num_correct_free = ((pred_cls == 0) & (obs_occ_labels == 0)).sum().float()
        recall_occupied = num_correct_occupied / (num_gt_occupied + 1e-6)
        recall_free = num_correct_free / (num_gt_free + 1e-6)
        precision_occupied = num_correct_occupied / (num_pred_occupied + 1e-6)
        precision_free = num_correct_free / (num_pred_free + 1e-6)
        ret_dict.update(
            recall_free=recall_free,
            recall_occupied=recall_occupied,
            precision_free=precision_free,
            precision_occupied=precision_occupied,
        )

        ret_dict["loss_ae"] = loss_ae
        ret_dict["num_occupied"] = num_occupied
        ret_dict["num_free"] = num_free
        ret_dict["num_valid_occupied"] = num_valid_occupied
        ret_dict["num_valid_free"] = num_valid_free
        return ret_dict
