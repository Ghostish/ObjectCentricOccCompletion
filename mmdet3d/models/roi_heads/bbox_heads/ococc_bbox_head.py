import warnings

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
from mmdet3d.models.roi_heads.bbox_heads.fsd_bbox_head import FullySparseBboxHead
from mmdet3d.ops import scatter_v2, build_mlp
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from mmdet.models import HEADS

from mmdet3d.models import builder
from mmdet3d.models.roi_heads.bbox_heads import (
    FullySparseBboxHead,
)
from mmdet3d.ops.sst.sst_ops import get_activation_layer
import pdb
from mmdet3d.models.occ.layers import (
    PositionalEncoding,
    SimpleDecoderLayer,
    SimpleEncoderLayer,
    TransformerDecoder,
    TransformerEncoder,
)


@HEADS.register_module()
class OccBBoxHead(FullySparseBboxHead):
    def __init__(
        self,
        num_blocks,
        in_channels,
        feat_channels,
        rel_mlp_hidden_dims,
        rel_mlp_in_channels,
        with_rel_mlp=True,
        with_cluster_center=False,
        with_distance=False,
        mode="max",
        xyz_normalizer=[20, 20, 4],
        geo_input=True,
        dropout=0,
        unique_once=True,
        occ_ae_head=None,
        roi_feature_channels=None,
        init_cfg=None,
        debug=False,
        fixed_ae=True,
        attn_num_head=4,
        attn_ffn_dim=2048,
        attn_dropout=0.1,
        loss_occ_comp=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="none", loss_weight=1.0
        ),
        num_classes=1,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        occ_label_thresh=0.8,
        reg_mlp=None,
        cls_mlp=None,
        latent_mlp=None,
        fusion_mlp=None,
        act="gelu",
        norm_cfg=dict(type="LN", eps=1e-3),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="none", loss_weight=1.0
        ),
        cls_dropout=0,
        reg_dropout=0,
        latent_dropout=0,
        fusion_dropout=0,
        with_corner_loss=False,
        with_roi_pos_encoding=False,
        roi_pos_enc_mlp=None,
        roi_enc_dropout=0,
        num_enc_layers=1,
        fused_mode="residual",
        rcnn_trans=True,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        self.occ_ae_head = builder.build_head(occ_ae_head)
        self.debug = debug
        self.fixed_ae = fixed_ae
        if self.fixed_ae:
            for p in self.occ_ae_head.parameters():
                p.requires_grad = False
            self.occ_ae_head.eval()
        # basically, we only train the transformer
        self.with_corner_loss = with_corner_loss
        encoder_layer = SimpleEncoderLayer(
            roi_feature_channels,
            attn_num_head,
            dim_feedforward=attn_ffn_dim,
            dropout=attn_dropout,
        )
        self.trans_enc = TransformerEncoder(encoder_layer, num_enc_layers)
        self.pos_enc = PositionalEncoding(roi_feature_channels)
        self.loss_occ_comp = build_loss(loss_occ_comp)
        self.num_classes = num_classes
        self.occ_label_thresh = occ_label_thresh
        self.with_roi_pos_encoding = with_roi_pos_encoding
        self.roi_feature_channels = roi_feature_channels
        if self.with_roi_pos_encoding:
            self.roi_pos_enc_mlp = build_mlp(
                7,  # xyz+wlh+r_y
                roi_pos_enc_mlp
                + [
                    roi_feature_channels,
                ],
                norm_cfg,
                True,
                act=act,
                dropout=roi_enc_dropout,
            )
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        if cls_mlp is not None:
            self.conv_cls = build_mlp(
                roi_feature_channels,
                cls_mlp
                + [
                    1,
                ],
                norm_cfg,
                True,
                act=act,
                dropout=cls_dropout,
            )
        else:
            self.conv_cls = nn.Linear(roi_feature_channels, 1)
        if reg_mlp is not None:
            self.conv_reg = build_mlp(
                roi_feature_channels,
                reg_mlp
                + [
                    self.box_code_size,
                ],
                norm_cfg,
                True,
                act=act,
                dropout=reg_dropout,
            )
        else:
            self.conv_reg = nn.Linear(roi_feature_channels, self.box_code_size)
        self.fused_mode = fused_mode
        if fused_mode == "residual":
            latent_in_channels = roi_feature_channels
        elif fused_mode == "concat" or fused_mode == "concat_residual":
            latent_in_channels = roi_feature_channels * 2
        else:
            raise NotImplementedError(f"Unknown fused_mode: {fused_mode}")
        if latent_mlp is not None:

            self.conv_latent = build_mlp(
                latent_in_channels,
                latent_mlp
                + [
                    roi_feature_channels,
                ],
                norm_cfg,
                True,
                act=act,
                dropout=latent_dropout,
            )
        else:
            self.conv_latent = nn.Linear(latent_in_channels, roi_feature_channels)
        if fusion_mlp is not None:
            self.conv_fused = build_mlp(
                roi_feature_channels * 2,
                fusion_mlp
                + [
                    roi_feature_channels,
                ],
                norm_cfg,
                True,
                act=act,
                dropout=fusion_dropout,
            )
        else:
            self.conv_fused = nn.Linear(roi_feature_channels * 2, roi_feature_channels)

        # roi encoder
        self.geo_input = geo_input
        self.unique_once = unique_once
        self.num_blocks = num_blocks
        block_list = []
        for i in range(num_blocks):
            return_point_feats = i != num_blocks - 1
            kwargs = dict(
                type="SIRLayer",
                in_channels=in_channels[i],
                feat_channels=feat_channels[i],
                with_distance=with_distance,
                with_cluster_center=with_cluster_center,
                with_rel_mlp=with_rel_mlp,
                rel_mlp_hidden_dims=rel_mlp_hidden_dims[i],
                rel_mlp_in_channel=rel_mlp_in_channels[i],
                with_voxel_center=False,
                voxel_size=[0.1, 0.1, 0.1],  # not used, placeholder
                point_cloud_range=[
                    -74.88,
                    -74.88,
                    -2,
                    74.88,
                    74.88,
                    4,
                ],  # not used, placeholder
                norm_cfg=norm_cfg,
                mode=mode,
                fusion_layer=None,
                return_point_feats=return_point_feats,
                return_inv=False,
                rel_dist_scaler=10.0,
                xyz_normalizer=xyz_normalizer,
                act=act,
                dropout=dropout,
            )
            encoder = builder.build_voxel_encoder(kwargs)
            block_list.append(encoder)
        self.block_list = nn.ModuleList(block_list)

        self.rcnn_trans = rcnn_trans

    @force_fp32(apply_to=("pts_features", "rois"))
    def roi_encode(self, pts_xyz, pts_features, pts_info, roi_inds, rois):
        """Forward pass.

        Args:
            seg_feats (torch.Tensor): Point-wise semantic features.
            part_feats (torch.Tensor): Point-wise part prediction features.

        Returns:
            tuple[torch.Tensor]: Score of class and bbox predictions.
        """
        assert pts_features.size(0) > 0

        rois_batch_idx = rois[:, 0]
        rois = rois[:, 1:]
        roi_centers = rois[:, :3]
        rel_xyz = pts_xyz[:, :3] - roi_centers[roi_inds]

        if self.unique_once:
            # note cd : roi_inds may contain -1, which is not allowed in torch_scatter,
            # so we need to use torch.unique to transfer roi_inds to a new index starting from 0
            new_coors, unq_inv = torch.unique(
                roi_inds, return_inverse=True, return_counts=False, dim=0
            )
        else:
            new_coors = unq_inv = None

        out_feats = pts_features
        f_cluster = torch.cat(
            [
                pts_info["local_xyz"],
                pts_info["boundary_offset"],
                pts_info["is_in_margin"][:, None],
                rel_xyz,
            ],
            dim=-1,
        )

        cluster_feat_list = []
        for i, block in enumerate(self.block_list):
            in_feats = torch.cat([pts_xyz, out_feats], 1)

            if self.geo_input:
                in_feats = torch.cat([in_feats, f_cluster / 10], 1)

            if i < self.num_blocks - 1:
                # return point features
                # note cd: SIR layer acts as a voxel encoder layer. Each roi is treated as a voxel, and the points in the roi are treated as points in the voxel.
                out_feats, out_cluster_feats = block(
                    in_feats,
                    roi_inds,
                    f_cluster,
                    unq_inv_once=unq_inv,
                    new_coors_once=new_coors,
                )
                cluster_feat_list.append(out_cluster_feats)
            if i == self.num_blocks - 1:
                # return group features
                out_cluster_feats, out_coors = block(
                    in_feats,
                    roi_inds,
                    f_cluster,
                    unq_inv_once=unq_inv,
                    new_coors_once=new_coors,
                )
                cluster_feat_list.append(out_cluster_feats)

        final_cluster_feats = torch.cat(cluster_feat_list, dim=1)

        if self.training and (out_coors == -1).any():
            assert (
                out_coors[0].item() == -1
            ), "This should hold due to sorted=True in torch.unique"

        nonempty_roi_mask = self.get_nonempty_roi_mask(out_coors, len(rois))

        final_cluster_feats = self.align_roi_feature_and_rois(
            final_cluster_feats, out_coors, len(rois)
        )
        return final_cluster_feats, nonempty_roi_mask, out_coors

    @force_fp32(apply_to=("pts_features", "rois"))
    def forward(
        self,
        pts_xyz,
        pts_features,
        pts_info,
        roi_inds,
        rois,
        roi_frame_inds,
    ):
        # each local roi_feat only knows information within the its own ROI (maybe enlarged) from the same frame.
        # the roi_feats can be used to decode occupancy observation
        if pts_xyz.size(0) == 0:
            final_cluster_feats = pts_features.new_zeros(
                (len(rois), self.roi_feature_channels)
            )
            nonempty_roi_mask = pts_features.new_zeros(len(rois), dtype=torch.bool)
        else:
            final_cluster_feats, nonempty_roi_mask, out_coors = self.roi_encode(
                pts_xyz,
                pts_features,
                pts_info,
                roi_inds,
                rois,
            )
        # local encoder only cares xyz, intensity and elongation of the points
        local_roi_feats, nonempty_roi_mask_local, local_xyz = self.occ_ae_head.encode(
            pts_xyz,
            pts_features[:, :2],
            pts_info,
            roi_inds,
            rois,
        )

        # after interacts with previous roi feats, each roi feats now encodes info from historical rois from the same tracklet.

        roi_feats_fused = self.transformer_forward(
            rois, roi_frame_inds, final_cluster_feats, nonempty_roi_mask
        )
        # since we already have the observation latent, we only need to predict the residual to complete observation
        if self.fused_mode == "residual":
            shape_latent_residual = self.conv_latent(roi_feats_fused)
            shape_latent = local_roi_feats + shape_latent_residual
        elif self.fused_mode == "concat":
            shape_latent = torch.cat([local_roi_feats, roi_feats_fused], dim=1)
            shape_latent = self.conv_latent(shape_latent)
        elif self.fused_mode == "concat_residual":
            shape_latent_residual = torch.cat([local_roi_feats, roi_feats_fused], dim=1)
            shape_latent_residual = self.conv_latent(shape_latent_residual)
            shape_latent = local_roi_feats + shape_latent_residual
        else:
            raise NotImplementedError(f"Unknown fused_mode: {self.fused_mode}")
        # shape_latent is used to decode occupancy observation
        if not self.training and self.test_cfg.get("online_tuning", None) is not None:
            shape_latent = self.online_tuning(
                shape_latent,
                local_xyz,
                rois,
                roi_inds,
                downsample_size=self.test_cfg.online_tuning["downsample_size"],
                balance_sample=self.test_cfg.online_tuning["balance_sample"],
                num_iter=self.test_cfg.online_tuning["num_iter"],
            )
        ret_dict = dict(
            fused_roi_feats=shape_latent,
            nonempty_roi_mask=nonempty_roi_mask,
            ori_roi_feats=local_roi_feats,
        )
        if self.rcnn_trans:
            roi_feats_fused = self.conv_fused(
                torch.cat([shape_latent, roi_feats_fused], dim=1)
            )
        else:
            roi_feats_fused = self.conv_fused(
                torch.cat([shape_latent, final_cluster_feats], dim=1)
            )

        cls_score = self.conv_cls(roi_feats_fused)
        bbox_pred = self.conv_reg(roi_feats_fused)
        ret_dict.update(cls_score=cls_score)
        ret_dict.update(bbox_pred=bbox_pred)

        return ret_dict

    def online_tuning(
        self,
        shape_latent,
        local_xyz,
        rois,
        roi_inds,
        downsample_size=-1,
        balance_sample=False,
        num_iter=10,
    ):
        (
            smp_pts_xyz_local,
            obs_occ_labels,
            smp_pts_roi_inds,
        ) = self.occ_ae_head.sample_observation(
            local_xyz,
            rois,
            roi_inds,
            downsample_size=downsample_size,
            balance_sample=balance_sample,
        )
        shape_latent = self.occ_ae_head.online_tuning_forward(
            shape_latent,
            smp_pts_xyz_local,
            obs_occ_labels,
            None,
            smp_pts_roi_inds,
            num_iter,
        )
        return shape_latent

    def loss(
        self,
        results_dict,
        rois,
        labels,
        bbox_targets,
        pos_batch_idx,
        pos_gt_bboxes,
        pos_gt_labels,
        reg_mask,
        label_weights,
        bbox_weights,
        pos_roi_local_xyz,
        gt_occ,
        occ_scores,
        occ_reg_mask,
        occ_pos_batch_idx,
        pos_gt_bboxes_occ,
        transform_occ=False,
        roi_frame_inds=None,
    ):
        losses = {}
        # calculate class loss
        cls_score = results_dict["cls_score"]
        nonempty_roi_mask = results_dict["nonempty_roi_mask"]
        bbox_pred = results_dict["bbox_pred"]

        num_total_samples = rcnn_batch_size = cls_score.shape[0]
        assert num_total_samples > 0

        # calculate class loss
        cls_flat = cls_score.view(-1)  # only to classify foreground and background
        label_weights = label_weights.clone()
        bbox_weights = bbox_weights.clone()
        reg_mask = reg_mask.clone()
        label_weights[~nonempty_roi_mask] = (
            0  # do not calculate cls loss for empty rois
        )
        label_weights[nonempty_roi_mask] = (
            1  # we use avg_factor in loss_cls, so we need to set it to 1
        )
        bbox_weights[...] = (
            1  # we use avg_factor in loss_bbox, so we need to set it to 1
        )

        reg_mask[~nonempty_roi_mask] = 0  # do not calculate loss for empty rois

        cls_avg_factor = num_total_samples * 1.0
        if self.train_cfg.get("sync_cls_avg_factor", False):
            cls_avg_factor = reduce_mean(bbox_weights.new_tensor([cls_avg_factor]))

        loss_cls = self.loss_cls(
            cls_flat, labels, label_weights, avg_factor=cls_avg_factor
        )
        losses["loss_rcnn_cls"] = loss_cls

        # calculate regression loss
        pos_inds = reg_mask > 0
        losses["num_pos_rois"] = pos_inds.sum().float()
        losses["num_neg_rois"] = (reg_mask <= 0).sum().float()

        reg_avg_factor = pos_inds.sum().item()
        if self.train_cfg.get("sync_reg_avg_factor", False):
            reg_avg_factor = reduce_mean(bbox_weights.new_tensor([reg_avg_factor]))

        if pos_inds.any() == 0:
            # fake a bbox loss
            losses["loss_rcnn_bbox"] = bbox_pred.sum() * 0
            if self.with_corner_loss:
                losses["loss_rcnn_corner"] = bbox_pred.sum() * 0
        else:
            pos_bbox_pred = bbox_pred[pos_inds]
            # bbox_targets should have same size with pos_bbox_pred in normal case. But reg_mask is modified by nonempty_roi_mask. So it could be different.
            # filter bbox_targets per sample

            bbox_targets = self.filter_pos_assigned_but_empty_rois(
                bbox_targets, pos_batch_idx, pos_inds, rois[:, 0].int()
            )

            assert not (pos_bbox_pred == -1).all(1).any()
            bbox_weights_flat = (
                bbox_weights[pos_inds].view(-1, 1).repeat(1, pos_bbox_pred.shape[-1])
            )

            code_weights = self.train_cfg.get("rcnn_code_weights", None)
            if code_weights is not None:
                code_weights = torch.tensor(
                    code_weights,
                    dtype=bbox_weights_flat.dtype,
                    device=bbox_weights_flat.device,
                )
                bbox_weights_flat = bbox_weights_flat * code_weights[None, :]

            if pos_bbox_pred.size(0) != bbox_targets.size(0):
                raise ValueError("Impossible after filtering bbox_targets")
                # I don't know why this happens
                losses["loss_rcnn_bbox"] = bbox_pred.sum() * 0
                if self.with_corner_loss:
                    losses["loss_rcnn_corner"] = bbox_pred.sum() * 0
                return losses

            assert bbox_targets.numel() > 0
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets,
                bbox_weights_flat,
                avg_factor=reg_avg_factor,
            )
            losses["loss_rcnn_bbox"] = loss_bbox

            if self.with_corner_loss:
                code_size = self.bbox_coder.code_size
                pos_roi_boxes3d = rois[..., 1 : code_size + 1].view(-1, code_size)[
                    pos_inds
                ]
                pos_roi_boxes3d = pos_roi_boxes3d.view(-1, code_size)
                batch_anchors = pos_roi_boxes3d.clone().detach()
                pos_rois_rotation = pos_roi_boxes3d[..., 6].view(-1)
                roi_xyz = pos_roi_boxes3d[..., 0:3].view(-1, 3)
                batch_anchors[..., 0:3] = 0
                # decode boxes
                pred_boxes3d = self.bbox_coder.decode(
                    batch_anchors, pos_bbox_pred.view(-1, code_size)
                ).view(-1, code_size)

                pred_boxes3d[..., 0:3] = rotation_3d_in_axis(
                    pred_boxes3d[..., 0:3].unsqueeze(1),
                    (pos_rois_rotation + np.pi / 2),
                    axis=2,
                ).squeeze(1)

                pred_boxes3d[:, 0:3] += roi_xyz

                # calculate corner loss
                assert pos_gt_bboxes.size(0) == pos_gt_labels.size(0)
                pos_gt_bboxes = self.filter_pos_assigned_but_empty_rois(
                    pos_gt_bboxes, pos_batch_idx, pos_inds, rois[:, 0].int()
                )
                pos_gt_labels = self.filter_pos_assigned_but_empty_rois(
                    pos_gt_labels, pos_batch_idx, pos_inds, rois[:, 0].int()
                )
                if self.train_cfg.get("corner_loss_only_car", True):
                    car_type_index = self.train_cfg["class_names"].index("Car")
                    car_mask = pos_gt_labels == car_type_index
                    pos_gt_bboxes = pos_gt_bboxes[car_mask]
                    pred_boxes3d = pred_boxes3d[car_mask]
                if len(pos_gt_bboxes) > 0:
                    loss_corner = (
                        self.get_corner_loss_lidar(pred_boxes3d, pos_gt_bboxes)
                        * self.corner_loss_weight
                    )
                else:
                    loss_corner = bbox_pred.sum() * 0

                losses["loss_rcnn_corner"] = loss_corner

        loss_occ = self.loss_occ(
            rois,
            results_dict["fused_roi_feats"],
            results_dict["ori_roi_feats"],
            occ_pos_batch_idx,
            pos_gt_bboxes_occ,
            occ_reg_mask.clone(),
            results_dict["nonempty_roi_mask"],
            pos_roi_local_xyz,
            gt_occ,
            occ_scores,
            transform_occ=transform_occ,
            roi_frame_inds=roi_frame_inds,
            # do_aug=results_dict["do_aug"],
        )

        losses.update(loss_occ)
        return losses

    def loss_occ(
        self,
        rois,
        roi_features,
        ori_roi_feats,
        pos_batch_idx,
        pos_gt_bboxes,
        reg_mask,
        nonempty_roi_mask,
        gt_smp_local_coords,
        gt_smp_occ_labels,
        gt_occ_label_scores,
        transform_occ=False,
        roi_frame_inds=None,
        do_aug=False,
    ):
        losses = {}
      
        reg_mask[~nonempty_roi_mask] = (
            0  # do not calculate loss for empty rois, the reg_mask is already filtered by nonempty_roi_mask
        )
        # reg_mask[roi_frame_mask == 0] = 0
        pos_inds = reg_mask > 0

        num_occupied = (gt_smp_occ_labels == 1).sum().float()
        num_free = gt_smp_occ_labels.numel() - num_occupied
        losses["num_occupied"] = num_occupied
        losses["num_free"] = num_free
        if self.train_cfg.get("contrastive_loss", False):
            losses["loss_contrastive_feats"] = F.mse_loss(
                roi_features, ori_roi_feats.detach()
            ) * self.train_cfg.get("contrastive_loss_weight", 1.0)
        if pos_inds.any() == 0:
            losses["loss_rcnn_occ"] = (
                self.occ_ae_head.occ_decoder.occ_forward(
                    roi_features, roi_features.new_zeros(roi_features.size(0), 3)
                )
                * 0
            )

            losses["recall_neg"] = roi_features.new_ones(1)
            losses["recall_pos"] = roi_features.new_ones(1)
            losses["precision_neg"] = roi_features.new_ones(1)
            losses["precision_pos"] = roi_features.new_ones(1)
            if self.train_cfg.get("residual_loss", False):
                losses["loss_rcnn_occ_residual"] = (
                    self.occ_ae_head.occ_decoder.occ_forward(
                        roi_features, roi_features.new_zeros(roi_features.size(0), 3)
                    )
                    * 0
                )
                losses["num_unmatched"] = roi_features.new_zeros(1)
        else:
            # filter targets according to pos_inds
            pos_roi_features = roi_features[pos_inds]  # M, D
            assert len(roi_features) == len(rois)
            assert (
                len(gt_smp_occ_labels) == len(gt_occ_label_scores) == len(pos_gt_bboxes)
            ), f"{len(gt_smp_occ_labels)}, {len(gt_occ_label_scores)}, {len(pos_gt_bboxes)}"
            assert gt_smp_occ_labels.dim() == 3, f"{gt_smp_occ_labels.dim()}"
            assert pos_gt_bboxes.dim() == 2, f"{pos_gt_bboxes.dim()}"
            occ_targets = self.filter_pos_assigned_but_empty_rois(
                gt_smp_occ_labels, pos_batch_idx, pos_inds, rois[:, 0].int()
            )  # (M, K, 1), M rois, each roi have K occ results

            occ_smp_xyz = self.filter_pos_assigned_but_empty_rois(
                gt_smp_local_coords, pos_batch_idx, pos_inds, rois[:, 0].int()
            )
            pos_gt_bboxes = self.filter_pos_assigned_but_empty_rois(
                pos_gt_bboxes, pos_batch_idx, pos_inds, rois[:, 0].int()
            )
    
            if transform_occ:

                pred_rois_boxes = rois[pos_inds][:, 1:]
                with torch.no_grad():
                    # first transform from gt box coordiante to ego frame coordiante
                    occ_smp_xyz = rotation_3d_in_axis(
                        occ_smp_xyz, pos_gt_bboxes[:, 6], axis=2
                    )
                    occ_smp_xyz += pos_gt_bboxes[..., None, 0:3]
                    # the occ label is the gravity center of the voxel, while the gt box is the bottom center
                    occ_smp_xyz[..., 2] += pos_gt_bboxes[:, None, 5] / 2
                    # occ_smp_xyz[...,2] += occ_sizes[:,None, 2]/2

                    # then transform the smp_xyz from the ego frame to the roi frame
                    occ_smp_xyz -= pred_rois_boxes[..., None, :3]
                    occ_smp_xyz[..., 2] -= pred_rois_boxes[:, None, 5] / 2

                    occ_smp_xyz = rotation_3d_in_axis(
                        occ_smp_xyz, -(pred_rois_boxes[:, 6]), axis=2
                    )
            if do_aug:
                occ_smp_xyz[..., 2] = -occ_smp_xyz[..., 2]
            occ_label_scores = self.filter_pos_assigned_but_empty_rois(
                gt_occ_label_scores, pos_batch_idx, pos_inds, rois[:, 0].int()
            )  # M,1
            # occ_label_scores = gt_occ_label_scores
            M, K, _ = occ_targets.shape
            # mask unconfident occ annos
            occ_weights = torch.zeros_like(occ_label_scores)
            occ_weights[occ_label_scores > self.occ_label_thresh] = 1
            occ_weights = occ_weights.view(M, 1).repeat(1, K)  # M, K
            pos_roi_features_rp = pos_roi_features[:, None, :].repeat(1, K, 1)  # M,K,D

            if self.train_cfg.get("no_loss_for_outside", False):
                with torch.no_grad():
                    pred_rois_boxes = rois[pos_inds][:, 1:]
                    pred_box_size = pred_rois_boxes[:, 3:6]  # M,3
                    occ_smp_mask = (occ_smp_xyz >= -pred_box_size[:, None, :] / 2).all(
                        dim=-1
                    ) & (occ_smp_xyz <= pred_box_size[:, None, :] / 2).all(
                        dim=-1
                    )  # M,K
                    occ_weights = occ_weights * occ_smp_mask.float()
            if self.train_cfg.get("no_loss_for_observed_feats", False):
                with torch.no_grad():
                    pos_ori_roi_feats = ori_roi_feats[pos_inds]
                    pos_ori_roi_feats_rp = pos_ori_roi_feats[:, None, :].repeat(1, K, 1)
                    ori_occ_preds = self.occ_ae_head.occ_decoder.occ_forward(
                        pos_ori_roi_feats_rp, occ_smp_xyz
                    )
                    ori_pred_cls = self.occ_ae_head.occ_decoder.get_cls_from_pred(
                        ori_occ_preds
                    )
                    unobserved_mask = ori_pred_cls == 0
                    occ_weights = occ_weights * unobserved_mask.float()

            occ_preds = self.occ_ae_head.occ_decoder.occ_forward(
                pos_roi_features_rp, occ_smp_xyz
            )
            occ_labels = (occ_targets[..., -1] == 1).long()  # 1 indicates occupied

            if self.train_cfg.get("residual_loss", False):
                assert "residual" in self.fused_mode, f"{self.fused_mode}"
                with torch.no_grad():
                    pos_ori_roi_feats = ori_roi_feats[pos_inds]
                    pos_ori_roi_feats_rp = pos_ori_roi_feats[:, None, :].repeat(1, K, 1)
                    ori_occ_preds = self.occ_ae_head.occ_decoder.occ_forward(
                        pos_ori_roi_feats_rp, occ_smp_xyz
                    )
                    ori_pred_cls = self.occ_ae_head.occ_decoder.get_cls_from_pred(
                        ori_occ_preds
                    ).view(-1)
                    unmatched_mask = occ_labels.view(-1) != ori_pred_cls
                losses["num_unmatched"] = unmatched_mask.sum().float()
                if unmatched_mask.any():
                    latent_residual = pos_roi_features_rp - pos_ori_roi_feats_rp
                    residual_occ_preds = self.occ_ae_head.occ_decoder.occ_forward(
                        latent_residual, occ_smp_xyz
                    )
                    loss_occ_residual = self.loss_occ_comp(
                        residual_occ_preds.view(
                            -1,
                        )[unmatched_mask],
                        occ_labels.view(-1)[unmatched_mask],
                        occ_weights.view(-1)[unmatched_mask],
                    )
                    losses["loss_rcnn_occ_residual"] = loss_occ_residual
                else:
                    warnings.warn("No unmatched samples found")
                    losses["loss_rcnn_occ_residual"] = (
                        self.occ_ae_head.occ_decoder.occ_forward(
                            roi_features,
                            roi_features.new_zeros(roi_features.size(0), 3),
                        )
                        * 0
                    )
            loss_occ = self.loss_occ_comp(
                occ_preds.view(
                    -1,
                ),
                occ_labels.view(-1),
                occ_weights.view(-1),
            )
           
            losses["loss_rcnn_occ"] = loss_occ
            with torch.no_grad():
                # information for logging

                pred_cls = self.occ_ae_head.occ_decoder.get_cls_from_pred(
                    occ_preds.view(-1)
                )
                occ_labels = occ_labels.view(-1)
                valid_mask = occ_weights.view(-1) > 0
                neg_tp = (
                    (occ_labels[valid_mask] == 0) & (pred_cls[valid_mask] == 0)
                ).sum()
                num_neg = (occ_labels[valid_mask] == 0).sum()
                num_pred_neg = (pred_cls[valid_mask] == 0).sum()
                pos_tp = (
                    (occ_labels[valid_mask] == 1) & (pred_cls[valid_mask] == 1)
                ).sum()
                num_pos = (occ_labels[valid_mask] == 1).sum()
                num_pred_pos = (pred_cls[valid_mask] == 1).sum()
                recall_neg = neg_tp / (num_neg + 1e-6)
                recall_pos = pos_tp / (num_pos + 1e-6)
                precision_neg = neg_tp / (num_pred_neg + 1e-6)
                precision_pos = pos_tp / (num_pred_pos + 1e-6)
                losses["recall_neg"] = recall_neg
                losses["recall_pos"] = recall_pos
                losses["precision_neg"] = precision_neg
                losses["precision_pos"] = precision_pos
        return losses

    def get_occ(
        self,
        local_roi_feats,
        rois,
        transform=True,
        ori_roi_feats=None,
    ):
        occ_list = self.occ_ae_head.get_occ(
            local_roi_feats,
            rois,
            transform=transform,
        )
        if ori_roi_feats is not None:
            ori_occ_list = self.occ_ae_head.get_occ(
                ori_roi_feats,
                rois,
                transform=transform,
            )
            new_occ_list = []
            for occs, ori_occs in zip(occ_list, ori_occ_list):
                new_occs = []
                for occ, ori_occ in zip(occs, ori_occs):
                    new_occ = torch.cat([occ, ori_occ], dim=0)
                    new_occs.append(new_occ)
                new_occ_list.append(new_occs)
            return new_occ_list

        return occ_list
    
    def transformer_forward(self, rois, roi_frame_inds, roi_feats, nonempty_roi_mask, trans_enc=None):
        if not self.training or self.train_cfg.get("fixed_length", True):
            return self.transformer_forward_fixed_length(
                rois, roi_frame_inds, roi_feats, nonempty_roi_mask,trans_enc)
        else:
            return self.transformer_forward_various_length(
                rois, roi_frame_inds, roi_feats, nonempty_roi_mask,trans_enc)
    def transformer_forward_fixed_length(self, rois, roi_frame_inds, roi_feats, nonempty_roi_mask, trans_enc=None):
        rois_batch_idx = rois[:, 0]

        B = int(rois_batch_idx.max().item() + 1)
        L = roi_frame_inds.numel() // B  # each batch contains L frames
        assert L * B == roi_frame_inds.numel()

        reorder_roi_feats, sorted_batch_idx, sorted_frame_idx = self.reorder_feats(
            roi_feats,
            roi_frame_inds,
            rois_batch_idx,
        )
        if self.debug:  # debug
            inv_feats = self.inverse_reorder_feats(
                reorder_roi_feats, sorted_batch_idx, sorted_frame_idx
            )
            stable_inv = (inv_feats == roi_feats).all()
            assert stable_inv
        reorder_frame_inds, _, _ = self.reorder_feats(
            roi_frame_inds.clone(),
            roi_frame_inds,
            rois_batch_idx,
            sorted_batch_idx,
            sorted_frame_idx,
        )
        reorder_frame_inds = reorder_frame_inds.squeeze(-1)


        reorder_roi_feats = reorder_roi_feats.view(
            B, L, reorder_roi_feats.shape[-1]
        ).permute(
            1, 0, 2
        )  # [L, B, d_model]
        pos_embed = self.pos_enc(reorder_frame_inds.transpose(0, 1))
        if self.with_roi_pos_encoding:
            reorder_rois_bbox,_,_ = self.reorder_feats(
                rois[:,1:], roi_frame_inds, rois_batch_idx, sorted_batch_idx, sorted_frame_idx
            )
            roi_embed = self.roi_pos_enc_mlp(reorder_rois_bbox).transpose(0, 1) # [L, B, d_model]
            pos_embed = pos_embed + roi_embed
        if not self.training and self.test_cfg.get("allow_attn_future", False):
            future_mask = None
        else:
            future_mask = self.get_future_mask(L, reorder_roi_feats.device)
        if trans_enc is None:
            trans_enc = self.trans_enc
        roi_feats = trans_enc(
            reorder_roi_feats,
            pos_enc=pos_embed,
            # key_padding_mask=torch.logical_not(
            #     reorder_valid_mask
            # ),  # invalid position should be True
            attn_mask=future_mask,
        ).transpose(
            0, 1
        )  # [B, L, d_model]
        roi_feats = self.inverse_reorder_feats(
            roi_feats, sorted_batch_idx, sorted_frame_idx
        )
        return roi_feats


    def transformer_forward_various_length(self, rois, roi_frame_inds, roi_feats, nonempty_roi_mask, trans_enc=None):
        # each trackletlet may have different length
        rois_batch_idx = rois[:, 0]

        B = int(rois_batch_idx.max().item() + 1)
        sorted_roi_feats_list = []
        sorted_rois_list = []
        reverse_frame_idx_list = []
        for b in range(B):
            roi_frame_inds_b = roi_frame_inds[rois_batch_idx == b] # [L]
            roi_feats_b = roi_feats[rois_batch_idx == b] # [L, D]
            frame_sorted_idx = roi_frame_inds_b.argsort()
            reverse_frame_idx = frame_sorted_idx.argsort() # use this to reverse the sorted feats to the original order
            sorted_feats = roi_feats_b[frame_sorted_idx]
            sorted_roi_feats_list.append(sorted_feats)
            reverse_frame_idx_list.append(reverse_frame_idx)
            if self.with_roi_pos_encoding:
                sorted_rois_b = rois[rois_batch_idx == b]
                sorted_rois_b = sorted_rois_b[frame_sorted_idx]
                sorted_rois_list.append(sorted_rois_b[:,1:]) # [L, 7]
        real_batch_size = len(sorted_roi_feats_list) # sometimes the real batch size is less than B because some batch may have no valid rois
        if real_batch_size < B:
            print(f"Some batch has no valid rois, real batch size is {real_batch_size}, original batch size is {B}")

        # pad to the same length
        max_len = max([len(x) for x in sorted_roi_feats_list])
        padded_sorted_roi_feats_list = []
        key_padding_list = []
        padded_sorted_rois_list = []
        for i in range(len(sorted_roi_feats_list)):

            sorted_feats = sorted_roi_feats_list[i]
            padded_sorted_feats = F.pad(sorted_feats, (0, 0, 0, max_len - len(sorted_feats)),"constant", 0) # [max_len, D]
            padded_sorted_roi_feats_list.append(padded_sorted_feats)
            key_padding = torch.zeros(max_len,device=sorted_feats.device, dtype=torch.bool)
            key_padding[len(sorted_feats):] = True
            key_padding_list.append(key_padding)
            if self.with_roi_pos_encoding:
                sorted_rois = sorted_rois_list[i]
                padded_sorted_rois = F.pad(sorted_rois, (0, 0, 0, max_len - len(sorted_rois)),"constant", 0) # [max_len, 7]
                padded_sorted_rois_list.append(padded_sorted_rois)
        regularized_roi_feats = torch.stack(padded_sorted_roi_feats_list, dim=0) # [B, max_len, D]

        if self.debug:
            roi_feats_list = []
            for i in range(real_batch_size):
                roi_feats_i = regularized_roi_feats[i]  # [max_len, d_model]
                unpadded_len = len(sorted_roi_feats_list[i])
                roi_feats_i = roi_feats_i[:unpadded_len]  # [L, d_model]
                reverse_frame_idx = reverse_frame_idx_list[i]
                roi_feats_i = roi_feats_i[reverse_frame_idx]
                roi_feats_list.append(roi_feats_i)
            roi_feats_inv = torch.cat(roi_feats_list, dim=0)
            assert (roi_feats_inv == roi_feats).all()
        # key padding mask for the transformer
        key_padding_mask = torch.stack(key_padding_list, dim=0) # [B, max_len]
        frame_inds = torch.arange(max_len, device=roi_frame_inds.device)[None, :].repeat(real_batch_size, 1) # [B, max_len]
        pos_embed = self.pos_enc(frame_inds.transpose(0, 1))
        if self.with_roi_pos_encoding:
            regularized_rois = torch.stack(padded_sorted_rois_list, dim=0) # [B, max_len, 7]
            roi_embed = self.roi_pos_enc_mlp(regularized_rois).transpose(0, 1) # [max_len, B, D]
            pos_embed = pos_embed + roi_embed
        regularized_roi_feats = regularized_roi_feats.permute(1, 0, 2) # [max_len, B, D]

        future_mask = self.get_future_mask(max_len, regularized_roi_feats.device)
        if trans_enc is None:
            trans_enc = self.trans_enc
        roi_feats = trans_enc(
            regularized_roi_feats,
            pos_enc=pos_embed,
            key_padding_mask=key_padding_mask,  # invalid position should be True
            attn_mask=future_mask,
        ).transpose(
            0, 1
        )  # [B, max_len, d_model]
        roi_feats_list = []
        for i in range(real_batch_size):
            roi_feats_i = roi_feats[i] # [max_len, d_model]
            unpadded_len = len(sorted_roi_feats_list[i])
            roi_feats_i = roi_feats_i[:unpadded_len] # [L, d_model]
            reverse_frame_idx = reverse_frame_idx_list[i]
            roi_feats_i = roi_feats_i[reverse_frame_idx]
            roi_feats_list.append(roi_feats_i)
        roi_feats = torch.cat(roi_feats_list, dim=0)
        return roi_feats

    def reorder_feats(
        self,
        feats,
        roi_frame_inds,
        roi_batch_inds,
        sort_batch_indices=None,
        sort_frame_indices=None,
    ):

        B = int(roi_batch_inds.max().item() + 1)
        L = roi_frame_inds.numel() // B
        if sort_batch_indices is None or sort_frame_indices is None:
            sort_batch_indices = torch.argsort(roi_batch_inds)
            roi_frame_inds = roi_frame_inds[sort_batch_indices]
            sort_frame_indices = torch.argsort(roi_frame_inds.view(B, L), dim=1)

        feats = feats[sort_batch_indices]
        feats = feats.view(B, L, -1)
        feats = torch.gather(
            feats, 1, sort_frame_indices[:, :, None].expand(-1, -1, feats.shape[-1])
        )

        return feats, sort_batch_indices, sort_frame_indices
    
    def get_future_mask(self, L, device,window_size=-1):
        # do not attend to the future
        if not self.training:
            window_size = self.test_cfg.get("attn_window_size", -1)
        mask = torch.ones(L, L, dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=1)
        if window_size > 0:
            for i in range(window_size-1,L):
                mask[i, :i-window_size+1] = 1
        return mask
    
    def inverse_reorder_feats(self, feats, sort_batch_indices, sort_frame_indices):
        B, L = sort_frame_indices.shape
        inverse_sort_frame_indices = torch.argsort(sort_frame_indices.view(B, L), dim=1)
        feats = feats.view(B, L, -1)
        feats = torch.gather(
            feats,
            1,
            inverse_sort_frame_indices[:, :, None].expand(-1, -1, feats.shape[-1]),
        )
        feats = feats.view(-1, feats.shape[-1])
        feats = feats[sort_batch_indices.argsort()]
        return feats

    def get_targets(
            self, sampling_results, rcnn_train_cfg, concat=True, transform_occ=True,num_occ_per_tracklet=-1
    ):
        """Generate targets.

        Args:
            sampling_results (list[:obj:`SamplingResult`]):
                Sampled results from rois.
            rcnn_train_cfg (:obj:`ConfigDict`): Training config of rcnn.
            concat (bool): Whether to concatenate targets between batches.

        Returns:
            tuple[torch.Tensor]: Targets of boxes and class prediction.
        """
        pos_bboxes_list = [
            res.pos_bboxes for res in sampling_results
        ]  # positive predictions
        pos_gt_bboxes_list = [
            res.pos_gt_bboxes for res in sampling_results
        ]  # positive ground truth
        iou_list = [res.iou for res in sampling_results]
        pos_label_list = [res.pos_gt_labels for res in sampling_results]
        occ_labels_list = [res.occ_labels for res in sampling_results]
        occ_scores_list = [res.occ_scores for res in sampling_results]

        # transform the occ xyz to rois frame

        targets = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            pos_gt_bboxes_list,
            iou_list,
            pos_label_list,
            occ_labels_list,
            occ_scores_list,
            cfg=rcnn_train_cfg,
            transform_occ=transform_occ,
            num_occ_per_tracklet=num_occ_per_tracklet,
        )

        (
            label,
            bbox_targets,
            pos_gt_bboxes,
            reg_mask,
            label_weights,
            bbox_weights,
            roi_local_xyz,
            gt_occ,
            occ_score,
            occ_reg_mask,
            pos_gt_bboxes_occ,
        ) = targets

        pos_gt_labels = pos_label_list
        bbox_target_batch_idx = []
        occ_target_batch_idx = []
        if concat:
            label = torch.cat(label, 0)
            bbox_target_batch_idx = torch.cat(
                [
                    t.new_ones(len(t), dtype=torch.int) * i
                    for i, t in enumerate(bbox_targets)
                ]
            )
            occ_target_batch_idx = torch.cat(
                [
                    t.new_ones(len(t), dtype=torch.int) * i
                    for i, t in enumerate(pos_gt_bboxes_occ)
                ]
            )
            bbox_targets = torch.cat(bbox_targets, 0)
            pos_gt_bboxes = torch.cat(pos_gt_bboxes, 0)
            pos_gt_labels = torch.cat(pos_gt_labels, 0)
            pos_gt_bboxes_occ = torch.cat(pos_gt_bboxes_occ, 0)
            reg_mask = torch.cat(reg_mask, 0)
            occ_reg_mask = torch.cat(occ_reg_mask, 0)

            label_weights = torch.cat(label_weights, 0)
            label_weights /= torch.clamp(label_weights.sum(), min=1.0)

            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_weights /= torch.clamp(bbox_weights.sum(), min=1.0)
            # remove None in the list then cat
            if len(pos_gt_bboxes_occ) > 0:
                pos_roi_local_xyz = torch.cat(
                    [e for e in roi_local_xyz if e is not None], 0
                )  # T_1 + T_2 +..T_B,K,3
                occ_score = torch.cat(
                    [e for e in occ_score if e is not None], 0
                )  # T_1 + T_2 +..T_B,
                gt_occ = torch.cat(
                    [e for e in gt_occ if e is not None], 0
                )  # T_1 + T_2 +..T_B, K,1
            else:
                pos_roi_local_xyz = pos_gt_bboxes.new_zeros(0, 0, 3)
                occ_score = pos_gt_bboxes.new_zeros(0)
                gt_occ = pos_gt_bboxes.new_zeros(0, 1)
        # print(len(pos_gt_bboxes),"gt")
        assert (reg_mask.sum() == len(pos_gt_bboxes)), f"{reg_mask.sum()}, {len(pos_gt_bboxes)}"
        assert (occ_reg_mask.sum() == len(pos_roi_local_xyz) == len(
            pos_gt_bboxes_occ)), f"{occ_reg_mask.sum()}, {len(pos_roi_local_xyz), len(pos_gt_bboxes_occ)}"

        return (
            label,
            bbox_targets,
            bbox_target_batch_idx,
            pos_gt_bboxes,
            pos_gt_labels,
            reg_mask,
            label_weights,
            bbox_weights,
            pos_roi_local_xyz,
            gt_occ,
            occ_score,
            occ_reg_mask,
            occ_target_batch_idx,
            pos_gt_bboxes_occ,
        )

    def _get_target_single(
            self,
            pos_bboxes,
            pos_gt_bboxes,
            ious,
            pos_labels,
            occ_label,
            occ_score,
            cfg,
            transform_occ=True,
            num_occ_per_tracklet=-1,  # -1 means all,
    ):
        """Generate training targets for a single sample.

        Args:
            pos_bboxes (torch.Tensor): Positive boxes with shape
                (N, 7).
            pos_gt_bboxes (torch.Tensor): Ground truth boxes with shape
                (M, 7).
            ious (torch.Tensor): IoU between `pos_bboxes` and `pos_gt_bboxes`
                in shape (N, M).
            occ_label (torch.Tensor): Occupancy label for this tracklet
            occ_score (torch.Tensor): Occupancy label confidence score for this tracklet,
            cfg (dict): Training configs.

            num_occ_per_tracklet (int): number of occupancy samples per tracklet, -1 means all

        Returns:
            tuple[torch.Tensor]: Target for positive boxes.
                (label, bbox_targets, pos_gt_bboxes, reg_mask, label_weights,
                bbox_weights)
        """
        assert pos_gt_bboxes.size(1) in (7, 9, 10)
        assert len(pos_gt_bboxes) == len(pos_bboxes)
        if pos_gt_bboxes.size(1) in (9, 10):
            pos_bboxes = pos_bboxes[:, :7]
            pos_gt_bboxes = pos_gt_bboxes[:, :7]

        label, label_weights = self.get_multi_class_soft_label(ious, pos_labels, cfg)

        # box regression target
        reg_mask = pos_bboxes.new_zeros(ious.size(0)).long()
        reg_mask[0: pos_gt_bboxes.size(0)] = 1
        bbox_weights = (reg_mask > 0).float()
        bbox_weights = self.get_class_wise_box_weights(bbox_weights, pos_labels, cfg)
        # occ_label.shape should be either K,4 or X_dim,Y_dim,Z_dim
        occ_reg_mask = torch.zeros_like(reg_mask)

        if reg_mask.bool().any():
            pos_gt_bboxes_ct = pos_gt_bboxes.clone().detach()
            roi_center = pos_bboxes[..., 0:3]
            roi_ry = pos_bboxes[..., 6] % (2 * np.pi)

            # canonical transformation
            pos_gt_bboxes_ct[..., 0:3] -= roi_center
            pos_gt_bboxes_ct[..., 6] -= roi_ry
            pos_gt_bboxes_ct[..., 0:3] = rotation_3d_in_axis(
                pos_gt_bboxes_ct[..., 0:3].unsqueeze(1), -(roi_ry + np.pi / 2), axis=2
            ).squeeze(1)

            # flip orientation if rois have opposite orientation
            ry_label = pos_gt_bboxes_ct[..., 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (
                    2 * np.pi
            )  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = ry_label > np.pi
            ry_label[flag] = ry_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
            pos_gt_bboxes_ct[..., 6] = ry_label

            rois_anchor = pos_bboxes.clone().detach()
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            bbox_targets = self.bbox_coder.encode(rois_anchor, pos_gt_bboxes_ct)

            assert len(pos_bboxes) == len(pos_gt_bboxes)
            if occ_label.dim() == 2:
                assert occ_label.size(1) == 4
                smp_pos = occ_label[:, 0:3]  # K,3
                gt_occ = occ_label[:, 3:4]  # K,1
            else:
                raise NotImplementedError

            with torch.no_grad():
                # for occ completion prediction, we may not want to regress occ for all positive rois,
                # to save memory and computation, we can randomly sample a subset of positive rois for occ regression
                num_gt = pos_gt_bboxes.size(0)
                if num_occ_per_tracklet > 0:
                    num_occ_per_tracklet = min(num_occ_per_tracklet, num_gt)
                else:
                    num_occ_per_tracklet = num_gt
                roi_local_xyz = smp_pos[None, ...].repeat(
                    num_occ_per_tracklet, 1, 1
                )  # N,K,3
                # selected_inds = torch.randperm(num_gt)[:num_occ_per_tracklet]
                # selected_inds = torch.sort(selected_inds)[0]
                selected_inds = torch.arange(num_gt)[-num_occ_per_tracklet:]
                pos_gt_bboxes_smp = pos_gt_bboxes[selected_inds]
                pos_bboxes_smp = pos_bboxes[selected_inds]
                occ_reg_mask[:num_gt][selected_inds] = 1
                if transform_occ:
                    # transform the sampled points from the gt box coordinate to the roi coordiante

                    # first transform from gt box coordiante to ego frame coordiante
                    roi_local_xyz = rotation_3d_in_axis(
                        roi_local_xyz, pos_gt_bboxes_smp[:, 6], axis=2
                    )
                    roi_local_xyz += pos_gt_bboxes_smp[..., None, 0:3]
                    # the occ label is the gravity center of the voxel, while the gt box is the bottom center
                    roi_local_xyz[..., 2] += pos_gt_bboxes_smp[:, None, 5] / 2

                    # then transform the smp_xyz from the ego frame to the roi frame
                    roi_local_xyz -= pos_bboxes_smp[..., None, :3]
                    roi_local_xyz[..., 2] -= pos_bboxes_smp[:, None, 5] / 2

                    # here we do not further rotate by 90 degree because the occ smp point is already rotated by 90 degree
                    roi_local_xyz = rotation_3d_in_axis(
                        roi_local_xyz, -(pos_bboxes_smp[:, 6]), axis=2
                    )  # N,K,3
                gt_occ = gt_occ[None].repeat(len(pos_gt_bboxes_smp), 1, 1)
                occ_score = occ_score.repeat(len(pos_gt_bboxes_smp)).float()
            # pdb.set_trace()
        else:
            # no fg bbox
            bbox_targets = pos_gt_bboxes.new_empty((0, 7))
            roi_local_xyz = None
            gt_occ = None
            # print(occ_score)
            occ_score = None
            pos_gt_bboxes_smp = pos_gt_bboxes.new_empty((0, 7))

        return (
            label,
            bbox_targets,
            pos_gt_bboxes,
            reg_mask,
            label_weights,
            bbox_weights,
            roi_local_xyz,
            gt_occ,
            occ_score,
            occ_reg_mask,
            pos_gt_bboxes_smp,
        )