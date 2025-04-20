import torch
import numpy as np

from mmcv.runner import BaseModule, force_fp32
from torch import nn as nn
import torch.nn.functional as F

from mmdet3d.core.bbox.structures import (
    LiDARInstance3DBoxes,
    rotation_3d_in_axis,
    xywhr2xyxyr,
)
from mmdet3d.models.builder import build_loss, build_roi_extractor
from mmdet3d.ops import scatter_v2, build_mlp
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from mmdet.models import HEADS

from mmdet3d.models.builder import build_roi_extractor
import torch.distributed as dist

from mmdet3d.ops.norm import AllReduce
from mmdet3d.ops.occ import occ_ops


class PosEncode(nn.Module):
    def __init__(self, L=10, bound=[-8.0, -8.0, -4.0, 8.0, 8.0, 4.0], use_norm=True):
        super().__init__()
        self.L = L
        self.norm_bound = bound
        self.use_norm = use_norm

    def forward(self, x):
        # x: N1, ..., D
        # return: N1, ..., D+2L

        assert x.size(-1) == 3
        if self.use_norm:
            min_bound = torch.tensor(self.norm_bound[:3], device=x.device)
            max_bound = torch.tensor(self.norm_bound[3:], device=x.device)
            x = (x - min_bound) / (
                max_bound - min_bound
            ) * 2.0 - 1.0  # normalize to [-1,1]

        ori_shape = x.shape[:-1] + (-1,)

        x = x.view(-1, x.size(-1))  # N, D
        x = x.unsqueeze(1)  # N, 1, D
        freq_bands = torch.pow(
            2, torch.linspace(0.0, self.L - 1, self.L, device=x.device)
        )  # L
        x = x * freq_bands.view(1, self.L, 1)  # N, L, D
        x = torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], dim=1)  # N, 2L, D
        x = x.view(*ori_shape)  # N1, ..., 2L*D
        return x


class OccDecoder(nn.Module):
    def __init__(
        self,
        roi_feature_channels,
        occ_mlp,
        use_positional_encoding=True,
        pos_encode_L=10,
        norm_pos=True,
        norm_cfg=dict(type="LN", eps=1e-3),
        act="gelu",
        occ_dropout=0.0,
        cls_dim=1,
        pos_thresh=0.5,
        use_ln=False,
    ):
        super().__init__()
        # add pos encoding
        if use_positional_encoding:
            self.pos_encode = PosEncode(L=pos_encode_L, use_norm=norm_pos)
            pos_enc_size = 2 * pos_encode_L * 3
        else:
            self.pos_encode = nn.Identity()
            pos_enc_size = 3
        self.cls_dim = cls_dim
        self.pos_thresh = pos_thresh
        assert cls_dim in (1, 2)
        if occ_mlp is not None:
            self.conv_occ = build_mlp(
                roi_feature_channels + pos_enc_size,  # condition on roi_feature
                occ_mlp + [self.cls_dim],
                norm_cfg,
                True,
                act=act,
                dropout=occ_dropout,
            )
        else:
            self.conv_occ = nn.Linear(roi_feature_channels, self.cls_dim)
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(roi_feature_channels)

    def forward(self, roi_features, smp_xyzs, pts_roi_inds):
        """
        Args:
            roi_features (torch.Tensor): roi features K,D
            smp_xyzs (torch.Tensor): sampled points N,3
            pts_roi_inds (torch.Tensor): sampled points' roi inds N, the value is from 0 to K-1
        """
        roi_feats_per_points = roi_features[pts_roi_inds]
        if self.use_ln:
            roi_feats_per_points = self.ln(roi_feats_per_points)
        occ_preds = self.conv_occ(
            torch.cat(
                [
                    roi_feats_per_points,
                    self.pos_encode(smp_xyzs),
                ],
                dim=-1,
            )
        )
        return occ_preds

    def occ_forward(self, roi_feats_per_points, smp_xyzs):
        """
        Args:
            roi_features (torch.Tensor): roi features K,D
            smp_xyzs (torch.Tensor): sampled points N,3
            pts_roi_inds (torch.Tensor): sampled points' roi inds N, the value is from 0 to K-1
        """
        
        if self.use_ln:
            roi_feats_per_points = self.ln(roi_feats_per_points)
        occ_preds = self.conv_occ(
            torch.cat(
                [
                    roi_feats_per_points,
                    self.pos_encode(smp_xyzs),
                ],
                dim=-1,
            )
        )
        return occ_preds

    def get_cls_from_pred(self, pred):
        """Get class from prediction.

        Args:
            pred (torch.Tensor): Prediction of shape (..., num_class).

        Returns:
            torch.Tensor: Class result.
        """
        if self.cls_dim == 1:
            return (pred.sigmoid() > self.pos_thresh).long().squeeze(-1)
        else:
            return pred.argmax(dim=-1)

    def get_roi_occ(
        self,
        roi_feats,
        rois,
        voxel_size,
        scale_wlh,
        offset_wlh,
        transform=True,
        return_score=False,
        random_sample_size=2048,
        occ_only=False):

        if roi_feats.size(0) == 0:
            return []
        assert roi_feats.size(0) == rois.size(0), f"{roi_feats.size(0)}, {rois.size(0)}"
        assert rois.size(1) in (8, 10)
        voxel_centers_list = occ_ops.generate_dense_voxel_centers(
                rois[:,4:7],
                voxel_size=voxel_size,
                scale_wlh=scale_wlh,
                offset_wlh=offset_wlh,
            )
        assert len(voxel_centers_list) == len(roi_feats)
        roi_inds_list = []
        occ_center_list = []
        if return_score:
            score_list = []
        bboxes_center = rois[:, 1:4]
        bboxes_sizes = rois[:, 4:7]
        bboxes_yaw = rois[:, 7]
        for j in range(len(voxel_centers_list)):
                voxel_centers = voxel_centers_list[j]

                K, _ = voxel_centers.shape
                roi_feat = roi_feats[j : j + 1]  # keep dim
                roi_feat_repeat = roi_feat.repeat(K, 1)

                occ_preds = self.occ_forward(roi_feat_repeat, voxel_centers)
                if not occ_only:
                    assert return_score
                    #randomly sample points from the voxel centers, including both occupied and free positions
                    if random_sample_size > 0:
                        selected = torch.randperm(K)[:random_sample_size]
                        selected_centers = voxel_centers[selected]
                    else:
                        selected = torch.ones((K,), device=voxel_centers.device,dtype=torch.bool)
                        selected_centers = voxel_centers

                else:
                    if self.cls_dim == 1:
                        selected = occ_preds.sigmoid().view(-1) > self.pos_thresh
                        selected_centers = voxel_centers[selected]  # K',3
                    else:
                        selected =  occ_preds[..., 1] > occ_preds[..., 0]
                        selected_centers = voxel_centers[selected]  # K',3
                        # occ_preds = occ_preds.softmax(dim=-1)
                        # selected_centers = voxel_centers[occ_preds[...,-1].view(-1)>0.55]  # K',3
                if return_score:
                    if self.cls_dim == 1:
                        score = occ_preds[selected].sigmoid().view(-1,1)
                    else:
                        score = occ_preds[selected].softmax(dim=-1)[...,1].view(-1,1)
                    score_list.append(score)
                if transform:
                    selected_centers = rotation_3d_in_axis(
                    selected_centers.unsqueeze(0),
                    bboxes_yaw[j : j + 1],  # keep dim
                    axis=2,
                    ).squeeze(0)
                    selected_centers += bboxes_center[j]
                    # the bboxes_center is the bottom center, while the occ origin is the gravity center of the bbox
                    # so we need to add the half of the bbox size to fix this
                    selected_centers[:, 2] += bboxes_sizes[j][2] / 2
                roi_inds = torch.full_like(selected_centers[:,0], j,dtype=torch.long)
                roi_inds_list.append(roi_inds)
                occ_center_list.append(selected_centers)
        if return_score:
            return torch.cat(occ_center_list, dim=0), torch.cat(roi_inds_list, dim=0), torch.cat(score_list, dim=0)
        return torch.cat(occ_center_list, dim=0), torch.cat(roi_inds_list, dim=0)
    

    def get_occ(
        self,
        roi_feats,
        rois,
        voxel_size,
        scale_wlh,
        offset_wlh,
        concat_batch=False,
        local_xyz=None,
        local_pts_roi_inds=None,
        return_full=False,
        transform=True,
    ):
        """Get occupancy prediction for each bbox, mainly used for test time visualization,
           The output occupancy are grouped by batch using lists

        Args:
            roi_feats (torch.Tensor): roi features N,D
            rois (torch.Tensor): rois with batch_id N,8 / N,10

        Returns:
            list [torch.Tensor] or list [list[torch.Tensor]] if concat_batch = True: occupancy prediction for each box
        """
        if roi_feats.size(0) == 0:
            return []
        assert roi_feats.size(0) == rois.size(0), f"{roi_feats.size(0)}, {rois.size(0)}"
        assert rois.size(1) in (8, 10)

        roi_batch_idx = rois[:, 0]
        bboxes = rois[:, 1:]
        # boxes without batch id
        batch_size = int(roi_batch_idx.max().item() + 1)

        res = []
        # generate voxel centers
        for i in range(batch_size):
            res_occ_centers = []
            curr_bboxes = bboxes[roi_batch_idx == i]
            cur_roi_feats = roi_feats[roi_batch_idx == i]
            bboxes_center = curr_bboxes[:, 0:3]
            bboxes_sizes = curr_bboxes[:, 3:6]
            bboxes_yaw = curr_bboxes[:, 6]
            voxel_centers_list = occ_ops.generate_dense_voxel_centers(
                bboxes_sizes,
                voxel_size=voxel_size,
                scale_wlh=scale_wlh,
                offset_wlh=offset_wlh,
            )
            assert len(voxel_centers_list) == len(bboxes_center)
            if local_xyz is not None:
                assert len(local_xyz) == len(
                    local_pts_roi_inds
                ), f"{len(local_xyz)}, {len(local_pts_roi_inds)}"
                local_centers_list = []
                rois_ids = torch.arange(rois.size(0), device=rois.device)
                cur_roi_ids = rois_ids[roi_batch_idx == i]
                for j in cur_roi_ids:
                    local_pts = local_xyz[local_pts_roi_inds == j]
                    local_centers_list.append(local_pts)
                assert len(local_centers_list) == len(bboxes_center)

            for j in range(len(voxel_centers_list)):
                voxel_centers = voxel_centers_list[j]
                if return_full:
                    occupied_centers = voxel_centers
                else:
                    K, _ = voxel_centers.shape
                    roi_feat = cur_roi_feats[j : j + 1]  # keep dim
                    roi_feat_repeat = roi_feat.repeat(K, 1)
                    # pos_roi_features_xyz = torch.cat(
                    #     [roi_feat_repeat, self.pos_encode(voxel_centers)], dim=-1
                    # )
                    # occ_preds = self.conv_occ(pos_roi_features_xyz).sigmoid()  # K,1
                    occ_preds = self.occ_forward(roi_feat_repeat, voxel_centers)
                    if self.cls_dim == 1:
                        occupied_centers = voxel_centers[
                            occ_preds.sigmoid().view(-1) > self.pos_thresh
                        ]  # K',3
                    else:
                        occupied_centers = voxel_centers[
                            occ_preds[..., 1] > occ_preds[..., 0]
                        ]  # K',3
                        # occ_preds = occ_preds.softmax(dim=-1)
                        # occupied_centers = voxel_centers[occ_preds[...,-1].view(-1)>0.55]  # K',3
                    if local_xyz is not None:
                        local_centers = local_centers_list[j]
                        # occupied_centers = torch.cat([occupied_centers, local_centers], dim=0)
                        occupied_centers = local_centers
                if transform:
                    # transform to LiDAR coordinate system
                    occupied_centers = rotation_3d_in_axis(
                        occupied_centers.unsqueeze(0),
                        bboxes_yaw[j : j + 1],  # keep dim
                        axis=2,
                    ).squeeze(0)
                    occupied_centers += bboxes_center[j]
                    # the bboxes_center is the bottom center, while the occ origin is the gravity center of the bbox
                    # so we need to add the half of the bbox size to fix this
                    occupied_centers[:, 2] += bboxes_sizes[j][2] / 2
                res_occ_centers.append(occupied_centers)
            if concat_batch:
                res.append(
                    torch.cat(res_occ_centers, dim=0)
                )  # concat the points with the same batch idx, yielding a single point clouds
            else:
                res.append(res_occ_centers)
        return res
