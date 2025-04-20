import pdb
import numpy as np
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
import mmcv
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, LiDARTracklet
from mmcv.parallel import DataContainer as DC
from os import path as osp
from mmdet3d.core import Box3DMode, Coord3DMode, show_result
from mmdet.models import build_detector
from mmdet3d.core.bbox.box_np_ops import rotation_3d_in_axis

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from ..builder import build_backbone, build_head, build_neck

from mmseg.models import SEGMENTORS
from .. import builder
from mmdet3d.ops import (
    scatter_v2,
    Voxelization,
    furthest_point_sample,
    get_inner_win_inds,
)
from scipy.sparse.csgraph import connected_components
from mmdet.core import multi_apply
from .base import Base3DDetector
from mmdet3d.models.segmentors.base import Base3DSegmentor

# from mmdet3d.utils import vis_bev_pc
from ipdb import set_trace


@DETECTORS.register_module()
class TrackletDetectorOCC(Base3DDetector):
    def __init__(
        self,
        roi_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super().__init__(
            init_cfg=init_cfg,
        )

        self.num_classes = roi_head["num_classes"]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cfg = self.train_cfg if self.train_cfg else self.test_cfg
        self.print_info = {}

        roi_head.update(train_cfg=train_cfg)
        roi_head.update(test_cfg=test_cfg)
        roi_head.pretrained = pretrained
        self.roi_head = builder.build_head(roi_head)

        # self.fake_linear = torch.nn.Linear(6, 6)

    def extract_feat(
        self,
    ):
        """
        For abstract class instantiate
        """
        pass

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            if len(res) == 0:
                print("***********Attention: Got zero-point input***********")
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(
                coor, (1, 0), mode="constant", value=i
            )  # prepend a batch index to the discretized coordinate
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def forward_train(
        self,
        points,
        pts_frame_inds=None,
        img_metas=None,
        tracklet=None,
        gt_tracklet_candidates=None,
        occ_labels=None,
        occ_labels_scores=None,
    ):
        losses = {}
        # losses['loss_fake'] = self.fake_linear(points[0]).mean()

        # After PointsRangeFilter, there might be very few empty input.
        points = self.fake_points_for_empty_input(points)
        pts_frame_inds = self.fake_points_for_empty_input(pts_frame_inds)

        self.from_collate_format(tracklet, gt_tracklet_candidates)
        self.tracklet_to_device(tracklet, gt_tracklet_candidates, points[0].device)
        # return losses
        # vis_bev_pc(points[0], None, name=f'tracklet_pc_test.png', dir='tracklet', pc_range=100)

        batch_idxs = []
        for i, p in enumerate(points):
            batch_idxs.append(
                torch.full((len(p),), i, dtype=torch.long, device=p.device)
            )
        cat_points = torch.cat(points, dim=0)
        if occ_labels is not None:
            losses = self.roi_head.forward_train(
                pts_xyz=cat_points[:, :3],
                pts_feats=cat_points[:, 3:],
                pts_batch_idx=torch.cat(batch_idxs, dim=0),
                pts_frame_inds=torch.cat(pts_frame_inds, dim=0),
                img_metas=img_metas,
                tracklet_list=tracklet,
                gt_candidates_list=gt_tracklet_candidates,
                gt_occs_list=occ_labels,
                gt_occ_scores_list=occ_labels_scores,
            )
        else:
            losses = self.roi_head.forward_train(
                pts_xyz=cat_points[:, :3],
                pts_feats=cat_points[:, 3:],
                pts_batch_idx=torch.cat(batch_idxs, dim=0),
                pts_frame_inds=torch.cat(pts_frame_inds, dim=0),
                img_metas=img_metas,
                tracklet_list=tracklet,
                gt_candidates_list=gt_tracklet_candidates,
            )

        return losses

    def simple_test(
        self,
        points,
        img_metas,
        pts_frame_inds,
        tracklet,
        gt_tracklet_candidates=None,
        occ_labels=None,
        occ_labels_scores=None,
        rescale=False,
        **kwargs,
    ):
        """Test function without augmentaiton."""
        # pdb.set_trace()
        points = self.fake_points_for_empty_input(points)
        pts_frame_inds = self.fake_points_for_empty_input(pts_frame_inds)

        self.from_collate_format(tracklet, gt_tracklet_candidates)
        self.tracklet_to_device(tracklet, gt_tracklet_candidates, points[0].device)

        batch_idxs = []
        for i, p in enumerate(points):
            batch_idxs.append(
                torch.full((len(p),), i, dtype=torch.long, device=p.device)
            )
        cat_points = torch.cat(points, dim=0)
        if self.test_cfg.get("visualize_occ", False):
            results, occs = self.roi_head.simple_test(
                pts_xyz=cat_points[:, :3],
                pts_feats=cat_points[:, 3:],
                pts_batch_idx=torch.cat(batch_idxs, dim=0),
                pts_frame_inds=torch.cat(pts_frame_inds, dim=0),
                img_metas=img_metas,
                tracklet_list=tracklet,
                gt_candidates_list=gt_tracklet_candidates,
            )
            return results, occs
        results = self.roi_head.simple_test(
            pts_xyz=cat_points[:, :3],
            pts_feats=cat_points[:, 3:],
            pts_batch_idx=torch.cat(batch_idxs, dim=0),
            pts_frame_inds=torch.cat(pts_frame_inds, dim=0),
            img_metas=img_metas,
            tracklet_list=tracklet,
            gt_candidates_list=gt_tracklet_candidates,
            gt_occs_list=occ_labels,
            gt_occ_scores_list=occ_labels_scores,
        )

        return results

    def aug_test(self, points, img_metas, pts_frame_inds, tracklet, rescale=False):
        """Test function with augmentaiton."""
        assert len(points) == len(img_metas) == len(pts_frame_inds) == len(tracklet)
        aug_result_list = []
        for p, meta, inds, trk in zip(points, img_metas, pts_frame_inds, tracklet):
            this_result = self.simple_test(p, meta, inds, trk)
            if self.test_cfg.get("visualize_occ", False):
                this_result = this_result[0]
            aug_result_list.append(this_result)
        bsz = len(points[0])
        num_augs = len(points)
        merged_result_list = []
        for i in range(bsz):
            aug_list_this_sample = []
            for k in range(num_augs):
                aug_list_this_sample.append(aug_result_list[k][i])
            this_merged_sample = LiDARTracklet.merge_augs(
                aug_list_this_sample, self.test_cfg["tta"], points[0][0].device
            )
            merged_result_list.append(this_merged_sample)

        return merged_result_list

    def pre_voxelize_within_frame(self, data_dict):
        batch_idx = data_dict["batch_idx"]
        frame_inds = data_dict["pts_frame_inds"]
        points = data_dict["points"]
        pts_feats = data_dict["pts_feats"]

        voxel_size = torch.tensor(
            self.cfg.pre_voxelization_size, device=batch_idx.device
        )
        pc_range = torch.tensor(self.cfg.point_cloud_range, device=points.device)
        coors = torch.div(
            points[:, :3] - pc_range[None, :3],
            voxel_size[None, :],
            rounding_mode="floor",
        ).long()
        coors = coors[:, [2, 1, 0]]  # to zyx order
        coors = torch.cat([batch_idx[:, None], frame_inds[:, None], coors], dim=1)

        new_coors, unq_inv = torch.unique(
            coors, return_inverse=True, return_counts=False, dim=0
        )

        out_dict = {}
        for data_name in data_dict:
            data = data_dict[data_name]
            if data.dtype in (torch.float, torch.float16):
                voxelized_data, voxel_coors = scatter_v2(
                    data,
                    coors,
                    mode="avg",
                    return_inv=False,
                    new_coors=new_coors,
                    unq_inv=unq_inv,
                )
                out_dict[data_name] = voxelized_data

        out_dict["batch_idx"] = voxel_coors[:, 0]
        out_dict["pts_frame_inds"] = voxel_coors[:, 1]
        return out_dict

    def split_by_batch(self, data, batch_idx, batch_size):
        assert batch_idx.max().item() + 1 <= batch_size
        data_list = []
        for i in range(batch_size):
            sample_mask = batch_idx == i
            data_list.append(data[sample_mask])
        return data_list

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        assert len(data_list) == batch_size
        if data_list[0] is None:
            return None
        data_shape = (len(batch_idx),) + data_list[0].shape[1:]
        full_data = data_list[0].new_zeros(data_shape)
        for i, data in enumerate(data_list):
            sample_mask = batch_idx == i
            full_data[sample_mask] = data
        return full_data

    def tracklet_to_device(self, tracklets, candidates_list, device):
        for t in tracklets:
            t.to(device)

        if candidates_list is not None:
            for t_list in candidates_list:
                for t in t_list:
                    t.to(device)

    def from_collate_format(self, tracklets, candidates_list):
        for t in tracklets:
            t.from_collate_format()
        if candidates_list is not None:
            for t_list in candidates_list:
                for t in t_list:
                    t.from_collate_format()

    def fake_points_for_empty_input(self, points_list):
        new_points_list = []
        for p in points_list:
            if len(p) == 0:
                print("Empty input occurs!!!")
                if p.ndim == 1:
                    new_data = p.new_zeros((1,))
                else:
                    new_data = p.new_zeros((1, p.size(1)))
                new_points_list.append(new_data)
            else:
                new_points_list.append(p)
        return new_points_list

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Override the one in super to support batch inference
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """

        for var, name in [(points, "points"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                "num of augmentations ({}) != num of image meta ({})".format(
                    len(points), len(img_metas)
                )
            )

        if self.test_cfg.get("tta", None) is not None:
            return self.aug_test(points, img_metas, **kwargs)
        else:
            img = [img] if img is None else img
            return self.simple_test(points, img_metas, **kwargs)
