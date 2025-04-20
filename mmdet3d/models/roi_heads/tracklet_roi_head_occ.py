import copy
import pdb
import warnings
import numpy as np
import torch
from torch.nn import functional as F

from mmdet3d.core import AssignResult, PseudoSampler
from mmdet3d.core.bbox import bbox3d2result, bbox3d2roi, LiDARInstance3DBoxes
from mmdet.core import build_assigner, build_sampler
from mmdet.models import HEADS
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis

from mmdet3d.models.roi_heads.bbox_heads.fsd_bbox_head import FullySparseBboxHead
from ..builder import build_head, build_roi_extractor, build_backbone
from .base_3droi_head import Base3DRoIHead
from ipdb import set_trace


@HEADS.register_module()
class TrackletRoIHeadOCC(Base3DRoIHead):
    """Part aggregation roi head for PartA2.
    Args:
        semantic_head (ConfigDict): Config of semantic head.
        num_classes (int): The number of classes.
        seg_roi_extractor (ConfigDict): Config of seg_roi_extractor.
        part_roi_extractor (ConfigDict): Config of part_roi_extractor.
        bbox_head (ConfigDict): Config of bbox_head.
        train_cfg (ConfigDict): Training config.
        test_cfg (ConfigDict): Testing config.
    """

    def __init__(
        self,
        num_classes=3,
        roi_extractor=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        general_cfg=dict(),
        history_only=False,
    ):
        super(TrackletRoIHeadOCC,self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )
        self.general_cfg = general_cfg
        self.num_classes = num_classes
        self.with_roi_scores = general_cfg.get("with_roi_scores", False)
        self.with_roi_corners = general_cfg.get("with_roi_corners", False)
        self.checkpointing = general_cfg.get("checkpointing", False)

        self.roi_extractor = build_roi_extractor(roi_extractor)

        self.init_assigner_sampler()

        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be setting at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is a deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        self.history_only = history_only

    def init_bbox_head(self, bbox_head):
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.train_cfg = self.train_cfg
        self.bbox_head.test_cfg = self.test_cfg

    def init_mask_head(self):
        pass
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_sampler = PseudoSampler()
        if self.train_cfg is not None:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)

    def forward_train(
        self,
        pts_xyz,
        pts_feats,
        pts_batch_idx,
        pts_frame_inds,
        img_metas,
        tracklet_list,
        gt_candidates_list,
        gt_occs_list,
        gt_occ_scores_list,
    ):
        losses = dict()
        # note cd: sample_results is a list of SamplingResult, SamplingResult defines the one2one mapping between each tracklet box and GT
        sample_results = self._assign_and_sample(
            tracklet_list,
            gt_candidates_list,
            gt_occs_list,
            gt_occ_scores_list,
            pts_batch_idx,
            pts_frame_inds,
        )

        bbox_results = self._bbox_forward_train(
            pts_xyz, pts_feats, pts_batch_idx, pts_frame_inds, sample_results
        )

        losses.update(bbox_results["loss_bbox"])

        return losses

    def test_occ_baseline(
        self,
        pts_xyz,
        pts_feats,
        pts_batch_idx,
        pts_frame_inds,
        img_metas,
        tracklet_list,
        gt_candidates_list=None,
        gt_occs_list=None,
        gt_occ_scores_list=None,
        **kwargs,
    ):
        """
        test the shape completion performance by simpling aggregating points in the track
        """
        if gt_candidates_list is not None:
            gt_tracklet_list, gt_occ_list, gt_occ_score_list = (
                self._select_one2one_candidates(
                    tracklet_list, gt_candidates_list, gt_occs_list, gt_occ_scores_list
                )
            )
            # for each box within a tracklet, select the corresponding gt box at the same timestamp. If no gt box at that timestamp, return a zero
            # [num_rois, 1 + 7], the first dimension is the mask indicating the whehter it is a valid gt box
            gt_rois = self.get_gt_rois(tracklet_list, gt_tracklet_list)
            assert (
                len(gt_candidates_list)
                == len(gt_occ_list)
                == len(gt_occ_score_list)
                == 1
            ), "only support batch size 1"
        else:
            gt_rois = None
        assert len(tracklet_list) == 1, "only support batch size 1"
        rois, roi_frame_inds, cls_preds, labels_3d = self.tracklets2rois(tracklet_list)
        assert rois[:, 0].max().item() + 1 == len(
            tracklet_list
        ), "make sure there is no empty tracklet"
        
        match_mask = gt_rois[:, 0] == 1
        if (
            gt_occ_list[0] is None
            or not match_mask.any()
            or gt_occ_score_list[0] < self.bbox_head.occ_label_thresh
        ):
            # inter = union = torch.zeros((1,),device=rois.device,dtype=rois.dtype)
            inters = []
            unions = []
        else:
            ext_pts_inds, ext_pts_roi_inds, ext_pts_info = self.roi_extractor(
                pts_xyz[:, :3],  # intensity might be in pts_xyz
                pts_batch_idx,
                pts_frame_inds,
                rois[:, :8],
                roi_frame_inds,
            )

            local_xyz = ext_pts_info["local_xyz"]
            # fixed coordinate system mismatch
            compensate_r = local_xyz.new_tensor((np.pi / 2,))
            local_xyz = rotation_3d_in_axis(
                local_xyz[None, :, :], compensate_r, axis=2
            ).squeeze(0)
            assert ext_pts_roi_inds.max() + 1 <= len(
                rois
            ), f"{ext_pts_roi_inds.max()} vs {len(rois)}"
            inters = []
            unions = []
            # inter = union = 0.0
            occ_xyz = gt_occ_list[0][..., :3]  # K,3
            K = occ_xyz.size(0)
            occ_label = gt_occ_list[0][..., 3]  # K,1
            occ_smp_label = (occ_label == 1).long()
            for i in range(len(rois)):
                if match_mask[i] != 1:
                    continue
                local_xyz_till_i = local_xyz[ext_pts_roi_inds <= i]
                gt_roi_i = gt_rois[:, 1:][i]
                roi_i = rois[:, 1:][i]
                if self.test_cfg.get("transform_to_gt", True):
                    with torch.no_grad():
                        # first transform from gt box coordiante to ego frame coordiante
                        occ_smp_xyz = rotation_3d_in_axis(
                            occ_xyz[None], gt_roi_i[None, 6], axis=2
                        )
                        occ_smp_xyz += gt_roi_i[None, None, 0:3]
                        # the occ label is the gravity center of the voxel, while the gt box is the bottom center
                        occ_smp_xyz[..., 2] += gt_roi_i[None, None, 5] / 2
                        # occ_smp_xyz[...,2] += occ_sizes[:,None, 2]/2

                        # then transform the smp_xyz from the ego frame to the roi frame
                        occ_smp_xyz -= roi_i[None, None, :3]
                        occ_smp_xyz[..., 2] -= roi_i[None, None, 5] / 2

                        occ_smp_xyz = rotation_3d_in_axis(
                            occ_smp_xyz, -(roi_i[None, 6]), axis=2
                        )
                occ_smp_xyz = occ_smp_xyz.squeeze(0)
                bbox_size = roi_i[3:6]  # 3,
                ae_voxel_size = self.bbox_head.occ_ae_head.voxel_size
                voxel_dims = torch.ceil(bbox_size / ae_voxel_size).to(
                    torch.int32
                )  # 3; [x_size, y_size, z_size]
                occ = torch.zeros(
                    (voxel_dims[0], voxel_dims[1], voxel_dims[2]),
                    dtype=torch.bool,
                    device=occ_smp_xyz.device,
                )  # x_size,
                roi_min_bound = -bbox_size / 2
                occ_voxel_coors = torch.floor(
                    (local_xyz_till_i - roi_min_bound[None,]) / ae_voxel_size
                ).to(torch.long)
                # some points might be outside the box
                in_box_mask = (occ_voxel_coors >= 0).all(dim=1) & (
                    occ_voxel_coors < voxel_dims[None]
                ).all(dim=1)
                occ_voxel_coors = occ_voxel_coors[in_box_mask]
                gt_occ_voxel_coors = torch.floor(
                    (occ_smp_xyz - roi_min_bound[None,]) / ae_voxel_size
                ).to(torch.long)
                in_box_gt_mask = (gt_occ_voxel_coors >= 0).all(dim=1) & (
                    gt_occ_voxel_coors < voxel_dims[None]
                ).all(dim=1)
                pred_occ = torch.zeros_like(occ_smp_label)

                if in_box_gt_mask.any():
                    occ[
                        occ_voxel_coors[:, 0],
                        occ_voxel_coors[:, 1],
                        occ_voxel_coors[:, 2],
                    ] = 1
                    in_box_gt_occ_voxel_coors = gt_occ_voxel_coors[in_box_gt_mask]
                    pred_occ[in_box_gt_mask] = occ[
                        in_box_gt_occ_voxel_coors[:, 0],
                        in_box_gt_occ_voxel_coors[:, 1],
                        in_box_gt_occ_voxel_coors[:, 2],
                    ].long()

                # inter += ((occ_preds_cls == 1) & (occ_smp_label == 1)).sum()
                # union += ((occ_preds_cls == 1) | (occ_smp_label == 1)).sum()
                inters.append(
                    ((pred_occ == 1) & (occ_smp_label == 1))
                    .sum(dim=-1, keepdims=True)
                    .cpu()
                )
                unions.append(
                    ((pred_occ == 1) | (occ_smp_label == 1))
                    .sum(dim=-1, keepdims=True)
                    .cpu()
                )
        return [dict(inters=inters, unions=unions)]
    
    def test_occ( 
            self,
            rois,
            fused_roi_feats,
            gt_rois,
            gt_occ_list,
            gt_occ_score_list,
            pts_xyz,
            pts_batch_idx,
            pts_frame_inds,
            roi_frame_inds,
        ):

        match_mask = gt_rois[:, 0] == 1
        occ_rois = rois
        if (
            gt_occ_list[0] is None
            or not match_mask.any()
            or gt_occ_score_list[0] < self.bbox_head.occ_label_thresh
        ):
            # inter = union = torch.zeros((1,),device=rois.device,dtype=rois.dtype)
            inters = []
            unions = []
            gt_boxes = []
        elif self.test_cfg.get("test_baseline", False):
            #test the shape completion performance by simpling aggregating points in the track
            ext_pts_inds, ext_pts_roi_inds, ext_pts_info = self.roi_extractor(
                pts_xyz[:, :3],  # intensity might be in pts_xyz
                pts_batch_idx,
                pts_frame_inds,
                rois[:, :8],
                roi_frame_inds,
            )

            local_xyz = ext_pts_info["local_xyz"]
            # fixed coordinate system mismatch
            compensate_r = local_xyz.new_tensor((np.pi / 2,))
            local_xyz = rotation_3d_in_axis(
                local_xyz[None, :, :], compensate_r, axis=2
            ).squeeze(0)
            assert ext_pts_roi_inds.max() + 1 <= len(
                rois
            ), f"{ext_pts_roi_inds.max()} vs {len(rois)}"
            inters = []
            unions = []
            gt_boxes = []
            occ_xyz = gt_occ_list[0][..., :3]  # K,3
            K = occ_xyz.size(0)
            occ_label = gt_occ_list[0][..., 3]  # K,1
            occ_smp_label = (occ_label == 1).long()
            for i in range(len(rois)):
                if match_mask[i] != 1:
                    continue
                local_xyz_till_i = local_xyz[ext_pts_roi_inds <= i]
                gt_roi_i = gt_rois[i, 1:]
                roi_i = rois[:, 1:][i]
                if self.test_cfg.get("transform_to_gt", True):
                    with torch.no_grad():
                        # first transform from gt box coordiante to ego frame coordiante
                        occ_smp_xyz = rotation_3d_in_axis(
                            occ_xyz[None], gt_roi_i[None, 6], axis=2
                        )
                        occ_smp_xyz += gt_roi_i[None, None, 0:3]
                        # the occ label is the gravity center of the voxel, while the gt box is the bottom center
                        occ_smp_xyz[..., 2] += gt_roi_i[None, None, 5] / 2
                        # occ_smp_xyz[...,2] += occ_sizes[:,None, 2]/2

                        # then transform the smp_xyz from the ego frame to the roi frame
                        occ_smp_xyz -= roi_i[None, None, :3]
                        occ_smp_xyz[..., 2] -= roi_i[None, None, 5] / 2

                        occ_smp_xyz = rotation_3d_in_axis(
                            occ_smp_xyz, -(roi_i[None, 6]), axis=2
                        )
                occ_smp_xyz = occ_smp_xyz.squeeze(0)
                bbox_size = roi_i[3:6]  # 3,
                ae_voxel_size = self.bbox_head.occ_ae_head.voxel_size
                voxel_dims = torch.ceil(bbox_size / ae_voxel_size).to(
                    torch.int32
                )  # 3; [x_size, y_size, z_size]
                occ = torch.zeros(
                    (voxel_dims[0], voxel_dims[1], voxel_dims[2]),
                    dtype=torch.bool,
                    device=occ_smp_xyz.device,
                )  # x_size,
                roi_min_bound = -bbox_size / 2
                occ_voxel_coors = torch.floor(
                    (local_xyz_till_i - roi_min_bound[None,]) / ae_voxel_size
                ).to(torch.long)
                # some points might be outside the box
                in_box_mask = (occ_voxel_coors >= 0).all(dim=1) & (
                    occ_voxel_coors < voxel_dims[None]
                ).all(dim=1)
                occ_voxel_coors = occ_voxel_coors[in_box_mask]
                gt_occ_voxel_coors = torch.floor(
                    (occ_smp_xyz - roi_min_bound[None,]) / ae_voxel_size
                ).to(torch.long)
                in_box_gt_mask = (gt_occ_voxel_coors >= 0).all(dim=1) & (
                    gt_occ_voxel_coors < voxel_dims[None]
                ).all(dim=1)
                pred_occ = torch.zeros_like(occ_smp_label)

                if in_box_gt_mask.any():
                    occ[
                        occ_voxel_coors[:, 0],
                        occ_voxel_coors[:, 1],
                        occ_voxel_coors[:, 2],
                    ] = 1
                    in_box_gt_occ_voxel_coors = gt_occ_voxel_coors[in_box_gt_mask]
                    pred_occ[in_box_gt_mask] = occ[
                        in_box_gt_occ_voxel_coors[:, 0],
                        in_box_gt_occ_voxel_coors[:, 1],
                        in_box_gt_occ_voxel_coors[:, 2],
                    ].long()

                inters.append(
                    ((pred_occ == 1) & (occ_smp_label == 1))
                    .sum(dim=-1, keepdims=True)
                    .cpu()
                )
                unions.append(
                    ((pred_occ == 1) | (occ_smp_label == 1))
                    .sum(dim=-1, keepdims=True)
                    .cpu()
                ) 
                gt_boxes.append(gt_roi_i[None])
        else:
            chunk_size = self.test_cfg.get("iou_chunk_size", -1)
            if chunk_size == -1:
                chunk_size = len(match_mask)

            pred_rois_boxes = occ_rois[match_mask][:, 1:]
            pos_gt_bboxes = gt_rois[match_mask][:, 1:]
            pos_roi_feats = fused_roi_feats[match_mask]

            occ_xyz = gt_occ_list[0][..., :3]  # K,3
            K = occ_xyz.size(0)
            occ_label = gt_occ_list[0][..., 3]  # K,1

            match_mask_chunks = torch.split(match_mask, chunk_size, dim=0)
            pred_rois_boxes_chunks = torch.split(pred_rois_boxes, chunk_size, dim=0)
            pos_gt_bboxes_chunks = torch.split(pos_gt_bboxes, chunk_size, dim=0)
            pos_roi_feats_chunks = torch.split(pos_roi_feats, chunk_size, dim=0)
            # inter = union = 0.0
            inters = []
            unions = []
            gt_boxes = []
            for (
                pos_roi_feats,
                pred_rois_boxes,
                pos_gt_bboxes,
            ) in zip(
                pos_roi_feats_chunks,
                pred_rois_boxes_chunks,
                pos_gt_bboxes_chunks,
            ):

                occ_smp_xyz = occ_xyz[None, ...].repeat(
                    pos_gt_bboxes.size(0), 1, 1
                )  # N,K,3

                occ_smp_label = occ_label[None, ...].repeat(
                    pos_gt_bboxes.size(0), 1
                )
                occ_smp_label = (occ_smp_label == 1).long()
                if self.test_cfg.get("transform_to_gt", True):
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
                if self.test_cfg.get("ignore_outside_occ", False):
                    pred_box_size = pred_rois_boxes[:, 3:6]  # N,3
                    occ_smp_mask = (
                        occ_smp_xyz >= -pred_box_size[:, None, :] / 2
                    ).all(dim=-1) & (
                        occ_smp_xyz <= pred_box_size[:, None, :] / 2
                    ).all(
                        dim=-1
                    )  # N,K
                else:
                    occ_smp_mask = torch.ones(
                        (len(occ_smp_xyz), K),
                        device=occ_smp_xyz.device,
                        dtype=torch.bool,
                    )

                pos_roi_features_rp = pos_roi_feats[:, None, :].repeat(1, K, 1)
                occ_preds = self.bbox_head.occ_ae_head.occ_decoder.occ_forward(
                    pos_roi_features_rp, occ_smp_xyz
                )
                occ_preds_cls = (
                    self.bbox_head.occ_ae_head.occ_decoder.get_cls_from_pred(
                        occ_preds
                    )
                )
                occ_preds_cls = (
                    occ_preds_cls * occ_smp_mask
                )  # set the outside points to 0
                inters.append(
                    ((occ_preds_cls == 1) & (occ_smp_label == 1)).sum(dim=1).cpu()
                )
                unions.append(
                    ((occ_preds_cls == 1) | (occ_smp_label == 1)).sum(dim=1).cpu()
                )
                gt_boxes.append(pos_gt_bboxes.cpu())
        return dict(inters=inters, unions=unions, gt_boxes=gt_boxes)





    def simple_test(
        self,
        pts_xyz,
        pts_feats,
        pts_batch_idx,
        pts_frame_inds,
        img_metas,
        tracklet_list,
        gt_candidates_list=None,
        gt_occs_list=None,
        gt_occ_scores_list=None,
        **kwargs,
    ):
        """Simple testing forward function of TrackletRoIHeadOCC.
        Note:
            This function assumes that the batch size is 1
        Args:
            feats_dict (dict): Contains features from the first stage.
            voxels_dict (dict): Contains information of voxels.
            img_metas (list[dict]): Meta info of each image.
            proposal_list (list[dict]): Proposal information from rpn.
        Returns:
            dict: Bbox results of one frame.
        """
        # if self.test_cfg.get("test_baseline", False):
        #     # use point accumulation to get the occ
        #     return self.test_occ_baseline(
        #         pts_xyz,
        #         pts_feats,
        #         pts_batch_idx,
        #         pts_frame_inds,
        #         img_metas,
        #         tracklet_list,
        #         gt_candidates_list,
        #         gt_occs_list,
        #         gt_occ_scores_list,
        #         **kwargs,
        #     )
        if gt_candidates_list is not None:
            gt_tracklet_list, gt_occ_list, gt_occ_score_list = (
                self._select_one2one_candidates(
                    tracklet_list, gt_candidates_list, gt_occs_list, gt_occ_scores_list
                )
            )
            # for each box with a tracklet, select the corresponding gt box at the same timestamp. If no gt box at that timestamp, return a zero
            # [num_rois, 1 + 7], the first dimension is the mask indicating the whehter it is a valid gt box
            gt_rois = self.get_gt_rois(tracklet_list, gt_tracklet_list)
            assert (
                len(gt_candidates_list)
                == len(gt_occ_list)
                == len(gt_occ_score_list)
                == 1
            ), "only support batch size 1"
        else:
            gt_rois = None
        assert len(tracklet_list) == 1, "only support batch size 1"
        rois, roi_frame_inds, cls_preds, labels_3d = self.tracklets2rois(tracklet_list)
        assert rois[:, 0].max().item() + 1 == len(
            tracklet_list
        ), "make sure there is no empty tracklet"


        bbox_results = self._bbox_forward(
            pts_xyz,
            pts_feats,
            pts_batch_idx,
            pts_frame_inds,
            rois,
            cls_preds,
            roi_frame_inds,
        )

        cls_score = bbox_results["cls_score"]
        bbox_pred = bbox_results["bbox_pred"]
        fused_roi_feats = bbox_results["fused_roi_feats"]


        decoded_result_list = self.bbox_head.get_bboxes_from_tracklet(
            rois,
            cls_score,
            bbox_pred,
            bbox_results["nonempty_roi_mask"],
            labels_3d,
            cls_preds,
            img_metas,
            gt_rois=gt_rois,
            cfg=self.test_cfg,
        )
        new_rois = torch.cat(
            [rois[:, 0:1], self.bbox_head.decode_from_rois(rois, bbox_pred)], dim=-1
        )

        out_tracklets = []
        assert len(decoded_result_list) == len(tracklet_list)
        for i in range(len(tracklet_list)):
            old_trk = copy.deepcopy(tracklet_list[i])
            boxes, scores, labels, valid_mask = decoded_result_list[i]
            if self.test_cfg.get("tta", None) is not None:
                boxes = self.inverse_aug(old_trk, boxes, img_metas[i])
            
            to_ego = True  # should transform to ego frame for evaluation
            old_trk.update_from_prediction(
                boxes, scores, labels, valid_mask, to_ego=to_ego
            )
            out_tracklets.append(old_trk)
        out_dict = {"out_tracklets": out_tracklets}
        if self.test_cfg.get("test_occ_iou", False):
            iou_dict = self.test_occ( 
                            rois,
                            fused_roi_feats,
                            gt_rois,
                            gt_occ_list,
                            gt_occ_score_list,
                            pts_xyz,
                            pts_batch_idx,
                            pts_frame_inds,
                            roi_frame_inds)
            out_dict.update(iou_dict)
        return [out_dict]

    def save_occ_from_tracklet(
        self,
        tracklet_list,
        bbox_results,
        gt_tracklet_list=None,
        gt_occ_list=None,
        gt_occ_score_list=None,
        save_gt_occ=False,
        gt_score=None,
    ):
        assert len(tracklet_list) == 1, "only support batch size 1"
        tracklet = copy.deepcopy(tracklet_list[0])
        if len(tracklet.box_list) > 0:
            tracklet.box_list = [
                LiDARInstance3DBoxes(b).to(bbox_results["fused_roi_feats"].device)
                for b in tracklet.box_list
            ]
            tracklet.score_list = [
                torch.tensor(s, device=bbox_results["fused_roi_feats"].device)
                for s in tracklet.score_list
            ]
        rois, roi_frame_inds, cls_preds, labels_3d = self.tracklets2rois([tracklet])
        if save_gt_occ:
            assert len(gt_tracklet_list) == 1, "only support batch size 1"
            gt_tracklet = copy.deepcopy(gt_tracklet_list[0])
            if len(gt_tracklet.box_list) > 0:
                gt_tracklet.box_list = [
                    LiDARInstance3DBoxes(b).to(bbox_results["fused_roi_feats"].device)
                    for b in gt_tracklet.box_list
                ]
                gt_tracklet.score_list = [
                    torch.tensor(s, device=bbox_results["fused_roi_feats"].device)
                    for s in gt_tracklet.score_list
                ]
            ego_gt_rois, _, gt_mask_score, _ = self.tracklets2rois([gt_tracklet])
            if (not (gt_mask_score == 1).any()) or gt_occ_list[0] is None:
                # warnings.warn(f"no gt box/occ anno for this track {tracklet.id} in {tracklet.segment_name} at {tracklet.ts_list[0]}")
                # print(gt_occ_list[0] is None, (gt_mask_score==1).any())
                return
            occ_xyz = gt_occ_list[0][..., :3]  # K,3
            occ_label = gt_occ_list[0][..., 3]  # K,1
            occ_xyz = occ_xyz[occ_label == 1]
            if len(occ_xyz) == 0:
                return
            pos_gt_bboxes = ego_gt_rois[gt_mask_score == 1][:, 1:]
            pred_rois_boxes = rois[gt_mask_score == 1][:, 1:]
            assert pos_gt_bboxes.size(1) == pred_rois_boxes.size(1) == 7

            with torch.no_grad():
                occ_smp_xyz = occ_xyz[None, ...].repeat(
                    pos_gt_bboxes.size(0), 1, 1
                )  # N,K,3

                occ_smp_xyz = rotation_3d_in_axis(
                    occ_smp_xyz, pos_gt_bboxes[:, 6], axis=2
                )
                occ_smp_xyz += pos_gt_bboxes[..., None, 0:3]
                # the occ label is the gravity center of the voxel, while the gt box is the bottom center
                occ_smp_xyz[..., 2] += pos_gt_bboxes[:, None, 5] / 2
                # occ_smp_xyz[...,2] += occ_sizes[:,None, 2]/2

                # # then transform the smp_xyz from the ego frame to the roi frame
                # occ_smp_xyz -= pred_rois_boxes[..., None, :3]
                # occ_smp_xyz[..., 2] -= pred_rois_boxes[:, None, 5] / 2
                #
                # occ_smp_xyz = rotation_3d_in_axis(
                #     occ_smp_xyz, -(pred_rois_boxes[:, 6]), axis=2
                # )# N,K,3
            # print(occ_smp_xyz.shape,occ_smp_xyz.device,pred_rois_boxes.device,pred_rois_boxes[...,None,:].shape)
            box_idxs_of_pts = points_in_boxes_gpu(
                occ_smp_xyz, pred_rois_boxes[..., None, :]
            )
            occ_centers = []
            for i in range(len(occ_smp_xyz)):
                idxs = box_idxs_of_pts[i]
                occ_center = occ_smp_xyz[i][idxs > -1]
                # occ_center = occ_smp_xyz[i]
                occ_centers.append(occ_center)
                # print(occ_smp_xyz[i].shape,occ_center.shape)
            score_list = torch.ones((len(occ_centers),), device=occ_smp_xyz.device)
            ts_list = gt_tracklet.ts_list
            ts_list = [
                ts_list[i] for i in range(len(ts_list)) if (gt_mask_score == 1)[i]
            ]
            assert len(occ_centers) == len(
                ts_list
            ), f"{len(occ_centers)} != {len(ts_list)}"

            assert tracklet.segment_name == gt_tracklet.segment_name
            assert tracklet.id == gt_tracklet.id
            assert tracklet.type == gt_tracklet.type
        else:
            occ_centers_list = self.bbox_head.get_occ(
                bbox_results["fused_roi_feats"],
                rois,
                transform=True,  # visualize in the canonical coordiante system
            )

            occ_centers = occ_centers_list[0]
            score_list = tracklet.score_list
            ts_list = tracklet.ts_list
            if gt_score is not None:
                assert len(score_list) == len(
                    gt_score
                ), f"{len(score_list)} != {len(gt_score)}"
                score_list = gt_score

        segment_name = tracklet.segment_name
        trk_id = tracklet.id
        trk_type = tracklet.type
        nonempty_roi_mask = bbox_results["nonempty_roi_mask"]
        for i in range(len(occ_centers)):
            if i < self.test_cfg.get("min_evaluate_length", 0):
                continue
            if (
                self.test_cfg.get("filter_empty_roi", False)
                and not nonempty_roi_mask[i]
            ):
                print(f"empty roi {i} in {segment_name} at {ts_list[i]}")
                continue
            score = score_list[i].item()
            occ_center = occ_centers[i]
            occ_center = occ_center.cpu().numpy()
            occ_center = np.pad(
                occ_center, ((0, 0), (0, 1)), mode="constant", constant_values=score
            )
            save_occ_path = (
                f"{self.test_cfg.occ_save_root}/{segment_name}/{ts_list[i]}/"
            )
            os.makedirs(save_occ_path, exist_ok=True)
            occ_center.astype(np.float32).tofile(
                os.path.join(save_occ_path, f"{trk_type}_{trk_id}.bin")
            )

    def inverse_aug(self, trk, boxes, meta):
        if meta["pcd_horizontal_flip"]:
            boxes.flip("horizontal")
            trk.flip("horizontal")
        if meta["pcd_vertical_flip"]:
            boxes.flip("vertical")
            trk.flip("vertical")
        if "pcd_rot_angle" in meta:
            assert trk.rot_angle == meta["pcd_rot_angle"]
            boxes.rotate(-meta["pcd_rot_angle"])
            trk.rotate(-meta["pcd_rot_angle"])
        return boxes

    def _bbox_forward_train(
        self, pts_xyz, pts_feats, pts_batch_idx, pts_frame_inds, sampling_results
    ):
        rois = bbox3d2roi(
            [res.bboxes for res in sampling_results]
        )  # note cd: [num_rois, 1 + 7 ], add one dimension for batch_idx
        roi_frame_inds = torch.cat(
            [res.bboxes_frame_inds for res in sampling_results]
        )  # note cd: [num_rois, 1] indicates each box's frame id
        roi_scores = torch.cat(
            [res.scores for res in sampling_results]
        )  # note cd: [num_rois, 1] indicates each box's pre-trained detection score

        bbox_results = self._bbox_forward(
            pts_xyz,
            pts_feats,
            pts_batch_idx,
            pts_frame_inds,
            rois,
            roi_scores,
            roi_frame_inds,
        )

        bbox_targets = self.bbox_head.get_targets(
            sampling_results,
            self.train_cfg,
            transform_occ=self.train_cfg.get("transform_occ_pre", False),
            num_occ_per_tracklet=self.train_cfg.get("num_occ_per_tracklet", -1),
        )

        loss_bbox = self.bbox_head.loss(
            bbox_results,
            rois,
            *bbox_targets,
            transform_occ=not self.train_cfg.get("transform_occ_pre", False),
            roi_frame_inds=roi_frame_inds,
        )
        # tracklet_debug
        # pos_rois = bbox3d2roi([res.pos_bboxes for res in sampling_results])
        # pos_gt_bboxes = LiDARInstance3DBoxes(torch.cat([res.pos_gt_bboxes for res in sampling_results], 0))
        # reg_targets = bbox_targets[1]
        # decode_gts = self.bbox_head.decode_from_rois(pos_rois, reg_targets)
        # ********** there are some yaw flip, maybe a bug, pay attention**********
        labels = bbox_targets[0].view(-1)
        labels = labels > 0.5
        cls_score = bbox_results["cls_score"].view(-1)
        cls_preds = cls_score.sigmoid().detach() > 0.5
        acc = (cls_preds == labels).float().mean().detach()
        precision_posbox = ((cls_preds == 1) & (labels == 1)).float().sum() / (
            (cls_preds == 1).float().sum().detach() + 1e-6
        )
        recall_posbox = ((cls_preds == 1) & (labels == 1)).float().sum() / (
            (labels == 1).float().sum().detach() + 1e-6
        )
        precision_negbox = ((cls_preds == 0) & (labels == 0)).float().sum() / (
            (cls_preds == 0).float().sum().detach() + 1e-6
        )
        recall_negbox = ((cls_preds == 0) & (labels == 0)).float().sum() / (
            (labels == 0).float().sum().detach() + 1e-6
        )
        loss_bbox["acc"] = acc
        loss_bbox["precision_posbox"] = precision_posbox
        loss_bbox["recall_posbox"] = recall_posbox
        loss_bbox["precision_negbox"] = precision_negbox
        loss_bbox["recall_negbox"] = recall_negbox

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, pts_xyz, pts_feats, pts_batch_idx, pts_frame_inds, rois, roi_scores, roi_frame_inds):

        assert pts_xyz.size(0) == pts_feats.size(0) == pts_batch_idx.size(0) == pts_frame_inds.size(0)

        ext_pts_inds, ext_pts_roi_inds, ext_pts_info = self.roi_extractor(
            pts_xyz[:, :3],  # intensity might be in pts_xyz
            pts_batch_idx,
            pts_frame_inds,
            rois[:, :8],
            roi_frame_inds,
        )

        new_pts_feats = pts_feats[ext_pts_inds]
        new_pts_xyz = pts_xyz[ext_pts_inds]

        if self.roi_extractor.combined:
            new_pts_frame_inds = pts_frame_inds[ext_pts_inds]
            roi_frame_inds_per_pts = roi_frame_inds[ext_pts_roi_inds]
            is_cur_frame = (new_pts_frame_inds == roi_frame_inds_per_pts).to(new_pts_feats.dtype)
            new_pts_feats = torch.cat([new_pts_feats, is_cur_frame.unsqueeze(1)], 1)
            if self.history_only:
                history_mask = (new_pts_frame_inds <= roi_frame_inds_per_pts)
                new_pts_feats = new_pts_feats[history_mask]
                new_pts_xyz = new_pts_xyz[history_mask]
                ext_pts_info = {k: v[history_mask] for k, v in ext_pts_info.items()}
                ext_pts_inds = ext_pts_inds[history_mask]
                ext_pts_roi_inds = ext_pts_roi_inds[history_mask]

        if self.with_roi_scores:
            pts_scores = roi_scores[ext_pts_roi_inds]
            new_pts_feats = torch.cat([new_pts_feats, pts_scores.unsqueeze(1)], 1)
        # def forward(self, pts_xyz, pts_features, pts_info, roi_inds, rois):

        if self.with_roi_corners:
            corners = LiDARInstance3DBoxes(rois[:, 1:]).corners.to(pts_feats.dtype)  # [num_rois, 8, 3]
            centers = rois[:, :3]
            corners = torch.cat([corners, centers[:, None, :]], 1)
            corners_per_pts = corners[ext_pts_roi_inds]
            offsets = corners_per_pts - new_pts_xyz[:, None, :]
            offsets = offsets.reshape(len(offsets), 27) / 10
            new_pts_feats = torch.cat([new_pts_feats, offsets], 1)

        bbox_results = self.bbox_head(
            new_pts_xyz,
            new_pts_feats,
            ext_pts_info,
            ext_pts_roi_inds,
            rois,
            roi_frame_inds,
        )
        return bbox_results

    def _assign_and_sample(
        self,
        tracklet_list,
        candidates_list,
        gt_occs_list,
        gt_occ_scores_list,
        pts_batch_idx,
        pts_frame_inds,
    ):
        """Assign and sample proposals for training.
        Args:
            proposal_list (list[dict]): Proposals produced by RPN.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels
        Returns:
            list[:obj:`SamplingResult`]: Sampled results of each training
                sample.
        """
        assert (
            len(tracklet_list)
            == len(candidates_list)
            == len(gt_occs_list)
            == len(gt_occ_scores_list)
        )
        (
            gt_tracklet_list,
            gt_occ_list,
            gt_occ_score_list,
        ) = self._select_one2one_candidates(
            tracklet_list, candidates_list, gt_occs_list, gt_occ_scores_list
        )
        assert (
            len(tracklet_list)
            == len(gt_tracklet_list)
            == len(gt_occ_list)
            == len(gt_occ_score_list)
        )

        sampling_results = []
        # bbox assign
        for tid in range(len(tracklet_list)):
            trk_pd = tracklet_list[tid]
            trk_gt = gt_tracklet_list[tid]
            pts_frame_inds_in_trk = pts_frame_inds[pts_batch_idx == tid]
            cur_boxes = trk_pd.concated_boxes()
            cur_gt_bboxes = trk_gt.concated_boxes()

            assign_result = self.bbox_assigner.assign(trk_pd, trk_gt)
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                cur_boxes.tensor,
                cur_gt_bboxes.tensor,
            )
            reorder_inds = torch.cat(
                [sampling_result.pos_inds, sampling_result.neg_inds]
            )
            sampling_result.iou = assign_result.max_overlaps[reorder_inds].detach()
            if self.train_cfg.get("keep_frame_inds", True):
                # keep original frame inds instead of starting from 0 to L
                bboxes_frame_inds = torch.unique(pts_frame_inds_in_trk)
                assert len(bboxes_frame_inds) <= len(cur_boxes)
                if len(bboxes_frame_inds) < len(cur_boxes):
                    # padding is used
                    warnings.warn("Padding is used in frame inds")
                    pad_len = len(cur_boxes) - len(bboxes_frame_inds)
                    bboxes_frame_inds = torch.cat(
                        [bboxes_frame_inds, bboxes_frame_inds[-1:].repeat(pad_len)]
                    )
                    # print(bboxes_frame_inds)

                sampling_result.bboxes_frame_inds = bboxes_frame_inds[reorder_inds]
            else:
                # # keep frame inds starting from 0 to L
                # pts_frame_inds[pts_batch_idx == tid] -= pts_frame_inds_in_trk.min()
                if self.train_cfg.get("random_shift_frame_inds", False):
                    # randomly shift the frame inds
                    num_boxes = len(cur_boxes)
                    max_num = 200

                    shift = torch.randint(0, max_num - num_boxes + 1, (1,)).item()
                    pts_frame_inds[pts_batch_idx == tid] += shift
                    sampling_result.bboxes_frame_inds = (
                        torch.arange(
                            len(cur_boxes), device=trk_pd.device, dtype=torch.long
                        )[reorder_inds]
                        + shift
                    )

                else:
                    sampling_result.bboxes_frame_inds = torch.arange(
                        len(cur_boxes), device=trk_pd.device, dtype=torch.long
                    )[reorder_inds]
            sampling_result.scores = assign_result.scores[reorder_inds]

            assert isinstance(
                self.bbox_sampler, PseudoSampler
            ), "bboxes_frame_inds is corrent only if using PseudoSampler"
            # assign each gt track an occ label
            sampling_result.occ_labels = gt_occ_list[tid]
            sampling_result.occ_scores = gt_occ_score_list[tid]

            if sampling_result.pos_gt_bboxes.size(1) == 4 and self.train_cfg.get(
                "hack_sampler_bug", False
            ):
                assert sampling_result.pos_gt_bboxes.size(0) == 0
                sampling_result.pos_gt_bboxes = sampling_result.pos_gt_bboxes.new_zeros(
                    (0, 7)
                )

            sampling_results.append(sampling_result)
        return sampling_results

    def _select_one2one_candidates(
        self, tracklet_list, candidates_list, gt_occs_list, gt_occ_scores_list
    ):
        if self.train_cfg is not None:
            candidate_thresh = self.train_cfg.get("candidate_thresh", 0.5)
        else:
            candidate_thresh = self.test_cfg.get("candidate_thresh", 0.5)
        out_trks = []
        out_gt_occs = []
        out_gt_occ_scores = []
        for trk, candidates, gt_occs, gt_occ_scores in zip(
            tracklet_list, candidates_list, gt_occs_list, gt_occ_scores_list
        ):
            if len(candidates) == 0:
                out_trks.append(trk.new_empty())
                out_gt_occs.append(None)
                out_gt_occ_scores.append(None)
                continue
            affinities = torch.tensor(
                [
                    (trk.intersection_ious(c) > candidate_thresh).sum()
                    for c in candidates
                ]
            )
            if self.train_cfg is not None and self.train_cfg.get(
                "merge_candidates", False
            ):
                merged_candidate = self._merge_candidates(candidates, affinities)
                out_trks.append(merged_candidate)
            else:
                argmax = torch.argmax(affinities).item()
                out_trks.append(candidates[argmax])
                out_gt_occs.append(gt_occs[argmax])
                out_gt_occ_scores.append(gt_occ_scores[argmax])
        return out_trks, out_gt_occs, out_gt_occ_scores

    def _merge_candidates(self, candidates, priority):
        candidates_len = [len(c) for c in candidates]
        candidates = [
            first
            for first, second in sorted(
                zip(candidates, priority), key=lambda pair: -pair[1]
            )
        ]
        new_len = [len(c) for c in candidates]
        base = candidates[0]
        for c in candidates[1:]:
            base.merge_not_exist(c)
        return base

    def tracklets2rois(self, tracklets):
        rois = bbox3d2roi([t.concated_boxes().tensor for t in tracklets])
        cls_preds = torch.cat([t.concated_scores() for t in tracklets])
        labels_3d = torch.cat([t.concated_labels() for t in tracklets])
        # during the testing the tracklets are not cut, so the frame_inds are just the range of the tracklet
        roi_frame_inds = torch.cat(
            [
                torch.arange(len(t), device=rois.device, dtype=torch.long)
                for t in tracklets
            ]
        )

        assert tracklets[0].type_format == "mmdet3d"
        assert (labels_3d <= 2).all(), "Holds in WOD"
        return rois, roi_frame_inds, cls_preds, labels_3d

    def convert_result_to_tracklet(self, tracklet, result):
        bboxes, scores, labels = result

    def get_gt_rois(self, tracklets, gt_tracklets):
        assert len(tracklets) == len(gt_tracklets)
        out_boxes_list = []
        out_mask_list = []
        for i in range(len(tracklets)):
            trk = tracklets[i]
            gt = gt_tracklets[i]
            boxes, mask = gt.concated_boxes_from_ts(trk.ts_list)
            out_boxes_list.append(boxes)
            out_mask_list.append(mask)
        gt_rois = torch.cat(out_boxes_list, 0)
        mask = torch.cat(out_mask_list, 0)
        gt_rois = torch.cat([mask[:, None], gt_rois], 1)
        return gt_rois
