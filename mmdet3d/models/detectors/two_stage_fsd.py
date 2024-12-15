import numpy as np
from .single_stage_fsd import SingleStageFSD
import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from .. import builder
from mmdet3d.core import Box3DMode, Coord3DMode, show_result
from mmcv.parallel import DataContainer as DC
from torch import nn
from os import path as osp
import mmcv
@DETECTORS.register_module()
class FSD(SingleStageFSD):

    def __init__(self,
                 backbone,
                 segmentor,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 neck=None,
                 bbox_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 cluster_assigner=None,
                 pretrained=None,
                 init_cfg=None,
                 train_detector=True,
                 train_occ_mlp=True,
                 train_bbox_head=False):
        super().__init__(
            backbone=backbone,
            segmentor=segmentor,
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            cluster_assigner=cluster_assigner,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

        # update train and test cfg here for now
        rcnn_train_cfg = train_cfg.rcnn if train_cfg else None
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        roi_head.pretrained = pretrained
        self.roi_head = builder.build_head(roi_head)
        self.num_classes = self.bbox_head.num_classes
        self.runtime_info = dict()
        self.train_detector = train_detector
        self.train_bbox_head = train_bbox_head
        if not train_detector and self.training:
            for name, p in self.named_parameters():
                if 'occ_head' in name:
                    if "conv_occ" in name and not train_occ_mlp:
                        print(f"freeze {name}")
                        p.requires_grad = False
                    elif "ln" in name and not train_occ_mlp:
                        print(f"freeze {name}")
                        p.requires_grad = False
                    else:
                        print(f"leave {name} to be trainable")
                        p.requires_grad = True
                elif 'roi_head' in name and "bbox_head" in name and ("conv_cls" in name or "conv_reg" in name):
                    if not train_bbox_head:
                        print(f"freeze {name}")
                        p.requires_grad = False
                    else:
                        print(f"leave {name} to be trainable")
                        p.requires_grad = True
                        
                else:
                    print(f"freeze {name}")
                    p.requires_grad = False

            
    @staticmethod
    def fixed_norm_and_droput(module):
        if (isinstance(module,nn.Dropout) or
            isinstance(module,nn.BatchNorm1d) or
            isinstance(module,nn.BatchNorm2d) or 
            isinstance(module,nn.BatchNorm3d) or 
            isinstance(module,nn.SyncBatchNorm)):
            module.eval()
            # print(module)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None,
                      **kwargs):
        if not self.train_detector:
            self.apply(self.fixed_norm_and_droput)
        if 'occ_labels' in kwargs:
            kwargs['occ_labels'] = [o[l>=0] for o, l in zip(kwargs['occ_labels'], gt_labels_3d)]
            kwargs['occ_labels_scores'] = [o[l>=0] for o, l in zip(kwargs['occ_labels_scores'], gt_labels_3d)]
        if 'occ_sizes' in kwargs:
            kwargs['occ_sizes'] = [o[l>=0] for o, l in zip(kwargs['occ_sizes'], gt_labels_3d)]
        gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l>=0] for l in gt_labels_3d]
        
        point_drop_ratio = self.train_cfg.get('point_drop_ratio', 0)
        if point_drop_ratio > 0:
            tmp_list = []
            for p in points:
                idx = torch.randperm(len(p)).to(p.device) # bug in torch1.8
                keep_num = int(len(p) * (1-point_drop_ratio))
                tmp_list.append(p[idx[:keep_num]])
            points = tmp_list

        losses = {}
        rpn_outs = super().forward_train(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_ignore=gt_bboxes_ignore,
            runtime_info=self.runtime_info,
        )
        if self.train_detector:
            losses.update(rpn_outs['rpn_losses'])

        proposal_list = self.bbox_head.get_bboxes(
            rpn_outs['cls_logits'], rpn_outs['reg_preds'], rpn_outs['cluster_xyz'], rpn_outs['cluster_inds'], img_metas
        )

        assert len(proposal_list) == len(gt_bboxes_3d)

        pts_xyz, pts_feats, pts_batch_inds = self.prepare_multi_class_roi_input(
            rpn_outs['all_input_points'],
            rpn_outs['valid_pts_feats'],
            rpn_outs['seg_feats'],
            rpn_outs['pts_mask'],
            rpn_outs['pts_batch_inds'],
            rpn_outs['valid_pts_xyz']
        )

        roi_losses = self.roi_head.forward_train(
            pts_xyz,
            pts_feats,
            pts_batch_inds,
            img_metas,
            proposal_list,
            gt_bboxes_3d,
            gt_labels_3d,
            **kwargs
        )

        losses.update(roi_losses)


        return losses
    
    def prepare_roi_input(self, points, cluster_pts_feats, pts_seg_feats, pts_mask, pts_batch_inds, cluster_pts_xyz):
        assert isinstance(pts_mask, list)
        pts_mask = pts_mask[0]
        assert points.shape[0] == pts_seg_feats.shape[0] == pts_mask.shape[0] == pts_batch_inds.shape[0]

        if self.training and self.train_cfg.get('detach_seg_feats', False):
            pts_seg_feats = pts_seg_feats.detach()

        if self.training and self.train_cfg.get('detach_cluster_feats', False):
            cluster_pts_feats = cluster_pts_feats.detach()
        
        pad_feats = cluster_pts_feats.new_zeros(points.shape[0], cluster_pts_feats.shape[1])
        pad_feats[pts_mask] = cluster_pts_feats
        assert torch.isclose(points[pts_mask], cluster_pts_xyz).all()

        cat_feats = torch.cat([pad_feats, pts_seg_feats], dim=1)

        return points, cat_feats, pts_batch_inds

    def prepare_multi_class_roi_input(self, points, cluster_pts_feats, pts_seg_feats, pts_mask, pts_batch_inds, cluster_pts_xyz):
        assert isinstance(pts_mask, list)
        bg_mask = sum(pts_mask) == 0
        assert points.shape[0] == pts_seg_feats.shape[0] == bg_mask.shape[0] == pts_batch_inds.shape[0]

        if self.training and self.train_cfg.get('detach_seg_feats', False):
            pts_seg_feats = pts_seg_feats.detach()

        if self.training and self.train_cfg.get('detach_cluster_feats', False):
            cluster_pts_feats = cluster_pts_feats.detach()


        ##### prepare points for roi head
        fg_points_list = [points[m] for m in pts_mask]
        all_fg_points = torch.cat(fg_points_list, dim=0)

        assert torch.isclose(all_fg_points, cluster_pts_xyz).all()

        bg_pts_xyz = points[bg_mask]
        all_points = torch.cat([bg_pts_xyz, all_fg_points], dim=0)
        #####

        ##### prepare features for roi head
        fg_seg_feats_list = [pts_seg_feats[m] for m in pts_mask]
        all_fg_seg_feats = torch.cat(fg_seg_feats_list, dim=0)
        bg_seg_feats = pts_seg_feats[bg_mask]
        all_seg_feats = torch.cat([bg_seg_feats, all_fg_seg_feats], dim=0)

        num_out_points = len(all_points)
        assert num_out_points == len(all_seg_feats)

        pad_feats = cluster_pts_feats.new_zeros(bg_mask.sum(), cluster_pts_feats.shape[1])
        all_cluster_pts_feats = torch.cat([pad_feats, cluster_pts_feats], dim=0)
        #####

        ##### prepare batch inds for roi head
        bg_batch_inds = pts_batch_inds[bg_mask]
        fg_batch_inds_list = [pts_batch_inds[m] for m in pts_mask]
        fg_batch_inds = torch.cat(fg_batch_inds_list, dim=0)
        all_batch_inds = torch.cat([bg_batch_inds, fg_batch_inds], dim=0)


        # pad_feats[pts_mask] = cluster_pts_feats

        cat_feats = torch.cat([all_cluster_pts_feats, all_seg_feats], dim=1)

        # sort for roi extractor
        all_batch_inds, inds = all_batch_inds.sort()
        all_points = all_points[inds]
        cat_feats = cat_feats[inds]

        return all_points, cat_feats, all_batch_inds
    
    def simple_test(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):


        rpn_outs = super().simple_test(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )

        proposal_list = rpn_outs['proposal_list']

        if self.test_cfg.get('skip_rcnn', False):
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in proposal_list
            ]
            return bbox_results

        if self.num_classes > 1 or self.test_cfg.get('enable_multi_class_test', False):
            prepare_func = self.prepare_multi_class_roi_input
        else:
            prepare_func = self.prepare_roi_input

        pts_xyz, pts_feats, pts_batch_inds = prepare_func(
            rpn_outs['all_input_points'],
            rpn_outs['valid_pts_feats'],
            rpn_outs['seg_feats'],
            rpn_outs['pts_mask'],
            rpn_outs['pts_batch_inds'],
            rpn_outs['valid_pts_xyz']
        )

        results = self.roi_head.simple_test(
            pts_xyz,
            pts_feats,
            pts_batch_inds,
            img_metas,
            proposal_list,
            gt_bboxes_3d,
            gt_labels_3d,
        )

        return results
    

    def extract_fg_by_gt(self, point_list, gt_bboxes_3d, gt_labels_3d, extra_width):
        if isinstance(gt_bboxes_3d[0], list):
            assert len(gt_bboxes_3d) == 1
            assert len(gt_labels_3d) == 1
            gt_bboxes_3d = gt_bboxes_3d[0]
            gt_labels_3d = gt_labels_3d[0]

        bsz = len(point_list)

        new_point_list = []
        for i in range(bsz):
            points = point_list[i]
            gts = gt_bboxes_3d[i].to(points.device)
            if len(gts) == 0:
                this_fg_mask = points.new_zeros(len(points), dtype=torch.bool)
                this_fg_mask[:min(1000, len(points))] = True
            else:
                if isinstance(extra_width, dict):
                    this_labels = gt_labels_3d[i]
                    enlarged_gts_list = []
                    for cls in range(self.num_classes):
                        cls_mask = this_labels == cls
                        if cls_mask.any():
                            this_enlarged_gts = gts[cls_mask].enlarged_box(extra_width[cls])
                            enlarged_gts_list.append(this_enlarged_gts)
                    enlarged_gts = gts.cat(enlarged_gts_list)
                else:
                    enlarged_gts = gts.enlarged_box(extra_width)
                pts_inds = enlarged_gts.points_in_boxes(points[:, :3])
                this_fg_mask = pts_inds > -1
                if not this_fg_mask.any():
                    this_fg_mask[:min(1000, len(points))] = True
            
            new_point_list.append(points[this_fg_mask])
        return new_point_list

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'

            pred_bboxes = result[batch_id]['boxes_3d']
            points = points[points[:,5] == 0] # remove points from previous timestamps
                
            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
                
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for convertion!')
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            
            show_result(points, None, pred_bboxes, out_dir, file_name, False)
            if 'occ_pred' in result[batch_id]:
                pred_occ = result[batch_id]['occ_pred']
                cat_occ = torch.cat(pred_occ, dim=0)
                cat_occ = cat_occ.cpu().numpy()
                # print(points.shape,cat_occ.shape)
                cat_occ = np.pad(cat_occ, ((0, 0), (0, points.shape[1]-cat_occ.shape[1])), mode='constant', constant_values=0)
                # points = np.concatenate([points, cat_occ], axis=0)
                cat_occ = Coord3DMode.convert_point(cat_occ, Coord3DMode.LIDAR,
                                                Coord3DMode.DEPTH)
                show_result(cat_occ, None, None, out_dir, file_name +"_occ", False)


            
