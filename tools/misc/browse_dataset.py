import argparse
import pdb

import numpy as np
import warnings

import torch
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress
from os import path as osp

from mmdet3d.core.bbox import (
    Box3DMode,
    CameraInstance3DBoxes,
    Coord3DMode,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
)
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis
from mmdet3d.core.bbox.box_np_ops import rotation_3d_in_axis as rotation_3d_in_axis_np
from mmdet3d.core.visualizer import (
    show_multi_modality_result,
    show_result,
    show_seg_result,
)
from mmdet3d.datasets import build_dataset
from ipdb import set_trace


def parse_args():
    parser = argparse.ArgumentParser(description="Browse a dataset")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--skip-type",
        type=str,
        nargs="+",
        default=["Normalize"],
        help="skip some useless pipeline",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="If there is no display interface, you can save it",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["det", "seg", "multi_modality-det", "mono-det", "occ", "track"],
        help="Determine the visualization method depending on the task.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Whether to perform online visualization. Note that you often "
        "need a monitor to do so.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


def build_data_cfg(config_path, skip_type, cfg_options):
    """Build data config for loading visualization data."""
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # extract inner dataset of `RepeatDataset` as `cfg.data.train`
    # so we don't need to worry about it later
    if cfg.data.train["type"] == "RepeatDataset":
        cfg.data.train = cfg.data.train.dataset
    train_data_cfg = cfg.data.train
    # eval_pipeline purely consists of loading functions
    # use eval_pipeline for data loading
    train_data_cfg["pipeline"] = [
        x for x in cfg.eval_pipeline if x["type"] not in skip_type
    ]

    return cfg


def to_depth_mode(points, bboxes):
    """Convert points and bboxes to Depth Coord and Depth Box mode."""
    if points is not None:
        points = Coord3DMode.convert_point(
            points.copy(), Coord3DMode.LIDAR, Coord3DMode.DEPTH
        )
    if bboxes is not None:
        bboxes = Box3DMode.convert(bboxes.clone(), Box3DMode.LIDAR, Box3DMode.DEPTH)
    return points, bboxes


def show_det_data(idx, dataset, out_dir, filename, show=False):
    """Visualize 3D point cloud and 3D bboxes."""
    example = dataset.prepare_train_data(idx)
    points = example["points"]._data.numpy()
    try:
        # gt_bboxes = dataset.get_ann_info(idx)['gt_bboxes_3d'].tensor
        gt_bboxes = example["gt_bboxes_3d"]._data.tensor
    except Exception:  # list of gt_bboxes_3d
        # bug :should transform to reference pose
        # tracklets = dataset.get_ann_info(idx)
        # box_list = [b for t in tracklets for b in t.box_list]
        # gt_bboxes = LiDARInstance3DBoxes.cat(box_list).tensor
        box_list = example["tracklet"]._data.box_list
        box_list = [LiDARInstance3DBoxes(b) for b in box_list]
        gt_bboxes = LiDARInstance3DBoxes.cat(box_list).tensor
    if dataset.box_mode_3d != Box3DMode.DEPTH:
        points, gt_bboxes = to_depth_mode(points, gt_bboxes)
    show_result(
        points, gt_bboxes.clone(), None, out_dir, filename, show=show, snapshot=True
    )


def show_tracklet_data(idx, dataset, out_dir, filename, show=False,norm=True,seperate=False):
    # if "10203656353524179475_7625_000_7645_000" not in filename:
    #     return
    example = dataset.prepare_train_data(idx)
    points = example["points"]._data.numpy()
    try:
        # gt_bboxes = dataset.get_ann_info(idx)['gt_bboxes_3d'].tensor
        gt_bboxes = example["gt_bboxes_3d"]._data.tensor

    except Exception:  # list of gt_bboxes_3d
        # bug :should transform to reference pose
        # tracklets = dataset.get_ann_info(idx)
        # box_list = [b for t in tracklets for b in t.box_list]
        # gt_bboxes = LiDARInstance3DBoxes.cat(box_list).tensor
        box_list = example["tracklet"]._data.box_list
        box_list = [LiDARInstance3DBoxes(b) for b in box_list]
        gt_bboxes = LiDARInstance3DBoxes.cat(box_list).tensor
    info = dataset.get_data_info(idx)
    if "occ_infos" in info:
        # set_trace()
        if len(info["occ_infos"]) > 0:
            occ_info = info["occ_infos"][0]  # only consider the first one
            label_iou = occ_info["label_iou"]
            label_trk_len = occ_info["label_trk_length"]
            tracklet_box_mean_size = gt_bboxes[:, 3:6].mean(dim=0)
            box_volume = tracklet_box_mean_size.prod().item()
            filename = filename + f"_iou{label_iou}_len{label_trk_len}_vol{box_volume}_{tracklet_box_mean_size[0]}_{tracklet_box_mean_size[1]}_{tracklet_box_mean_size[2]}"
    if norm:
        pts_frame_inds = example["pts_frame_inds"]._data.numpy()
        norm_points_list = []
        norm_boxes_list = []
        for i, bbox in enumerate(example["tracklet"]._data.box_list):
            points_i = points[pts_frame_inds == i]
            bbox = bbox.reshape(1, -1)
            norm_points_i = points_i[:, :3] - bbox[:, :3]
            norm_points_i[:, 2] = norm_points_i[:, 2] - bbox[0, 5] / 2
            norm_points_i = rotation_3d_in_axis_np(norm_points_i[None, ..., :3], -bbox[0, 6:7], axis=2)
            norm_points_i = norm_points_i[0]
            norm_bbox = np.zeros_like(bbox)
            norm_bbox[:, 3:] = bbox[:, 3:]
            norm_bbox[:, 2] -= bbox[0, 5] / 2
            norm_bbox[:, 6] = 0
            norm_points_list.append(norm_points_i)
            norm_boxes_list.append(norm_bbox)
            if seperate:

                show_result(
                    points_i, bbox, None, out_dir, filename, show=show, snapshot=True,suffix=f"_frame{i}"
                )
        points = np.concatenate(norm_points_list, axis=0)
        gt_bboxes = np.concatenate(norm_boxes_list, axis=0)
        gt_bboxes = torch.from_numpy(gt_bboxes)
    if not seperate:
        if dataset.box_mode_3d != Box3DMode.DEPTH:
            points, gt_bboxes = to_depth_mode(points, gt_bboxes)
        show_result(
            points, gt_bboxes.clone(), None, out_dir, filename, show=show, snapshot=True
    )


def show_occ_data(idx, dataset, out_dir, filename, show=False):
    """Visualize 3D point cloud and 3D bboxes."""
    example = dataset.prepare_train_data(idx)
    points = example["points"]._data.numpy()
    data_info = dataset.get_data_info(idx)
    print(data_info["occ_infos"])
    print(data_info["ann_info"]["track_id"])
    gt_bboxes = example["gt_bboxes_3d"]._data.tensor
    occ_labels = example["occ_labels"]._data
    occ_labels_scores = example["occ_labels_scores"]._data
    # print(gt_bboxes.shape,occ_labels.shape,occ_labels_scores.shape)
    occ_points = []
    for occ, occ_labels_score, box in zip(occ_labels, occ_labels_scores, gt_bboxes):
        if occ_labels_score > 0:
            print(len(occ), occ.shape, (occ[:, -1] == 1).sum(), (occ[:, -1] == 2).sum())
            occ = occ[occ[..., -1] == 1]  # filter out empty points
            occ_xyz = occ[..., :3]

            occupied_centers = rotation_3d_in_axis(
                occ_xyz.unsqueeze(0), box[6:7], axis=2  # keep dim
            ).squeeze(0)
            occupied_centers += box[:3].view(1, 3)
            # the bboxes_center is the bottom center, while the occ origin is the gravity center of the bbox
            # so we need to add the half of the bbox size to fix this
            occupied_centers[:, 2] += box[5] / 2
            occ_points.append(occupied_centers.numpy())

    occ_points = np.concatenate(occ_points, axis=0)
    if dataset.box_mode_3d != Box3DMode.DEPTH:
        points, gt_bboxes = to_depth_mode(points, gt_bboxes)
        occ_points, _ = to_depth_mode(occ_points, None)
    show_result(
        points,
        gt_bboxes.clone(),
        None,
        out_dir,
        filename,
        show=show,
        snapshot=True,
        occ_points=occ_points,
    )


def show_seg_data(idx, dataset, out_dir, filename, show=False):
    """Visualize 3D point cloud and segmentation mask."""
    example = dataset.prepare_train_data(idx)
    points = example["points"]._data.numpy()
    gt_seg = example["pts_semantic_mask"]._data.numpy()
    show_seg_result(
        points,
        gt_seg.copy(),
        None,
        out_dir,
        filename,
        np.array(dataset.PALETTE),
        dataset.ignore_index,
        show=show,
        snapshot=True,
    )


def show_proj_bbox_img(idx, dataset, out_dir, filename, show=False, is_nus_mono=False):
    """Visualize 3D bboxes on 2D image by projection."""
    try:
        example = dataset.prepare_train_data(idx)
    except AttributeError:  # for Mono-3D datasets
        example = dataset.prepare_train_img(idx)
    gt_bboxes = dataset.get_ann_info(idx)["gt_bboxes_3d"]
    img_metas = example["img_metas"]._data
    img = example["img"]._data.numpy()
    # need to transpose channel to first dim
    img = img.transpose(1, 2, 0)
    # no 3D gt bboxes, just show img
    if gt_bboxes.tensor.shape[0] == 0:
        gt_bboxes = None
    if isinstance(gt_bboxes, DepthInstance3DBoxes):
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            None,
            out_dir,
            filename,
            box_mode="depth",
            img_metas=img_metas,
            show=show,
        )
    elif isinstance(gt_bboxes, LiDARInstance3DBoxes):
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            img_metas["lidar2img"],
            out_dir,
            filename,
            box_mode="lidar",
            img_metas=img_metas,
            show=show,
        )
    elif isinstance(gt_bboxes, CameraInstance3DBoxes):
        # TODO: remove the hack of box from NuScenesMonoDataset
        if is_nus_mono:
            from mmdet3d.core.bbox import mono_cam_box2vis

            gt_bboxes = mono_cam_box2vis(gt_bboxes)
        show_multi_modality_result(
            img,
            gt_bboxes,
            None,
            img_metas["cam_intrinsic"],
            out_dir,
            filename,
            box_mode="camera",
            img_metas=img_metas,
            show=show,
        )
    else:
        # can't project, just show img
        warnings.warn(f"unrecognized gt box type {type(gt_bboxes)}, only show image")
        show_multi_modality_result(img, None, None, None, out_dir, filename, show=show)


def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = build_data_cfg(args.config, args.skip_type, args.cfg_options)
    try:
        dataset = build_dataset(
            cfg.data.train, default_args=dict(filter_empty_gt=False)
        )
    except TypeError:  # seg dataset doesn't have `filter_empty_gt` key
        dataset = build_dataset(cfg.data.train)
    data_infos = dataset.data_infos
    dataset_type = cfg.dataset_type

    # configure visualization mode
    vis_task = args.task  # 'det', 'seg', 'multi_modality-det', 'mono-det'

    for idx, data_info in enumerate(track_iter_progress(data_infos)):
        if dataset_type in ["KittiDataset", "WaymoDataset"]:
            data_path = data_info["point_cloud"]["velodyne_path"]
        elif dataset_type in [
            "ScanNetDataset",
            "SUNRGBDDataset",
            "ScanNetSegDataset",
            "S3DISSegDataset",
        ]:
            data_path = data_info["pts_path"]
        elif dataset_type in ["NuScenesDataset", "LyftDataset"]:
            data_path = data_info["lidar_path"]
        elif dataset_type in ["NuScenesMonoDataset"]:
            data_path = data_info["file_name"]
        elif dataset_type in ["WaymoTrackletDataset", "WaymoTrackletDatasetWithOcc"]:
            data_path = dataset.get_data_info(idx)["pts_filename"]
        elif dataset_type in ["MultiOccWaymoDataset"]:
            data_path = data_info["point_cloud"]["velodyne_path"]
        else:
            raise NotImplementedError(f"unsupported dataset type {dataset_type}")

        file_name = osp.splitext(osp.basename(data_path))[0]

        if vis_task in ["det", "multi_modality-det"]:
            # show 3D bboxes on 3D point clouds
            show_det_data(idx, dataset, args.output_dir, file_name, show=args.online)
        if vis_task in ["track"]:
            # show tracklet data
            show_tracklet_data(
                idx, dataset, args.output_dir, file_name, show=args.online
            )
        if vis_task in ["occ"]:
            # show 3D bboxes on 3D point clouds
            show_occ_data(idx, dataset, args.output_dir, file_name, show=args.online)
            break
        if vis_task in ["multi_modality-det", "mono-det"]:
            # project 3D bboxes to 2D image
            show_proj_bbox_img(
                idx,
                dataset,
                args.output_dir,
                file_name,
                show=args.online,
                is_nus_mono=(dataset_type == "NuScenesMonoDataset"),
            )
        elif vis_task in ["seg"]:
            # show 3D segmentation mask on 3D point clouds
            show_seg_data(idx, dataset, args.output_dir, file_name, show=args.online)


if __name__ == "__main__":
    main()
