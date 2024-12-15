import glob

import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import os
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from pdb import set_trace as st
import yaml
import torch
import torch.nn.functional as F
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.points import BasePoints, get_points_type
from collections import OrderedDict

import random
from scipy import signal
from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D
import warnings
from mmdet3d.datasets.pipelines.transforms_3d import (
    ObjectNameFilter,
    ObjectRangeFilter,
    RandomFlip3D,
)


@PIPELINES.register_module()
class LoadAnnotationsOcc(object):
    """Load Annotations Occ."""
    def __init__(self, compute_score=False):
        self.compute_score = compute_score
    def __call__(self, results):
        occ_infos = results["occ_infos"]
        # assert len(occ_infos) == len(results["gt_bboxes_3d"]), f'{len(occ_infos)},{len(results["gt_bboxes_3d"])}'
        occ_grids = []
        occ_scores = []  # how confident is the occ grid annotation
        occ_lengths = []  # how long is the tracklet used to annotate this occ grid
        for occ in occ_infos:
            score = occ["label_iou"]
            label_trk_length = occ["label_trk_length"]
            if occ["occ_label_name"] is None:
                # fake an empty grid
                occ_grids.append(torch.zeros(1, 1, 1, dtype=torch.int))
                score = 0.0
            else:
                try:
                    occ = torch.from_numpy(np.load(occ["occ_label_name"])["occ"])
                    occ_grids.append(occ)
                    if self.compute_score:
                        num_unknown = (occ == 0).sum().item()
                        num_known = occ.numel() - num_unknown
                        score = num_known / occ.numel()
                        # print(score)

                except FileNotFoundError:
                    # fake an empty grid
                    occ_grids.append(torch.zeros(1, 1, 1, dtype=torch.int))
                    # print(f"no occ label for this track with ctrl length {label_trk_length}")
                    score = 0.0
            occ_scores.append(score)
            occ_lengths.append(label_trk_length)
        if "gt_bboxes_3d" in results and len(results["gt_bboxes_3d"]) > len(occ_grids):
            # gt_bboxes_3d could be more than occ labels when copy paste is used
            # pad occ labels
            pad_length = len(results["gt_bboxes_3d"]) - len(occ_grids)
            # print(f'padding occ labels',len(results["gt_bboxes_3d"]), len(occ_grids))
            for _ in range(pad_length):
                occ_grids.append(torch.zeros(1, 1, 1, dtype=torch.int))
                occ_scores.append(0.0)
                occ_lengths.append(0)
        results["occ_label_list"] = occ_grids
        results["occ_scores"] = torch.tensor(occ_scores)
        results["occ_lengths"] = torch.tensor(occ_lengths, dtype=torch.int)
        return results


@PIPELINES.register_module()
class MirrorOccLabel(object):
    """Mirror the occ label along the x axis. Only affect the unknown voxels.
    In other words, we fill the unknown voxel with the occupancy value of the mirrored voxel.
    """

    def __call__(self, results):
        if "occ_label_list" in results:
            occ_grids = results["occ_label_list"]
            new_occ_grids = []
            for i, occ_grid in enumerate(occ_grids):
                XS, YS, ZS = occ_grid.shape
                occ_grid_flat = occ_grid.clone().view(-1)
                unknown_mask = occ_grid_flat == 0
                mid_size = XS // 2
                voxel_coors_x, voxel_coors_y, voxel_coors_z = torch.meshgrid(
                    torch.arange(
                        XS,
                        dtype=torch.long,
                    ),
                    torch.arange(
                        YS,
                        dtype=torch.long,
                    ),
                    torch.arange(
                        ZS,
                        dtype=torch.long,
                    ),
                )  # x_size, y_size, z_size, 3
                mirror_voxel_coors_x = (
                    (voxel_coors_x + 0.5 - mid_size) * -1.0 + mid_size
                ).long()

                xmirror_voxel_coors = torch.stack(
                    [mirror_voxel_coors_x, voxel_coors_y, voxel_coors_z], dim=-1
                ).view(-1, 3)

                occ_grid_flat[unknown_mask] = occ_grid[
                    xmirror_voxel_coors[unknown_mask][:, 0],
                    xmirror_voxel_coors[unknown_mask][:, 1],
                    xmirror_voxel_coors[unknown_mask][:, 2],
                ]
                new_occ_grids.append(occ_grid_flat.view(XS, YS, ZS))
        results["occ_label_list"] = new_occ_grids
        return results


@PIPELINES.register_module()
class RandomSampleOccPoints(object):
    """Sample a fixed number of occ points for training."""

    def __init__(
        self,
        num_sample_points=1024,
        pos_sample_weight=0.5,
        voxel_size=0.2,
        use_unknown=False,
        use_potential=False,
        mirror_x=False,
        balance_sample=False,
        weighted_sample=True,
    ):
        self.num_sample_points = num_sample_points
        self.pos_sample_weight = pos_sample_weight
        self.voxel_size = voxel_size
        self.use_unknown = use_unknown
        self.use_potential = use_potential
        if use_potential:
            self.potential = {}
        self.mirror_x = mirror_x
        self.balance_sample = balance_sample
        self.weighted_sample = weighted_sample

    def __call__(self, results):
        if "occ_label_list" in results:
            occ_infos = results["occ_infos"]
            occ_grids = results["occ_label_list"]  # list[tensor(X,Y,Z)]
            occ_scores = results["occ_scores"]  # tensor(K,)
            if len(occ_grids) == 0:
                if self.num_sample_points == -1:
                    num_sample_points = 0
                else:
                    num_sample_points = self.num_sample_points
                results["sample_occs"] = torch.zeros((0, num_sample_points))  # N,K
                results["sample_occ_centers"] = torch.zeros(
                    (0, num_sample_points, 3)
                )  # N,K,3
                results["occ_sizes"] = torch.zeros((0, 3))  # N,3
                return results
            sample_occs = []
            sample_occ_centers = []
            occ_sizes = []
            for i, (occ_grid, occ_score, info) in enumerate(
                zip(occ_grids, occ_scores, occ_infos)
            ):
                if not (occ_grid > 0).any():
                    # all voxel are unknown
                    # fake an empty grid
                    assert (
                        occ_score == 0
                    ), "occ_score should be 0 if no occ grid is annotated"
                    if self.num_sample_points == -1:
                        num_sample_points = 0
                    else:
                        num_sample_points = self.num_sample_points
                    sample_centered = torch.zeros(num_sample_points, 3)
                    sample_occ = torch.zeros(num_sample_points)
                    w, l, h = 0.0, 0.0, 0.0
                else:
                    XS, YS, ZS = occ_grid.shape
                    occ_grid_flat = occ_grid.view(-1)
                    voxel_coors_x, voxel_coors_y, voxel_coors_z = torch.meshgrid(
                        torch.arange(
                            XS,
                            dtype=torch.long,
                        ),
                        torch.arange(
                            YS,
                            dtype=torch.long,
                        ),
                        torch.arange(
                            ZS,
                            dtype=torch.long,
                        ),
                    )  # x_size, y_size, z_size, 3
                    if self.mirror_x:
                        # fill the unknowns with the mirrored voxel
                        unknown_mask = occ_grid_flat == 0
                        mid_size = XS // 2
                        mirror_voxel_coors_x = (
                            (voxel_coors_x + 0.5 - mid_size) * -1.0 + mid_size
                        ).long()
                        xmirror_voxel_coors = torch.stack(
                            [mirror_voxel_coors_x, voxel_coors_y, voxel_coors_z], dim=-1
                        ).view(-1, 3)
                        occ_grid_flat[unknown_mask] = occ_grid[
                            xmirror_voxel_coors[unknown_mask][:, 0],
                            xmirror_voxel_coors[unknown_mask][:, 1],
                            xmirror_voxel_coors[unknown_mask][:, 2],
                        ]
                    voxel_coors = torch.stack(
                        [voxel_coors_x, voxel_coors_y, voxel_coors_z], dim=-1
                    ).view(
                        -1, 3
                    )  # x_size* y_size* z_size, 3
                    if not self.use_unknown:
                        valid_voxel_coors = voxel_coors[
                            occ_grid_flat > 0
                        ]  # remove unknown voxels
                        valid_occ_grid = occ_grid_flat[occ_grid_flat > 0]
                    else:
                        valid_voxel_coors = voxel_coors
                        valid_occ_grid = occ_grid_flat.clone()
                    w, l, h = (
                        float(XS) * self.voxel_size,
                        float(YS) * self.voxel_size,
                        float(ZS) * self.voxel_size,
                    )

                    # center the coordinate system at the gravity center of the voxel grid
                    min_bound = torch.tensor(
                        [-w / 2, -l / 2, -h / 2], dtype=torch.float32
                    )

                    valid_voxel_centers = (
                        valid_voxel_coors.to(torch.float) * self.voxel_size
                        + min_bound
                        + self.voxel_size / 2
                    )  # K, 3,
                    if self.num_sample_points == -1:
                        #sample all valid voxels
                        sample_idx = torch.arange(len(valid_voxel_centers))
                    elif self.balance_sample:
                        num_pos = int(self.num_sample_points * self.pos_sample_weight)
                        num_neg = self.num_sample_points - num_pos
                        idxs = torch.arange(len(valid_occ_grid))
                        pos_idxs = idxs[valid_occ_grid == 1]
                        neg_idxs = idxs[valid_occ_grid != 1]
                        if len(pos_idxs) == 0 or len(neg_idxs) == 0:
                            sample_weights = torch.ones_like(
                                valid_occ_grid, dtype=torch.float
                            )
                            print(
                                f'no pos or neg voxels in {info["occ_label_name"]},pos {len(pos_idxs)}, neg {len(neg_idxs)}, score {occ_score}'
                            )
                            sample_idx = torch.multinomial(
                                sample_weights,
                                self.num_sample_points,
                                replacement=len(valid_occ_grid)
                                < self.num_sample_points,
                            )
                            occ_scores[i] = 0.0  # do not use this sample
                        else:
                            pos_choice = torch.multinomial(
                                torch.ones_like(pos_idxs, dtype=torch.float),
                                num_pos,
                                replacement=len(pos_idxs) < num_pos,
                            )
                            neg_choice = torch.multinomial(
                                torch.ones_like(neg_idxs, dtype=torch.float),
                                num_neg,
                                replacement=len(neg_idxs) < num_neg,
                            )
                            sample_idx = torch.cat(
                                [pos_idxs[pos_choice], neg_idxs[neg_choice]], dim=0
                            )
                    elif self.use_potential:
                        # print(valid_occ_grid.dtype)
                        potential = self.potential.get(
                            info["occ_label_name"],
                            torch.ones_like(valid_occ_grid, dtype=torch.float),
                        )
                        if len(valid_occ_grid) < self.num_sample_points:
                            sample_idx = torch.multinomial(
                                1 / potential, self.num_sample_points, replacement=True
                            )
                        else:
                            _, sample_idx = torch.topk(
                                potential, self.num_sample_points, dim=0, largest=False
                            )
                        potential[sample_idx] += 1
                        self.potential[info["occ_label_name"]] = potential
                    elif self.weighted_sample:
                        try:
                            sample_weights = torch.ones_like(valid_occ_grid) * (
                                1 - self.pos_sample_weight
                            )
                            sample_weights[valid_occ_grid == 1] = self.pos_sample_weight
                            # sample at known voxels
                            sample_idx = torch.multinomial(
                                sample_weights,
                                self.num_sample_points,
                                replacement=len(valid_occ_grid)
                                < self.num_sample_points,
                            )

                        except Exception as e:
                            # print(sample_weights.sum(),sample_weights.shape)
                            # print(info['occ_label_name'])
                            # raise e
                            # incase there is no sampled weights
                            sample_weights = torch.ones_like(
                                valid_occ_grid, dtype=torch.float
                            )
                            # sample at known voxels
                            sample_idx = torch.multinomial(
                                sample_weights,
                                self.num_sample_points,
                                replacement=len(valid_occ_grid)
                                < self.num_sample_points,
                            )
                    else:
                        #random sample at known voxels
                        sample_idx = torch.multinomial(
                            torch.ones_like(valid_occ_grid, dtype=torch.float),
                            self.num_sample_points,
                            replacement=len(valid_occ_grid) < self.num_sample_points,
                        )

                    sample_centered = valid_voxel_centers[sample_idx]  # K, 3
                    sample_occ = valid_occ_grid[sample_idx]  # K,

                sample_occs.append(sample_occ)
                sample_occ_centers.append(sample_centered)
                occ_sizes.append(torch.tensor([w, l, h], dtype=torch.float32))
            if self.num_sample_points != -1:
                # do not stack if we sample all voxels because of the different number of voxels
                results["sample_occs"] = torch.stack(sample_occs, dim=0)  # N,K
                results["sample_occ_centers"] = torch.stack(
                    sample_occ_centers, dim=0
                )  # N,K,3
            else:
                results["sample_occs"] = sample_occs
                results["sample_occ_centers"] = sample_occ_centers
            results["occ_sizes"] = torch.stack(occ_sizes, dim=0)  # N,3

        return results


@PIPELINES.register_module()
class JitterOccCenter(object):
    """
    jitter the occ center within the voxel
    """

    def __init__(self, voxel_size=0.2):
        self.voxel_size = voxel_size

    def __call__(self, results):
        sample_occ_centers = results["sample_occ_centers"]
        jitter_noise = (
            torch.rand_like(sample_occ_centers) * self.voxel_size - self.voxel_size / 2
        )  # [-voxel_size/2, voxel_size/2]
        results["sample_occ_centers"] = sample_occ_centers + jitter_noise
        return results


@PIPELINES.register_module()
class RandomFlip3DWithOcc(RandomFlip3D):
    def random_flip_data_3d(self, input_dict, direction="horizontal"):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ["horizontal", "vertical"]
        if len(input_dict["bbox3d_fields"]) == 0:  # test mode
            input_dict["bbox3d_fields"].append("empty_box3d")
            input_dict["empty_box3d"] = input_dict["box_type_3d"](
                np.array([], dtype=np.float32)
            )
        assert len(input_dict["bbox3d_fields"]) == 1
        for key in input_dict["bbox3d_fields"]:
            if "points" in input_dict:
                input_dict["points"] = input_dict[key].flip(
                    direction, points=input_dict["points"]
                )
            else:
                input_dict[key].flip(direction)
        if "centers2d" in input_dict:
            assert (
                self.sync_2d is True and direction == "horizontal"
            ), "Only support sync_2d=True and horizontal flip with images"
            w = input_dict["img_shape"][1]
            input_dict["centers2d"][..., 0] = w - input_dict["centers2d"][..., 0]

        if "seed_info" in input_dict:
            for seed in input_dict["seed_info"]:
                seed["gt_bboxes_3d"].flip(direction)

        if "sample_occ_centers" in input_dict:
            # the occ centers are all on the canonical coordinate system
            # fliping only affects the x axis , assuming x-axis points to the right, y-axis points to the front,
            # should alter this behavior if the occ label coordinates change
            input_dict["sample_occ_centers"][:, :, 0] = -input_dict[
                "sample_occ_centers"
            ][:, :, 0]


@PIPELINES.register_module()
class ObjectRangeFilterWithOcc(ObjectRangeFilter):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bev_range = self.pcd_range[[0, 1, 3, 4]]

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        sample_occ_centers = input_dict["sample_occ_centers"]
        sample_occs = input_dict["sample_occs"]
        occ_scores = input_dict["occ_scores"]
        occ_sizes = input_dict["occ_sizes"]
        occ_lengths = input_dict["occ_lengths"]
        mask = gt_bboxes_3d.in_range_bev(self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]

        sample_occ_centers = sample_occ_centers[mask]
        sample_occs = sample_occs[mask]
        occ_scores = occ_scores[mask]
        occ_sizes = occ_sizes[mask]
        occ_lengths = occ_lengths[mask]
        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d

        input_dict["sample_occ_centers"] = sample_occ_centers
        input_dict["sample_occs"] = sample_occs
        input_dict["occ_scores"] = occ_scores
        input_dict["occ_sizes"] = occ_sizes
        input_dict["occ_lengths"] = occ_lengths

        return input_dict


@PIPELINES.register_module()
class FilterOccByScoreAndLength(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, score_threshold=0.0, min_length=10):
        self.score_threshold = score_threshold
        self.min_length = min_length

    def __call__(self, input_dict):
        """Call function to filter occ labels by the occ scores. 
        The corresponding gt_bboxes_3d and gt_labels_3d are also filtered.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        sample_occ_centers = input_dict["sample_occ_centers"]
        sample_occs = input_dict["sample_occs"]
        occ_scores = input_dict["occ_scores"]
        occ_sizes = input_dict["occ_sizes"]
        occ_lengths = input_dict["occ_lengths"]
        mask = (occ_scores > self.score_threshold) & (occ_lengths >= self.min_length)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]

        sample_occ_centers = sample_occ_centers[mask]
        sample_occs = sample_occs[mask]
        occ_scores = occ_scores[mask]
        occ_sizes = occ_sizes[mask]
        occ_lengths = occ_lengths[mask]
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d

        input_dict["sample_occ_centers"] = sample_occ_centers
        input_dict["sample_occs"] = sample_occs
        input_dict["occ_scores"] = occ_scores
        input_dict["occ_sizes"] = occ_sizes
        input_dict["occ_lengths"] = occ_lengths

        return input_dict


@PIPELINES.register_module()
class OccFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if "sample_occ_centers" in results and "sample_occs" in results:
            sample_occ_centers = results["sample_occ_centers"]
            sample_occs = results["sample_occs"]
            occ_labels = torch.cat(
                [sample_occ_centers, sample_occs.unsqueeze(-1)], dim=-1
            )  # N,K,4
            results["occ_labels"] = DC(occ_labels)
        if "occ_scores" in results:
            results["occ_labels_scores"] = DC(results["occ_scores"])
        if "occ_sizes" in results:
            results["occ_sizes"] = DC(results["occ_sizes"])
        # assert len(results["occ_labels"]) == len(results["gt_bboxes_3d"]), f'{len(results["occ_labels"])},{len(results["gt_bboxes_3d"])} {len(results["occ_infos"])}'

        results = super().__call__(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_names={self.class_names}, "
        repr_str += f"with_gt={self.with_gt}, with_label={self.with_label})"
        return repr_str


@PIPELINES.register_module()
class LoadPointsAndOccPredFromFile(LoadPointsFromFile):
    """
    Load points and Occ prediction from file, and merge them into a single point cloud
    """
    def __init__(self,
                 coord_type,
                 points_load_dim=6,
                 occs_load_dim=4,
                 points_use_dim=[0, 1, 2],
                 occs_use_dim=[0, 1, 2, 3],
                 file_client_args=dict(backend='disk'),
                 tanh_dim=None,
                 score_threshold=0.0,
                 filter_prob = 1.0,
                 drop_occ_ratio=0.0,
                 ):
        if isinstance(points_use_dim, int):
            points_use_dim = list(range(points_use_dim))
        assert max(points_use_dim) < points_load_dim, \
            f'Expect all used dimensions < {points_load_dim}, got {points_use_dim}'

        if isinstance(occs_use_dim, int):
            occs_use_dim = list(range(occs_use_dim))
        assert max(occs_use_dim) < occs_load_dim, \
            f'Expect all used dimensions < {occs_load_dim}, got {occs_use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.points_load_dim = points_load_dim
        self.points_use_dim = points_use_dim
        self.occs_load_dim = occs_load_dim
        self.occs_use_dim = occs_use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.tanh_dim = tanh_dim
        self.score_threshold = score_threshold # only keep occ points with score > threshold
        self.filter_prob = filter_prob
        self.drop_occ_ratio = drop_occ_ratio



    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        occ_pred_filename = results['occ_pred_filename']
        try:
            if occ_pred_filename.endswith('.bin'):
                occ_points = self._load_points(occ_pred_filename)
                occ_points = occ_points.reshape(-1, self.occs_load_dim)
                occ_points = occ_points[:, self.occs_use_dim]
            else:
                if not os.path.exists(occ_pred_filename):
                    raise FileNotFoundError
                bin_files = glob.glob(os.path.join(occ_pred_filename, '*.bin'))
                occ_points = []
                for bin_file in bin_files:
                    occ_points.append(self._load_points(bin_file))
                occ_points = np.concatenate(occ_points, axis=0)
                occ_points = occ_points.reshape(-1, self.occs_load_dim)
                occ_points = occ_points[:, self.occs_use_dim]
            if torch.rand(1) <= self.filter_prob:
                occ_points = occ_points[occ_points[:, -1] > self.score_threshold]
            if len(occ_points) == 0:
                # warnings.warn(f'No occ points with score > {self.score_threshold} in {occ_pred_filename}')
                occ_points = np.zeros((0, len(self.points_use_dim) + 2), dtype=np.float32)
        except FileNotFoundError:
            # warnings.warn(f'No occ prediction file {occ_pred_filename}')
            occ_points = np.zeros((0, len(self.points_use_dim) + 2), dtype=np.float32)


        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.points_load_dim)
        points = points[:, self.points_use_dim]
        if self.tanh_dim is not None:
            # only used for SST. FSD applies tanh in the segmentation model.
            assert isinstance(self.tanh_dim, list)
            assert max(self.tanh_dim) < points.shape[1]
            assert min(self.tanh_dim) > 2
            points[:, self.tanh_dim] = np.tanh(points[:, self.tanh_dim])

        attribute_dims = None
        if len(occ_points) > 0:
            assert occ_points.shape[-1] == 4, f'Expect occ points to have 4 dimensions, got {occ_points.shape[-1]}'
        extra_dim = points.shape[-1] - 3
        if extra_dim > 0 and len(occ_points) > 0:
            # generated occ points may have no extra attributes, so we need to pad them
            occ_points_extra = np.pad(occ_points[:,:3], ((0, 0), (0, extra_dim)), mode='constant', constant_values=0)
            # need to keep the objectness score
            occ_points = np.concatenate([occ_points_extra, occ_points[:,3:]], axis=-1)
            
        # pad occ score to real points, the score should be 0 for real points
        points = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        if len(occ_points) > 0:
            # pad indicator to occ points, the indicator should be 1 for occ points
            occ_points = np.pad(occ_points, ((0, 0), (0, 1)), mode='constant', constant_values=1) 
        # pad indicator to real points, the indicator should be 0 for real points
        points = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        merge_points = np.concatenate([points, occ_points], axis=0)




        points_class = get_points_type(self.coord_type)
        merge_points = points_class(
            merge_points, points_dim=merge_points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = merge_points


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str



@PIPELINES.register_module()
class LoadOccPredFromFile(LoadPointsFromFile):
    """
    Load Occ prediction from file, and merge it with previous loaded point cloud into a single point cloud
    """
    def __init__(self,
                 coord_type,
                 occs_load_dim=4,
                 occs_use_dim=[0, 1, 2, 3],
                 file_client_args=dict(backend='disk'),
                 score_threshold=0.0,
                 ):

        if isinstance(occs_use_dim, int):
            occs_use_dim = list(range(occs_use_dim))
        assert max(occs_use_dim) < occs_load_dim, \
            f'Expect all used dimensions < {occs_load_dim}, got {occs_use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.occs_load_dim = occs_load_dim
        self.occs_use_dim = occs_use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.score_threshold = score_threshold # only keep occ points with score > threshold

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        points = results['points']
        points_use_dim = points.points_dim
        occ_pred_filename = results['occ_pred_filename']
        # pad occ score and indicator to real points, the score and indicator should be 0 for real points
        padding = points.tensor.new_zeros((len(points.tensor), 2))
        points.tensor = torch.cat([points.tensor, padding], dim=1)
        points.points_dim += 2
        try:
            if occ_pred_filename.endswith('.bin'):
                occ_points = self._load_points(occ_pred_filename)
                occ_points = occ_points.reshape(-1, self.occs_load_dim)
                occ_points = occ_points[:, self.occs_use_dim]
            else:
                if not os.path.exists(occ_pred_filename):
                    raise FileNotFoundError
                bin_files = glob.glob(os.path.join(occ_pred_filename, '*.bin'))
                occ_points = []
                for bin_file in bin_files:
                    occ_points.append(self._load_points(bin_file))
                occ_points = np.concatenate(occ_points, axis=0)
                occ_points = occ_points.reshape(-1, self.occs_load_dim)
                occ_points = occ_points[:, self.occs_use_dim]
            occ_points = occ_points[occ_points[:, -1] > self.score_threshold]
            if len(occ_points) == 0:
                warnings.warn(f'No occ points with score > {self.score_threshold} in {occ_pred_filename}')
                occ_points = np.zeros((0, points_use_dim + 2), dtype=np.float32)
        except FileNotFoundError:
            warnings.warn(f'No occ prediction file {occ_pred_filename}')
            occ_points = np.zeros((0, points_use_dim + 2), dtype=np.float32)

        if len(occ_points) > 0:
            assert occ_points.shape[-1] == 4, f'Expect occ points to have 4 dimensions, got {occ_points.shape[-1]}'
        else:
            results['points'] = points
            return results
        extra_dim = points_use_dim - 3
        if extra_dim > 0 and len(occ_points) > 0:
            # generated occ points may have no extra attributes, so we need to pad them
            occ_points_extra = np.pad(occ_points[:, :3], ((0, 0), (0, extra_dim)), mode='constant', constant_values=0)
            # need to keep the objectness score
            occ_points = np.concatenate([occ_points_extra, occ_points[:, 3:]], axis=-1)


        if len(occ_points) > 0:
            # pad indicator to occ points, the indicator should be 1 for occ points
            occ_points = np.pad(occ_points, ((0, 0), (0, 1)), mode='constant', constant_values=1)
        occ_points = points.new_point(occ_points)

        results['points'] = points.cat([points, occ_points])
        return results
