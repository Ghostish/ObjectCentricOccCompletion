from collections import defaultdict
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.core.bbox.structures.lidar_tracklet import LiDARTracklet
from mmdet3d.ops.sst.sst_ops import scatter_v2

from tools.ctrl.utils import get_pc_from_time_stamp, read_bin, generate_tracklets
import numpy as np
import os.path as osp
import argparse
import pickle as pkl

# import open3d as o3d
import mmcv
import torch.multiprocessing as mp
from glob import glob


# import pdb
def tracklet_list2dict(tracklets):
    trks_dict = defaultdict(list)
    for trk in tracklets:
        segname = trk.segment_name
        trks_dict[segname].append(trk)
    return trks_dict


# pdb.set_trace()
def write_occ_obj(points, occ, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    color = np.zeros((N, 3)).astype(int)
    # color[occ==0] = [255, 0, 0]
    color[occ == 1] = [0, 255, 0]
    color[occ == 2] = [0, 0, 255]
    fout = open(out_filename, "w")
    for i in range(N):
        fout.write(
            "v %f %f %f %d %d %d\n"
            % (
                points[i, 0],
                points[i, 1],
                points[i, 2],
                color[i, 0],
                color[i, 1],
                color[i, 2],
            )
        )

    fout.close()


def write_obj(points, out_filename, color=None):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, "w")
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                "v %f %f %f %d %d %d\n"
                % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2])
            )
        elif color:
            fout.write(
                "v %f %f %f %d %d %d\n"
                % (
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    color[0],
                    color[1],
                    color[2],
                )
            )
        else:
            fout.write("v %f %f %f\n" % (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def get_local_point_list(
    trk, ts2idx, waymo_data_root, split="training", box_mode="avg", cache_pcs=None
):
    local_pc_list = []
    box_sizes = []
    for i in range(len(trk)):
        # print(len(trk))
        box = trk.box_list[i]
        ts = trk.ts_list[i]
        if cache_pcs is None:
            pc = get_pc_from_time_stamp(ts, ts2idx, waymo_data_root, split=split)
        elif ts in cache_pcs:
            pc = cache_pcs[ts]
        else:
            pc = get_pc_from_time_stamp(ts, ts2idx, waymo_data_root, split=split)
            cache_pcs[ts] = pc
        pc = torch.from_numpy(pc[:, :3]).cuda()
        box.tensor = box.tensor.cuda()  # x,y,z,w,l,h,yaw
        inbox_inds = box.points_in_boxes(pc)
        in_this_box_pc = pc[inbox_inds == 0]
        if len(in_this_box_pc) == 0:
            continue

        box = box.clone()
        origin = box.tensor[:, :3]
        rz = box.tensor[0, 6]
        tranlate = -origin
        rot = -rz
        # to local coordinate system centered at box's center, x axis points to the front of the object
        local_pc = in_this_box_pc + tranlate
        box.translate(tranlate)
        local_pc, _ = box.rotate(rot, local_pc)
        # optional, centered at the front face of the box
        # local_pc[:, 1:2] = (local_pc[:, 1:2] + box.tensor[0, 4:5] / 2)
        # box.translate(box.tensor[0, 4:5] / 2)

        local_pc_list.append(local_pc)
        box_sizes.append(box.tensor[:, 3:6])
    assert len(local_pc_list) > 0, "no points in the tracklet"
    if box_mode == "avg":
        local_box_size = torch.mean(torch.cat(box_sizes, dim=0), dim=0)
    elif box_mode == "max":
        local_box_size, _ = torch.max(torch.cat(box_sizes, dim=0), dim=0)
    local_box_tensor = torch.zeros((1, 7), device=box_sizes[0].device)
    local_box_tensor[0, 3:6] = local_box_size
    local_box = LiDARInstance3DBoxes(local_box_tensor)

    return local_pc_list, local_box


def point_cloud_to_range_image_idx(points, extrinsics, inclinations, range_image_size):
    """Convert point cloud to range image index.

    Args:
        points (np.ndarray): Points in shape (B, N, 3).
        extrinsics (np.ndarray): Extrinsic matrix in shape (B, 4, 4).
        inclinations (np.ndarray): Inclinations in shape (B,H,).
        range_image_size (int): Range image size. [height, width]

    Returns:
        np.ndarray: Range image index in shape (B, N, 2).
        np.ndarray: Range image range in shape (B, N).
    """

    height, width = range_image_size
    # vehicle_to_laser = torch.linalg.inv(extrinsics).to(torch.float64)
    # for 4090 GPUs, torch.linalg.inv is not supported
    vehicle_to_laser = (
        torch.linalg.inv(extrinsics.cpu()).to(extrinsics.device).to(torch.float64)
    )
    rotation = vehicle_to_laser[:, :3, :3]  # B,3,3
    translation = torch.unsqueeze(vehicle_to_laser[:, :3, 3], dim=1)  # B,1,3
    # points in sensor frame
    points = torch.einsum("bij,bkj->bik", points, rotation) + translation  # B,N,3
    xy_norm = torch.linalg.norm(points[..., :2], dim=-1)  # B,N
    point_inclinations = torch.atan2(points[..., 2], xy_norm)  # B,N
    point_ri_row_indices = []
    for b in range(len(point_inclinations)):
        points_inclination_diff = torch.abs(
            point_inclinations[b, :, None] - inclinations[b, None, :]
        )  # N,H
        point_ri_row_indices.append(torch.argmin(points_inclination_diff, dim=-1))  # N
    point_ri_row_indices = torch.stack(point_ri_row_indices, dim=0)  # B,N

    az_correction = torch.atan2(extrinsics[:, 1, 0], extrinsics[:, 0, 0])  # B,
    point_azimuth = (
        torch.atan2(points[..., 1], points[..., 0]) + az_correction[..., None]
    )  # B,N
    point_azimuth_gt_pi_mask = point_azimuth > np.pi
    point_azimuth_lt_minus_pi_mask = point_azimuth < -np.pi
    point_azimuth = (
        point_azimuth - point_azimuth_gt_pi_mask.to(torch.float32) * 2 * np.pi
    )
    point_azimuth = (
        point_azimuth + point_azimuth_lt_minus_pi_mask.to(torch.float32) * 2 * np.pi
    )
    point_ri_col_indices = (
        width - 1.0 + 0.5 - (point_azimuth + np.pi) / (2.0 * np.pi) * width
    )  # reverse index: pi -> 0.5, -pi -> width-1 + 0.5
    point_ri_col_indices = torch.round(point_ri_col_indices)  # B,N
    point_ri_col_indices = torch.fmod(point_ri_col_indices, width).to(
        torch.int32
    )  # B,N

    ri_indices = torch.stack(
        [point_ri_row_indices, point_ri_col_indices], dim=-1
    )  # B,N,2
    ri_range = torch.linalg.norm(points, dim=-1)  # B,N
    # out_mask = (point_ri_col_indices > width - 1) & (point_ri_col_indices < 0)

    return ri_indices, ri_range


parser = argparse.ArgumentParser()
# parser.add_argument('config', type=str)
parser.add_argument(
    "--bin-path",
    type=str,
    default="./data/waymo/waymo_format/train_gt.bin",
)
parser.add_argument("--split", type=str, default="training")
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--ngpus", type=int, default=8)
parser.add_argument("--chunksize", type=int, default=1)
parser.add_argument("--voxel-size", type=float, default=0.2)
parser.add_argument("--type", type=str, default="vehicle")
parser.add_argument("--data-root", type=str, default="./data/waymo/")
parser.add_argument(
    "--out-dir", type=str, default="./work_dirs/occ_annotate/waymo_occ_gt"
)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--cpu-voxelization", action="store_true", default=False)
parser.add_argument("--save-mean-var", action="store_true", default=False)
parser.add_argument("--overwrite", action="store_true", default=False)
args = parser.parse_args()


class OccAnnotator(object):
    type_mapping = {
        "vehicle": 1,
        "pedestrian": 2,
        "cyclist": 3,
    }

    LiDAR_NAME_LIST = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]

    def __init__(
        self,
        data_root,
        out_dir,
        split,
        voxel_size,
        bin_path,
        object_type="vehicle",
        workers=64,
        debug=False,
        cpu_voxelization=False,
        overwrite=False,
        save_mean_var=False,
        ngpus=8,
    ):
        self.data_root = data_root
        self.out_dir = out_dir
        self.kitti_format_root = osp.join(data_root, "kitti_format")
        self.raw_format_root = osp.join(data_root, "waymo_raw", split)
        self.idx2ts_path = osp.join(self.kitti_format_root, "idx2timestamp.pkl")
        self.voxel_size = voxel_size
        self.debug = debug
        self.workers = workers
        self.split = split
        self.cpu_voxelization = cpu_voxelization
        self.overwrite = overwrite
        self.save_mean_var = save_mean_var

        self.ngpus = ngpus
        self.types = set(
            [
                self.type_mapping[object_type],
            ]
        )
        self.bin_path = bin_path

        basename = osp.basename(bin_path)
        name = basename.split(".")[0]
        tracklets = self.generate_or_load_tracklet(bin_path, f"{name}_tracklets.pkl")
        self.trk_dicts = tracklet_list2dict(tracklets)
        self.segment_names = list(self.trk_dicts.keys())
        self.segment_names.sort()

        with open(self.idx2ts_path, "rb") as fr:
            self.idx2ts = pkl.load(fr)
        self.ts2idx = {ts: idx for idx, ts in self.idx2ts.items()}

    def generate_or_load_tracklet(self, bin_path, file_name):
        if osp.isfile(osp.join(self.out_dir, file_name)):
            tracklets = mmcv.load(osp.join(self.out_dir, file_name))

        else:
            bin_data = read_bin(bin_path)
            tracklets = generate_tracklets(bin_data, self.types)
            self.tracklets_to_collate(tracklets)
            mmcv.mkdir_or_exist(self.out_dir)
            mmcv.dump(tracklets, osp.join(self.out_dir, file_name))
        return tracklets

    def tracklets_to_collate(self, tracklets):
        for t in tracklets:
            t.to_collate_format()

    def tracklets_from_collate(self, tracklets):
        for t in tracklets:
            t.from_collate_format()

    def annotate_one_seg(self, segname_idx):

        segment_name = self.segment_names[segname_idx]
        trks = self.trk_dicts[segment_name]

        print(
            f"---------start annotating segment {segment_name} with {len(trks)} tracks..."
        )
        cache_segment_pcs = {}
        for trk in trks:
            self.annotate_trk(trk, cache_segment_pcs)
        print(
            f"+++++++++finished annotating segment {segment_name} with {len(trks)} tracks..."
        )

    def annotate_trk(self, trk, cache_pcs=None):
        if self.workers > 1:
            wid = mp.current_process()._identity[0] - 1
            torch.cuda.set_device(wid % self.ngpus)
        try:
            trk.from_collate_format()
        except:
            pass
        segname = trk.segment_name
        trk_id = trk.id
        uuid = trk.uuid
        out_path = osp.join(self.out_dir, self.split, segname)
        mmcv.mkdir_or_exist(out_path)
        out_name = osp.join(out_path, f"{trk_id}.npz")
        # if osp.isfile(out_name) and not self.overwrite:
        #     return
        if osp.isfile(out_name):
            if not self.overwrite:
                try:
                    _ = np.load(out_name)
                    # print(f"skip {out_name}, already exists")
                    return
                except:
                    print(f"error loading {out_name}, overwrite")
                    pass
        if len(trk) < 10:
            return
        try:
            # get local point cloud and box by centering and aggregating all points in the tracklet
            local_pc_list, local_box = get_local_point_list(
                trk,
                self.ts2idx,
                self.kitti_format_root,
                split=self.split,
                box_mode="max",
                cache_pcs=cache_pcs,
            )
        except AssertionError as e:
            # empty tracklet, ignore
            print(e)
            return
        local_pc_agg = torch.cat(local_pc_list, dim=0)
        if self.cpu_voxelization:
            assert (
                not self.save_mean_var
            ), "not implemented in cpu when save_mean_var is True"
            local_pc_agg_cpu = local_pc_agg.cpu().numpy()

            # create 3d voxel grid from the box and local point cloud given voxel size, using open3d
            voxel_size = self.voxel_size
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(local_pc_agg_cpu)

            corners = local_box.corners[0].cpu().numpy()  # 8,3
            box_center = local_box.gravity_center[0].cpu().numpy()  # 3,
            box_size = local_box.dims[0].cpu().numpy()  # 3,
            min_bound = np.min(corners, axis=0)
            max_bound = np.max(corners, axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(local_pc_agg_cpu)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
                pcd,
                voxel_size=voxel_size,
                min_bound=min_bound,
                max_bound=max_bound,
            )
            voxel_grid_dense = o3d.geometry.VoxelGrid.create_dense(
                box_center - box_size / 2,
                [0, 1, 0],  # color
                voxel_size,
                box_size[0],
                box_size[1],
                box_size[2],
            )
            dense_voxels_all = voxel_grid_dense.get_voxels()
            dense_all_centers = []
            dense_all_coords = []
            for voxel in dense_voxels_all:
                voxel_center = voxel_grid_dense.get_voxel_center_coordinate(
                    voxel.grid_index
                )
                dense_all_centers.append(voxel_center)
                dense_all_coords.append(voxel.grid_index)
            voxel_centers = o3d.utility.Vector3dVector(dense_all_centers)
            occ = voxel_grid.check_if_included(voxel_centers)

            occ = torch.tensor(occ).cuda()  # N,
            voxel_coors = torch.tensor(dense_all_coords, dtype=torch.long).cuda()  # N,3
            voxel_centers = torch.tensor(dense_all_centers).cuda()  # N,3
            voxel_dims = torch.max(voxel_coors, dim=0)[0] + 1  # x_size, y_size, z_size
            unknown_centers = voxel_centers[~occ]

        else:
            bbox_size = local_box.tensor[0, 3:6]  # 3,
            voxel_dims = torch.ceil(bbox_size / self.voxel_size).to(
                torch.int32
            )  # 3; [x_size, y_size, z_size]
            occ = torch.zeros(
                (voxel_dims[0], voxel_dims[1], voxel_dims[2]),
                dtype=torch.bool,
                device=local_pc_agg.device,
            )  # x_size, y_size, z_size
            corners = local_box.corners[0]  # 8,3
            min_bound = torch.min(corners, dim=0)[0]  # 3,
            max_bound = torch.max(corners, dim=0)[0]  # 3,
            quantized_pc = torch.floor((local_pc_agg - min_bound) / self.voxel_size).to(
                torch.long
            )  # N,3, occupied voxels coords

            # points may fall right on the boundary, remove them
            local_pc_agg = local_pc_agg[(quantized_pc < voxel_dims[None]).all(dim=1)]
            quantized_pc = quantized_pc[(quantized_pc < voxel_dims[None]).all(dim=1)]

            assert (
                quantized_pc.max(dim=0)[0] < voxel_dims
            ).all(), f"{quantized_pc.max(dim=0)[0], {voxel_dims}, {bbox_size} }"
            occ[quantized_pc[:, 0], quantized_pc[:, 1], quantized_pc[:, 2]] = (
                True  # x_size, y_size, z_size
            )

            voxel_coors_x, voxel_coors_y, voxel_coors_z = torch.meshgrid(
                torch.arange(
                    voxel_dims[0], dtype=torch.long, device=local_pc_agg.device
                ),
                torch.arange(
                    voxel_dims[1], dtype=torch.long, device=local_pc_agg.device
                ),
                torch.arange(
                    voxel_dims[2], dtype=torch.long, device=local_pc_agg.device
                ),
            )  # x_size, y_size, z_size, 3

            voxel_coors = torch.stack(
                [voxel_coors_x, voxel_coors_y, voxel_coors_z], dim=-1
            ).view(
                -1, 3
            )  # x_size* y_size* z_size, 3
            occ = occ.view(-1)
            if self.debug:
                voxel_centers = (
                    voxel_coors.to(torch.float64) * self.voxel_size
                    + min_bound
                    + self.voxel_size / 2
                )  # N, 3, unoccupied voxels centers
                unknown_centers = voxel_centers[~occ]
            else:
                un_occ_coors = voxel_coors[~occ]  # N, 3, unoccupied voxels coords
                unknown_centers = (
                    un_occ_coors.to(torch.float64) * self.voxel_size
                    + min_bound
                    + self.voxel_size / 2
                )  # N, 3, unoccupied voxels centers
        if un_occ_coors.size(0) > 0:
            ego_unknown_centers_list = []
            camera_extrinsic_dict = defaultdict(list)
            camera_inclination_dict = defaultdict(list)
            range_image_dict = defaultdict(list)
            range_image_size_list = {}
            ts_list = trk.ts_list
            for i in range(len(ts_list)):
                ts = ts_list[i]
                file_idx = self.ts2idx[ts]
                # local to ego at timestamp ts_list[i]
                box = trk.box_list[i]
                box = box.clone().tensor
                origin = box[:, :3]
                rz = box[0, 6]
                tranlate = origin
                rot = rz

                rot_sin = torch.sin(rot)
                rot_cos = torch.cos(rot)
                rot_mat_T = torch.tensor(
                    [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                    device=unknown_centers.device,
                    dtype=unknown_centers.dtype,
                )
                ego_unknown_centers = unknown_centers @ rot_mat_T
                ego_unknown_centers = ego_unknown_centers + tranlate
                ego_unknown_centers_list.append(ego_unknown_centers)

                # read raw frame data at timestamp ts
                raw_frame_path = osp.join(self.raw_format_root, f"{file_idx}.pkl")
                if not osp.isfile(raw_frame_path):
                    print(f"{raw_frame_path} not found, skip this segment {segname}")
                    return
                try:
                    frame_data = mmcv.load(raw_frame_path)
                except:
                    print(f"error loading {raw_frame_path}, skip this segment {segname}")
                    return
                for camera in self.LiDAR_NAME_LIST:
                    inclination = frame_data[f"{camera}_BEAM_INCLINATION"]  # H,
                    extrinsic = frame_data[f"{camera}_LIDAR_EXTRINSIC"]  # 4,4
                    ri = frame_data[f"{camera}_RANGE_IMAGE_MERGE_VIRTUAL"]  # H,W;range

                    camera_extrinsic_dict[camera].append(extrinsic)
                    camera_inclination_dict[camera].append(inclination)
                    range_image_dict[camera].append(ri)
                    range_image_size_list[camera] = ri.shape
                # if args.debug:
                #     break

            ego_unknown_centers = torch.stack(ego_unknown_centers_list, dim=0)  # B,N,3
            visibility_list = []
            for camera in self.LiDAR_NAME_LIST:
                extrinsics = np.stack(camera_extrinsic_dict[camera], axis=0)  # B,4,4
                inclinations = np.stack(camera_inclination_dict[camera], axis=0)  # B,H
                inclinations = np.flip(inclinations, axis=1)
                range_images = np.stack(range_image_dict[camera], axis=0)  # B,H,W

                extrinsics = torch.tensor(extrinsics).cuda()
                inclinations = torch.tensor(inclinations.copy()).cuda()
                range_images = torch.tensor(range_images).cuda()

                range_image_size = range_image_size_list[camera]
                ri_indices, ri_range = point_cloud_to_range_image_idx(
                    ego_unknown_centers, extrinsics, inclinations, range_image_size
                )  # B,N,2; B,N
                # gather value from range images using ri_indices
                ri_values = []
                for i in range(len(ri_indices)):
                    ri_values.append(
                        range_images[i, ri_indices[i, :, 0], ri_indices[i, :, 1]]
                    )
                ri_values = torch.stack(ri_values, dim=0)  # B,N
                visibility = torch.zeros_like(ri_values, dtype=torch.int32)  # B,N
                visibility[(ri_values >= ri_range)] = 2  # empty points, B,N
                # for each camera, if a point is found to be empty in any frame,
                # it is considered as empty even if it is occluded in other frames
                visibility, _ = torch.max(visibility, dim=0)  # N,
                visibility_list.append(visibility)
            visibility = torch.stack(visibility_list, dim=0)  # C,N
            # for each point, if it is found empty in any camera, it is considered as empty
            visibility, _ = torch.max(
                visibility, dim=0
            )  # N, 2 indicates empty, 0 indicates occluded

            visible_occ = torch.zeros_like(
                occ, dtype=torch.int32
            )  # 0 indicates occluded/unknown, 1 indicates occupied, 2 indicates empty

            visible_occ[~occ] = visibility
            visible_occ[occ] = 1
        else:
            visible_occ = torch.zeros_like(
                occ, dtype=torch.int32
            )  # 0 indicates occluded/unknown, 1 indicates occupied, 2 indicates empty
            visible_occ[occ] = 1
            print(trk.id, trk.uuid, trk.segment_name)
            print(unknown_centers.shape, voxel_coors.shape, occ.shape)
        if self.cpu_voxelization:
            occ = torch.zeros(
                voxel_dims[0],
                voxel_dims[1],
                voxel_dims[2],
                dtype=torch.int32,
                device=visible_occ.device,
            )
            occ[voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2]] = visible_occ
        else:
            occ = visible_occ.view(voxel_dims[0], voxel_dims[1], voxel_dims[2])
        if self.debug:
            # write_occ_obj(voxel_centers[visible_occ !=0], visible_occ[visible_occ !=0], "./test_occ.obj")
            if self.cpu_voxelization:
                out_path = osp.join(
                    self.out_dir,
                    f"{segname}__{trk_id}__{len(trk)}__{args.voxel_size}_virtual_CPU",
                )
            else:
                out_path = osp.join(
                    self.out_dir,
                    f"{segname}__{trk_id}__{len(trk)}__{args.voxel_size}_virtual_GPU",
                )
            mmcv.mkdir_or_exist(out_path)
            write_obj(
                voxel_centers[visible_occ == 1].cpu().numpy(),
                osp.join(out_path, "occ_1.obj"),
                color=[255, 0, 0],
            )
            write_obj(
                voxel_centers[visible_occ == 2].cpu().numpy(),
                osp.join(out_path, "occ_2.obj"),
                color=[0, 255, 0],
            )
            write_obj(
                voxel_centers[visible_occ == 0].cpu().numpy(),
                osp.join(out_path, "occ_0.obj"),
                color=[0, 0, 255],
            )
            write_obj(local_pc_agg.cpu().numpy(), osp.join(out_path, "local_pc.obj"))

            np.savez(osp.join(out_path, "occ.npz"), occ=occ.cpu().numpy())
        else:
            # if self.cpu_voxelization:
            #     occ = torch.zeros(
            #         voxel_dims[0],
            #         voxel_dims[1],
            #         voxel_dims[2],
            #         dtype=torch.int32,
            #         device=visible_occ.device,
            #     )
            #     occ[
            #         voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2]
            #     ] = visible_occ
            # else:
            #     occ = visible_occ.view(voxel_dims[0], voxel_dims[1], voxel_dims[2])
            if self.save_mean_var:
                mean_xyz, new_coors, unq_inv = scatter_v2(
                    local_pc_agg, quantized_pc, "mean", return_inv=True
                )
                var = (local_pc_agg - mean_xyz[unq_inv]) ** 2
                var_xyz, _, _ = scatter_v2(
                    var, quantized_pc, "mean", unq_inv=unq_inv, new_coors=new_coors
                )
                mean_var = torch.zeros(
                    (voxel_dims[0], voxel_dims[1], voxel_dims[2], 6),
                    dtype=local_pc_agg.dtype,
                    device=local_pc_agg.device,
                )
                mean_var[new_coors[:, 0], new_coors[:, 1], new_coors[:, 2], :] = (
                    torch.cat([mean_xyz, var_xyz], dim=1)
                )
                np.savez(
                    out_name, occ=occ.cpu().numpy(), mean_var=mean_var.cpu().numpy()
                )
            else:
                np.savez(out_name, occ=occ.cpu().numpy())

    def annotate_segment(self, chunksize=-1):
        seg_len = len(self.segment_names)
        if self.workers > 1:
            if chunksize == -1:
                chunksize = seg_len // self.workers

            print(
                f"Start annotating ..., number of workers: {self.workers}, chunksize: {chunksize}"
            )
            print(f"Total {seg_len} segments")
            mmcv.track_parallel_progress(
                self.annotate_one_seg,
                range(seg_len),
                self.workers,
                keep_order=False,
                chunksize=chunksize,
            )
        else:
            print(
                f"Start annotating ..., using one process, number of segments: {seg_len}"
            )
            mmcv.track_progress(self.annotate_one_seg, range(seg_len))
        print("\nFinished ...")


def load_tracklets(bin_path, cache_path):
    if osp.isfile(cache_path):
        tracklets = mmcv.load(cache_path)
    else:
        bin_data = read_bin(bin_path)
        tracklets = generate_tracklets(bin_data, args.type)
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(tracklets, cache_path)
    return tracklets


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    annotator = OccAnnotator(
        data_root=args.data_root,
        out_dir=args.out_dir,
        split=args.split,
        voxel_size=args.voxel_size,
        bin_path=args.bin_path,
        object_type=args.type,
        workers=args.workers,
        debug=args.debug,
        cpu_voxelization=args.cpu_voxelization,
        overwrite=args.overwrite,
        save_mean_var=args.save_mean_var,
        ngpus=args.ngpus,
    )

    annotator.annotate_segment(args.chunksize)
