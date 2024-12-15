import torch



def generate_dense_voxel_centers(
    bbox_sizes,
    voxel_size,
    scale_wlh=[1.0, 1.0, 1.0],
    offset_wlh=[0.0, 0.0, 0.0],
    as_volume=False,
):
    """Generate voxel centers,given the bbox_sizes.
        The centers are in the object coordinate system, where each origin coincides with the gravity center of the canonical bbox.

    Args:
        bbox_sizes (torch.Tensor): sizes of bboxes.

    Returns:
        list[torch.Tensor]: Voxel centers.
    """
    voxel_centers_list = []

    for size in bbox_sizes:
        size = size * size.new_tensor(scale_wlh) + size.new_tensor(offset_wlh)

        XS, YS, ZS = torch.ceil(size / voxel_size)
        voxel_coors_x, voxel_coors_y, voxel_coors_z = torch.meshgrid(
            torch.arange(XS, dtype=torch.long, device=bbox_sizes.device),
            torch.arange(YS, dtype=torch.long, device=bbox_sizes.device),
            torch.arange(ZS, dtype=torch.long, device=bbox_sizes.device),
        )  # x_size, y_size, z_size, 3

        voxel_coors = torch.stack(
            [voxel_coors_x, voxel_coors_y, voxel_coors_z], dim=-1
        ).view(
            -1, 3
        )  # x_size* y_size* z_size, 3

        # center the coordinate system at the gravity center of the voxel grid
        min_bound = -size / 2

        voxel_centers = (
            voxel_coors.to(torch.float) * voxel_size + min_bound + voxel_size / 2
        )  # K, 3,
        if as_volume:
            voxel_centers = voxel_centers.view(
                XS.int().item(), YS.int().item(), ZS.int().item(), 3
            )
        voxel_centers_list.append(voxel_centers)
    return voxel_centers_list


def quantize_points(
    points,
    rois,
    rois_points_idx,
    voxel_size,
    scale_wlh=[1.0, 1.0, 1.0],
    offset_wlh=[0.0, 0.0, 0.0],
    to_center=False,
):
    """Quantize points to voxel coors, the whole volume is centered at the ROIs.
       The ROIs are first enlarged using scale_wlh and offset_wlh.
       the points should be in the local coordinate system defined by the ROIs.

    Args:
        points (torch.Tensor): Points to be quantized. Nx3
        rois (torch.Tensor): Rois, (N, 8) or (N, 10),
            each row is (batch_idx, x, y, z, w, l, h, ry) or (batch_idx, x, y, z, w, l, h,ry,dx,dy).
        rois_points_idx (torch.Tensor): Indices of rois each points belong to , (N,).
        to_center (bool): Whether convert quantize coordinates to voxel center.

    """
    # quantize the points to the voxel centers
    # assert (rois_points_idx >= 0).all()
    bboxes_sizes = rois[:, 4:7]
    bboxes_sizes = bboxes_sizes * bboxes_sizes.new_tensor(scale_wlh).view(
        1, 3
    ) + bboxes_sizes.new_tensor(offset_wlh).view(1, 3)

    roi_min_bound = -bboxes_sizes / 2
    min_bound_per_point = roi_min_bound[rois_points_idx]
    voxel_coors = torch.floor((points - min_bound_per_point) / voxel_size).to(
        torch.long
    )
    if to_center:
        voxel_centers = (
            voxel_coors.to(torch.float) * voxel_size
            + min_bound_per_point
            + voxel_size / 2
        )  # K, 3,
        return voxel_centers
    return voxel_coors


def jitter_voxel_center(voxel_size, voxel_centers):
    jitter_noise = (
        torch.rand_like(voxel_centers) * voxel_size - voxel_size / 2
    )  # [-voxel_size/2, voxel_size/2]
    return voxel_centers + jitter_noise
