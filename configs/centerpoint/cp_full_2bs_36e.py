# align with official waymo model: voxel_size, max_voxels, loss weight, test_cfg, stride
_base_ = [
    '../_base_/datasets/waymo-3d-3class.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
]
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]

voxel_size = [0.32, 0.32, 6]
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=20,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(32000, 60000)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=3,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(468, 468)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        checkpoint_stages=[0,]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=3, class_names=['car', 'pedestrian', 'cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
            # reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2),
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-74.88, -74.88, -10.0, 74.88, 74.88, 10.0],
            max_num=4096,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[468, 468, 1],
            voxel_size=voxel_size,
            out_size_factor=1,
            dense_reg=1, # what is this
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            point_cloud_range=point_cloud_range,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])), # seems corresponding to common_heads
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-80, -80, -10, 80, 80, 10],
            max_per_img=500, # what is this
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175], # not used in normal nms, seems task-wise
            score_threshold=0.1,
            pc_range=point_cloud_range[:2], # seems not used
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096, # change 
            post_max_size=500, # change
            nms_thr=0.7)))


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=1)
    ),
    # test=dict(load_interval=198) # for vis
)
runner = dict(type='EpochBasedRunner', max_epochs=36)
evaluation = dict(interval=36)