_base_ = [
    '../_base_/datasets/waymo-3d-3class-5dim.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py',
]

point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]
voxel_size = (0.1, 0.1, 0.15)
grid_size = [1504, 1504, 40] # x y z
out_size_factor = 8
out_sparse_shape = [s//out_size_factor for s in grid_size[::-1]] # z y x

model = dict(
    type='DynamicCenterPoint',

    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE', # change to dynamic scatter vfe
        in_channels=5,
        feat_channels=[16, 16],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='SparseScatterEncoder',
        in_channels=  [16,        32,       32,       64,       64,         128,        128],
        feat_channels=[[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128], [128, 128]],
        window_sizes= [2, 2, 4, 4, 4, 4, 4],
        strides=[1, 2, 1, 2, 1, 2, 1],
        shifted=[True, False, True, False, True, False, True],
        out_sparse_shape=out_sparse_shape,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        mode='max',
        # pe_config=dict(
        #     mlp=[64,],
        # ),
        pe_config=None,
    ),

    backbone=dict(
        type='SECOND',
        in_channels=128 * 5,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),

    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=3, class_names=['car', 'pedestrian', 'cyclist']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=point_cloud_range,
            max_num=4096,
            score_threshold=0.1,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True
    ),
    # model training and testing settings
    train_cfg=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
        out_size_factor=out_size_factor,
        dense_reg=1, # what is this
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        point_cloud_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    ), # seems corresponding to common_heads
    test_cfg=dict(
        post_center_limit_range=[-80, -80, -10, 80, 80, 10],
        max_per_img=500, # what is this
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175], # not used in normal nms, seems task-wise
        score_threshold=0.1,
        pc_range=point_cloud_range[:2], # seems not used
        out_size_factor=out_size_factor,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        pre_max_size=4096, # change 
        post_max_size=500, # change
        nms_thr=0.7
    )
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=6)
evaluation = dict(interval=6)

# fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=1)
    ),
)
