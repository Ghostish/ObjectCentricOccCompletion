_base_ = [
    "../_base_/datasets/waymo-tracklet-vehicle.py",
    "../_base_/schedules/cosine_2x.py",
    "../_base_/default_runtime.py",
]

seg_voxel_size = (0.2, 0.2, 0.2)
point_cloud_range = [-204.8, -204.8, -4.0, 204.8, 204.8, 8.0]
class_names = [
    "Car",
]
num_classes = len(class_names)

model = dict(
    type="TrackletDetectorOT",
    roi_head=dict(
        type="TrackletRoIHead",
        num_classes=num_classes,
        general_cfg=dict(
            with_roi_scores=True,
        ),
        roi_extractor=dict(
            type="TrackletPointRoIExtractor",
            extra_wlh=[0.5, 0.5, 0.5],
            max_inbox_point=512,
            max_all_point=(300000, 600000),
            debug=False,
            combined=False,
        ),
        bbox_head=dict(
            type="FullySparseBboxHeadTransBase",
            num_classes=num_classes,
            num_blocks=6,
            in_channels=[24, 144, 144, 144, 144, 144],
            feat_channels=[
                [128, 128],
            ]
            * 6,
            rel_mlp_hidden_dims=[
                [16, 32],
            ]
            * 6,
            rel_mlp_in_channels=[
                13,
            ]
            * 6,
            reg_mlp=[512, 512],
            cls_mlp=[512, 512],
            mode="max",
            xyz_normalizer=[20, 20, 4],
            act="gelu",
            geo_input=True,
            with_corner_loss=True,
            corner_loss_weight=1.0,
            bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
            norm_cfg=dict(type="LN", eps=1e-3),
            unique_once=True,
            loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=2.0),
            loss_cls=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                reduction="mean",
                loss_weight=1.0,
            ),
            cls_dropout=0.1,
            reg_dropout=0.1,
            roi_feature_channels=1536,
            attn_num_head=4,
            attn_ffn_dim=512,
            attn_dropout=0.1,
            num_enc_layers=3,
            roi_pos_enc_mlp=[512,512],
            roi_enc_dropout=0,
            use_transformer=False
        ),
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        use_roi_frame_inds=True,

    ),
    train_cfg=dict(
        # pre_voxelization_size=(0.1, 0.1, 0.1),
        pre_voxelization_size=None,
        assigner=dict(  # Car
            type="TrackletAssigner",
        ),
        hack_sampler_bug=True,
        cls_pos_thr=(0.8,),
        cls_neg_thr=(0.2,),
        sync_reg_avg_factor=True,
        sync_cls_avg_factor=True,
        corner_loss_only_car=True,  # default True, explicitly set to False to disable
        class_names=class_names,
        rcnn_code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    test_cfg=dict(
        batch_inference=True,
        identical_decode=False,
        # tta=dict(
        #     merge="weighted",
        # ),
    ),
)

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=24)
evaluation = dict(interval=24)

train_pipeline = [
    dict(
        type="LoadTrackletPoints",
        load_dim=6,
        use_dim=5,  # remove the wrong timestamp
        max_points=1024,
        debug=False,
    ),
    dict(
        type="LoadTrackletAnnotations",
    ),
    dict(  # optional
        type="TrackletCutting",
        ratio=0.0,
        max_length=150,
        shift_time_zero=True,
    ),
    dict(
        type="TrackletPoseTransform",
        concat=False,
    ),
    dict(
        type="TrackletNoise",
        center_noise_cfg=dict(max_noise=[0.2, 0.2, 0.1], consistent=False),
        size_noise_cfg=dict(max_noise=[0.2, 0.2, 0.1], consistent=False),
        yaw_noise_cfg=dict(max_noise=0.2, consistent=False),
    ),
    dict(
        type="PointDecoration",
        properties=["yaw", "size", "score"],
        concat=True,
    ),
    dict(
        type="TrackletRandomFlip",
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="TrackletGlobalRotScaleTrans",
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0.2],
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=[-204.7, -204.7, -3.99, 204.7, 204.7, 7.99],
    ),
    dict(type="PointShuffle"),
    dict(type="TrackletFormatBundle", class_names=class_names),
    dict(
        type="Collect3D",
        keys=["points", "pts_frame_inds", "tracklet", "gt_tracklet_candidates"],
    ),
]
test_pipeline = [
    dict(
        type="LoadTrackletPoints",
        load_dim=6,
        use_dim=5,  # remove the wrong timestamp
        max_points=1024,
        debug=False,
    ),
    dict(
        type="TrackletPoseTransform",
        concat=False,
    ),
    dict(
        type="PointDecoration",
        properties=["yaw", "size", "score"],
        concat=True,
    ),
    dict(
        type="PointsRangeFilter",
        point_cloud_range=[-204.7, -204.7, -3.99, 204.7, 204.7, 7.99],
    ),
    dict(type="PointShuffle"),
    dict(type="TrackletFormatBundle", class_names=class_names),
    dict(type="Collect3D", keys=["points", "pts_frame_inds", "tracklet"]),
]

tta_pipeline = [
    dict(
        type="LoadTrackletPoints",
        load_dim=6,
        use_dim=5,  # remove the wrong timestamp
        max_points=1024,
        debug=False,
    ),
    dict(
        type="TrackletPoseTransform",
        concat=False,
    ),
    dict(
        type="PointDecoration",
        properties=["yaw", "size", "score"],
        concat=True,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        # pts_rots=[0, ],
        # pts_rots=[-4*pi/5, -2*pi/5, 0, 2*pi/5, 4*pi/5, ],
        # pts_rots=[0, 2*pi/5, 4*pi/5, -4*pi/5, -2*pi/5, ], # Note the order if use iou clamp
        # pts_rots=[pi/2, 0, -pi/2],
        flip=True,
        pcd_horizontal_flip=True,  # double flip
        pcd_vertical_flip=True,
        transforms=[
            dict(
                type="TrackletGlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(
                type="TrackletRandomFlip",
                flip_ratio_bev_horizontal=0.0,
                flip_ratio_bev_vertical=0.0,
            ),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="PointShuffle"),
            dict(type="TrackletFormatBundle", class_names=class_names),
            dict(type="Collect3D", keys=["points", "pts_frame_inds", "tracklet"]),
        ],
    ),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=1,
        # type="ClassBalancedDataset",
        # oversample_thr=0.6,
        filter_empty_gt=True,
        dataset=dict(
            ann_file="/LiZhen_team/dataset/waymo/tracklet_data/fsd_base_1f_vehicle_training_gt_candidates.pkl",
            tracklet_proposals_file="/LiZhen_team/dataset/waymo/tracklet_data/fsd_base_1f_vehicle_training.pkl",
            # ann_file="/LiZhen_team/dataset/waymo/tracklet_data/cp_vehicle_training_gt_candidates.pkl",
            # tracklet_proposals_file="/LiZhen_team/dataset/waymo/tracklet_data/cp_vehicle_training.pkl",
            # ann_file='./data/waymo/tracklet_data/fsd_base_vehicle_training_gt_candidates.pkl',
            # tracklet_proposals_file='./data/waymo/tracklet_data/fsd_base_vehicle_training.pkl',
            # ann_file="/LiZhen_team/dataset/waymo/tracklet_data/fsd6f6e_vehicle_full_val_gt_candidates.pkl",
            # ann_file="/backup_data_2/data/waymo/kitti_format/tracklet_data/fsd6f6e_vehicle_full_training_gt_candidates.pkl",
            # tracklet_proposals_file="/LiZhen_team/dataset/waymo/tracklet_data/fsd6f6e_vehicle_full_val.pkl",
            # tracklet_proposals_file="/backup_data_2/data/waymo/kitti_format/tracklet_data/fsd6f6e_vehicle_full_training.pkl",
            pipeline=train_pipeline,
            load_interval=1,
        ),
    ),
    val=dict(
        pipeline=eval_pipeline,
        min_tracklet_points=1,
        samples_per_gpu=8,
    ),
    test=dict(
        # tracklet_proposals_file='./data/waymo/tracklet_data/fsd_base_vehicle_val.pkl',
        # tracklet_proposals_file='/LiZhen_team/dataset/waymo/tracklet_data/fsd_pastfuture_vehicle_val.pkl',
        tracklet_proposals_file="/LiZhen_team/dataset/waymo/tracklet_data/fsd_base_1f_vehicle_val.pkl",
        # tracklet_proposals_file="/LiZhen_team/dataset/waymo/tracklet_data/cp_vehicle_val.pkl",
        pipeline=test_pipeline,
        # pipeline=tta_pipeline,
        min_tracklet_points=1,
        samples_per_gpu=8,
    ),
)
log_config = dict(
    interval=50,
)

optimizer = dict(
    lr=1e-6,
)
checkpoint_config = dict(
    interval=1,  # save checkpoint every interval epochs
    max_keep_ckpts=1  # only keep max_keep_ckpts last checkpoints
)
runner = dict(type="EpochBasedRunner", max_epochs=24
              )
