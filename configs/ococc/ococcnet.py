_base_ = [
    "../_base_/datasets/waymo-tracklet-vehicle.py",
    "../_base_/schedules/cosine_2x.py",
    "../_base_/default_runtime.py",
]
data_root = "data/waymo/kitti_format/"
dataset_type = "WaymoTrackletDatasetWithOcc"
seg_voxel_size = (0.2, 0.2, 0.2)
point_cloud_range = [-204.8, -204.8, -4.0, 204.8, 204.8, 8.0]
class_names = [
    "Car",
]
num_classes = len(class_names)
occ_voxel_size = 0.2
ae_voxel_size = 0.2
reg_len = 32
model = dict(
    type="TrackletDetectorOCC",
    roi_head=dict(
        type="TrackletRoIHeadOCC",
        num_classes=num_classes,
        general_cfg=dict(
            with_roi_scores=True,
        ),
        history_only=True,
        roi_extractor=dict(
            type="TrackletPointRoIExtractor",
            extra_wlh=[0.5, 0.5, 0.5],
            max_inbox_point=4096,
            max_all_point=(300000, 600000),
            debug=False,
            combined=False,
        ),
        bbox_head=dict(
            type="OccBBoxHead",
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
            with_rel_mlp=True,
            with_cluster_center=False,
            with_distance=False,
            mode="max",
            xyz_normalizer=[20, 20, 4],
            geo_input=True,
            dropout=0,
            unique_once=True,
            occ_ae_head=dict(
                type="OccAutoEncoder",
                backbone=dict(
                    type="SIR",
                    num_blocks=6,
                    in_channels=[15, 131, 131, 131, 131, 131],
                    feat_channels=[
                        [128, 128],
                    ]
                    * 6,
                    rel_mlp_hidden_dims=[
                        [16, 32],
                    ]
                    * 6,
                    with_rel_mlp=True,
                    with_cluster_center=False,
                    with_distance=False,
                    norm_cfg=dict(type="LN", eps=1e-3),
                    mode="max",
                    # xyz_normalizer=[20, 20, 4],
                    xyz_normalizer=[1, 1, 1],
                    act="gelu",
                    dropout=0,
                    unique_once=True,
                ),
                voxel_size=ae_voxel_size,
                loss_occ_ae=dict(
                    type="CrossEntropyLoss",
                    reduction="none",
                    use_sigmoid=True,
                    loss_weight=1.0,
                    # alpha=0.05,
                ),
                online_sample_size=-1,
                balance_sample=True,
                occ_decoder=dict(
                    roi_feature_channels=1536,
                    occ_mlp=[512, 1024, 1024],
                    use_positional_encoding=True,
                    pos_encode_L=10,
                    norm_pos=True,
                    norm_cfg=dict(type="LN", eps=1e-3),
                    act="gelu",
                    occ_dropout=0.1,
                    cls_dim=1,
                    pos_thresh=0.5,
                    use_ln=True,
                ),
                with_voxelize_centers=True,
                compensate_encoder_coors=True,
            ),
            num_classes=num_classes,
            roi_feature_channels=1536,
            attn_num_head=4,
            # attn_ffn_dim=2048,
            attn_ffn_dim=512,
            attn_dropout=0.1,
            loss_occ_comp=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                reduction="none",
                loss_weight=1,
            ),
            bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
            occ_label_thresh=0.4,
            cls_mlp=[512, 512],
            reg_mlp=[512, 512],
            # latent_mlp=[1024, 1024],
            latent_mlp=[2048, 2048],
            fusion_mlp=[2048, 2048],
            act="gelu",
            norm_cfg=dict(type="LN", eps=1e-3),
            loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=2.0),
            loss_cls=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                reduction="mean",
                loss_weight=1.0,
            ),
            cls_dropout=0.1,
            reg_dropout=0.1,
            latent_dropout=0.1,
            fusion_dropout=0.1,
            with_roi_pos_encoding=True,
            roi_pos_enc_mlp=[512, 512],
            num_enc_layers=3,
            fixed_ae=False,
            fused_mode="concat",
            rcnn_trans=False,
        ),
        pretrained=None,
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
        fixed_length=True,
        num_occ_per_tracklet=-1,
        random_shift_frame_inds=True,
        keep_frame_inds=False,
        residual_loss=False,
        contrastive_loss=False,
        no_loss_for_outside=False,
        no_loss_for_observed_feats=False,
        contrastive_loss_weight=1.0,
    ),
    test_cfg=dict(
        batch_inference=True,
        test_occ_iou=True,
        iou_chunk_size=10,
        ignore_outside_occ=True,
        test_baseline=False,
    ),
)

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
    dict(type="LoadAnnotationsOcc", compute_score=False),
    dict(
        type="RandomSampleOccPoints",
        num_sample_points=512,
        pos_sample_weight=0.5,
        voxel_size=occ_voxel_size,
        use_unknown=False,
        use_potential=False,
        balance_sample=True,
        weighted_sample=True,
    ),
    # dict(type="JitterOccCenter", voxel_size=occ_voxel_size),
    dict(type="TrackletRegularization", reg_len=reg_len),
    dict(
        type="TrackletPoseTransform",
        concat=False,
    ),
    # dict(
    #     type="TrackletNoise",
    #     center_noise_cfg=dict(max_noise=[1, 1, 0.25], consistent=False),
    #     size_noise_cfg=dict(max_noise=[0.3, 0.3, 0.2], consistent=False),
    #     yaw_noise_cfg=dict(max_noise=0.2, consistent=False),
    # ),
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
    dict(type="TrackletOccFormatBundle", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "points",
            "pts_frame_inds",
            "tracklet",
            "gt_tracklet_candidates",
            "occ_labels",
            "occ_labels_scores",
        ],
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
        type="LoadTrackletAnnotations",
    ),
    dict(type="LoadAnnotationsOcc"),
    dict(
        type="RandomSampleOccPoints",
        num_sample_points=-1,
        pos_sample_weight=0.5,
        voxel_size=occ_voxel_size,
        use_unknown=False,
        use_potential=False,
        balance_sample=True,
        weighted_sample=True,
    ),
    dict(
        type="TrackletPoseTransform",
        concat=False,
    ),
    # dict(
    #     type="TrackletNoise",
    #     center_noise_cfg=dict(max_noise=[0.1, 0.1, 0.1], consistent=False),
    #     # size_noise_cfg=dict(max_noise=[0.2, 0.2, 0.1], consistent=False),
    #     yaw_noise_cfg=dict(max_noise=0.1, consistent=False),
    # ),
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
    dict(type="TrackletOccFormatBundle", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "points",
            "pts_frame_inds",
            "tracklet",
            "gt_tracklet_candidates",
            "occ_labels",
            "occ_labels_scores",
        ],
    ),
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
            dict(type="TrackletOccFormatBundle", class_names=class_names),
            dict(type="Collect3D", keys=["points", "pts_frame_inds", "tracklet"]),
        ],
    ),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = test_pipeline
train_data1 = dict(
    type="RepeatDataset",
    times=1,
    # type="ClassBalancedDataset",
    # oversample_thr=0.8,
    filter_empty_gt=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="data/waymo/tracklet_data/fsd_base_1f_vehicle_training_gt_candidates.pkl",
        tracklet_proposals_file="data/waymo/tracklet_data/fsd_base_1f_vehicle_training.pkl",
        occ_anno_root="data/waymo/waymo_occ_gt/waymo_occ_gt/training",
        pose_file=data_root + "poses.pkl",
        pipeline=train_pipeline,
        load_interval=1,
        box_type_3d="LiDAR",
        min_tracklet_points=100,
        min_tracklet_length=reg_len,
        # min_tracklet_length=8,
        classes=class_names,
    ),
)
train_data2 = dict(
    type="RepeatDataset",
    times=1,
    # type="ClassBalancedDataset",
    # oversample_thr=0.8,
    filter_empty_gt=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="data/waymo/tracklet_data/cp_vehicle_training_gt_candidates.pkl",
        tracklet_proposals_file="data/waymo/tracklet_data/cp_vehicle_training.pkl",
        occ_anno_root="data/waymo/waymo_occ_gt/waymo_occ_gt/training",
        pose_file=data_root + "poses.pkl",
        pipeline=train_pipeline,
        load_interval=1,
        box_type_3d="LiDAR",
        min_tracklet_points=100,
        min_tracklet_length=reg_len,
        # min_tracklet_length=8,
        classes=class_names,
    ),
)
train_data3 = dict(
    type="RepeatDataset",
    times=1,
    # type="ClassBalancedDataset",
    # oversample_thr=0.8,
    filter_empty_gt=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="data/waymo/tracklet_data/gt_base_vehicle_training_gt_candidates.pkl",
        tracklet_proposals_file="data/waymo/tracklet_data/gt_base_vehicle_training.pkl",
        occ_anno_root="data/waymo/waymo_occ_gt/waymo_occ_gt/training",
        pose_file=data_root + "poses.pkl",
        pipeline=train_pipeline,
        load_interval=1,
        box_type_3d="LiDAR",
        min_tracklet_points=100,
        min_tracklet_length=reg_len,
        # min_tracklet_length=8,
        classes=class_names,
    ),
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=[train_data1,train_data2,train_data3],

    val=dict(
        type=dataset_type,
        data_root=data_root,
        occ_anno_root="data/waymo/waymo_occ_gt/waymo_occ_gt/training",
        ann_file="data/waymo/tracklet_data/fsd_base_1f_vehicle_val_gt_candidates.pkl",
        tracklet_proposals_file="data/waymo/tracklet_data/fsd_base_1f_vehicle_val.pkl",
        pose_file=data_root + "poses.pkl",
        pipeline=eval_pipeline,
        load_interval=1,
        box_type_3d="LiDAR",
        min_tracklet_points=100,
        min_tracklet_length=200,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        occ_anno_root="data/waymo/waymo_occ_gt/waymo_occ_gt/training",
        ann_file="data/waymo/tracklet_data/fsd_base_1f_vehicle_val_gt_candidates.pkl",
        tracklet_proposals_file="data/waymo/tracklet_data/fsd_base_1f_vehicle_val.pkl",

        pose_file=data_root + "poses.pkl",
        pipeline=eval_pipeline,
        load_interval=1,
        box_type_3d="LiDAR",
        min_tracklet_points=-1,
        min_tracklet_length=-1,
    ),
)
log_config = dict(
    interval=50,
)

optimizer = dict(
    lr=1e-6,
)
# optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
find_unused_parameters = False
checkpoint_config = dict(
    interval=1,  # save checkpoint every interval epochs
    max_keep_ckpts=1,  # only keep max_keep_ckpts last checkpoints
)
# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=24)
evaluation = dict(interval=100)
