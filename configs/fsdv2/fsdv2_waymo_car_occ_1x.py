_base_ = [
    '../_base_/datasets/waymo-fsd-car-occ.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]

seg_voxel_size = (0.25, 0.25, 0.2)
virtual_voxel_size=(0.5, 0.5, 0.5) #(400, 400, 12)
pre_2nd_voxelization = (0.1, 0.1, 0.1)
point_cloud_range = [-80, -80, -2, 80, 80, 4]
class_names = ['Car',]
num_classes = len(class_names)
# seg_score_thresh = (0.3, 0.25, 0.25)
seg_score_thresh = (0.3, )

segmentor = dict(
    type='VoteSegmentor',

    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=5 + 2,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    middle_encoder=dict(
        type='PseudoMiddleEncoderForSpconvFSD',
    ),

    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, 640, 640],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128, # dummy
        encoder_channels=((128, ), (128, 128, ), (128, 128, ), (128, 128, 128), (256, 256, 256), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, ), (1, 1, ), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 256), (256, 256, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1), (1, 1)), # decoder paddings seem useless in SubMConv
        return_multiscale_features=True,
    ),


    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=67 + 64,
        hidden_dims=[128, 128],
        num_classes=num_classes,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='naiveSyncBN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=1.0),
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for training log
        class_names=('Car', ), # for training log
        centroid_offset=False,
    ),
)

model = dict(
    type='FSDV2',

    segmentor=segmentor,

    virtual_point_projector=dict(
        in_channels=75 + 64,
        hidden_dims=[64, 64],
        norm_cfg=dict(type='naiveSyncBN1d'),

        ori_in_channels=67 + 64,
        ori_hidden_dims=[64, 64],

        recover_in_channels=128 + 3, # with point2voxel offset
        recover_hidden_dims=[128, 128],
    ),

    multiscale_cfg=dict(
        multiscale_levels=[0, 1, 2],
        projector_hiddens=[[256, 128], [128, 128], [128, 128]],
        fusion_mode='avg',
        target_sparse_shape=[12, 320, 320],
        norm_cfg=dict(type='naiveSyncBN1d'),
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=67,
        feat_channels=[64, 128],
        voxel_size=virtual_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    backbone=dict(
        type='VirtualVoxelMixer',
        in_channels=128,
        sparse_shape=[12, 320, 320],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, ), (64, 64, ), ),
        encoder_paddings=((1, ), (1, 1,), (1, 1,), ),
        decoder_channels=((64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 1), (1, 1),), # decoder paddings seem useless in SubMConv
    ),

    bbox_head=dict(
        type='FSDV2Head',
        num_classes=num_classes,
        bbox_coder=dict(type='BasePointBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_center=dict(type='L1Loss', loss_weight=0.5),
        loss_size=dict(type='L1Loss', loss_weight=0.5),
        loss_rot=dict(type='L1Loss', loss_weight=0.2),
        in_channel=128,
        shared_mlp_dims=[256, 256],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='LN'),
        tasks=[
            dict(class_names=['Car',]),
            # dict(class_names=['Pedestrian',]),
            # dict(class_names=['Cyclist',]),
        ],
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128),  # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type='FSDSeparateHead',
            norm_cfg=dict(type='LN'),
            act='relu',
        ),
        as_rpn=True,
    ),
    roi_head=dict(
        type='GroupCorrectionHead',
        num_classes=num_classes,
        roi_extractor=dict(
             type='DynamicPointROIExtractor',
             extra_wlh=[0.5, 0.5, 0.5],
             max_inbox_point=256,
             max_all_pts=100000,
             debug=False,
             with_virtual=False,
        ),
        bbox_head=dict(
            type='FullySparseBboxHead',
            num_classes=num_classes,
            num_blocks=6,
            in_channels=[144, 144, 144, 144, 144, 144], 
            feat_channels=[[128, 128], ] * 6,
            rel_mlp_hidden_dims=[[16, 32],] * 6,
            rel_mlp_in_channels=[13, ] * 6,
            reg_mlp=[512, 512],
            cls_mlp=[512, 512],
            mode='max',
            xyz_normalizer=[20, 20, 4],
            act='gelu',
            geo_input=True,
            with_corner_loss=True,
            corner_loss_weight=1.0,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            norm_cfg=dict(type='LN', eps=1e-3),
            unique_once=True,

            loss_bbox=dict(
                type='L1Loss',
                reduction='mean',
                loss_weight=2.0),

            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0),
            cls_dropout=0.1,
            reg_dropout=0.1,
        ),
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None
    ),

    train_cfg=dict(
        score_thresh=seg_score_thresh,
        sync_reg_avg_factor=True,
        virtual_voxel_size=virtual_voxel_size,
        disable_pretrain=True,
        disable_pretrain_topks=[600, 200, 200],
        pre_2nd_voxelization=pre_2nd_voxelization,
        rpn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.5,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500,
        ),
        rcnn=dict(
            assigner=[
                dict( # Car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                ),
                dict( # Ped
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict( # Cyc
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
            ],

            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.55,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True
            ),
            cls_pos_thr=(0.8, 0.65, 0.65),
            cls_neg_thr=(0.2, 0.15, 0.15),
            sync_reg_avg_factor=True,
            sync_cls_avg_factor=True,
            corner_loss_only_car=True,
            class_names=class_names,
        )
    ),
    test_cfg=dict(
        score_thresh=seg_score_thresh,
        virtual_voxel_size=virtual_voxel_size,
        pre_2nd_voxelization=pre_2nd_voxelization,
        skip_rcnn=False,
        # rpn=dict(
        #     use_rotate_nms=True,
        #     nms_pre=-1,
        #     nms_thr=0.25,
        #     score_thr=0.1, 
        #     min_bbox_size=0,
        #     max_num=500,
        # ),
        # rcnn=dict(
        #     use_rotate_nms=True,
        #     nms_pre=-1,
        #     nms_thr=0.25,
        #     score_thr=0.1, 
        #     min_bbox_size=0,
        #     max_num=500,
        # ),
        rpn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            # nms_thr=0.25,
            nms_thr=0.7,
            score_thr=0.1, 
            min_bbox_size=0,
            max_num=500,
        ),
        rcnn=dict( # better setting
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.7,
            score_thr=0.0001, 
            min_bbox_size=0,
            max_num=500,
            rcnn_score_nms=True,
        ),
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=1,
            )
    ),
)

log_config=dict(
    interval=50,
)

custom_hooks = [
    dict(type='DisableAugmentationHook', num_last_epochs=1, skip_type_keys=('ObjectSample', 'RandomFlip3D', 'GlobalRotScaleTrans')),
    dict(type='EnableFSDDetectionHookIter', enable_after_iter=4000, threshold_buffer=0.3, buffer_iter=8000) 
]

optimizer = dict(
    lr=3e-5,
)