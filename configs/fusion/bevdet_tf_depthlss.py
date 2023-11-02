# mAP: 0.6893
# mATE: 0.2962
# mASE: 0.2499
# mAOE: 0.3032
# mAVE: 0.3047
# mAAE: 0.1883
# NDS: 0.7104
# Eval time: 96.9s
#
# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.893   0.174   0.144   0.048   0.318   0.190
# truck   0.654   0.337   0.191   0.083   0.291   0.218
# bus     0.757   0.343   0.176   0.045   0.529   0.248
# trailer 0.445   0.530   0.216   0.516   0.220   0.191
# construction_vehicle    0.306   0.704   0.433   0.888   0.139   0.306
# pedestrian      0.878   0.150   0.280   0.402   0.232   0.104
# motorcycle      0.786   0.208   0.231   0.262   0.491   0.235
# bicycle 0.663   0.183   0.249   0.426   0.218   0.013
# traffic_cone    0.786   0.136   0.307   nan     nan     nan
# barrier 0.725   0.197   0.272   0.059   nan     nan





point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset'
data_root = '/data0/yy_datasets/nuscenes_bevdet/'
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(896, 1600),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=-0.0)
grid_config = dict(
    x=[-54.0, 54.0, 0.6],
    y=[-54.0, 54.0, 0.6],
    z=[-5, 3, 8],
    depth=[1.0, 60.0, 0.5])
voxel_size = [0.075, 0.075, 0.2]
numC_Trans = 80
multi_adj_frame_id_cfg = (1, 4, 1)
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=False,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(896, 1600),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0)),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk'),
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-54.0, 54.0, 0.6],
            y=[-54.0, 54.0, 0.6],
            z=[-5, 3, 8],
            depth=[1.0, 60.0, 0.5])),
    dict(type='PointToEGO', downsample=1, grid_config=None),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54, -54, -5.0, 54, 54, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54, -54, -5.0, 54, 54, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'points', 'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth'
        ])
]
test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(896, 1600),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        is_train=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk'),
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-54.0, 54.0, 0.6],
            y=[-54.0, 54.0, 0.6],
            z=[-5, 3, 8],
            depth=[1.0, 60.0, 0.5])),
    dict(type='PointToEGO', downsample=1, grid_config=None),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs','gt_depth'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='NuScenesDataset',
        data_root='/data0/yy_datasets/nuscenes_bevdet/',
        ann_file=
        '/data0/yy_datasets/nuscenes_bevdet/bevdetv2sweep10-nuscenes_infos_train.pkl',
        load_interval=1,
        pipeline=train_pipeline,
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        img_info_prototype='bevdet'),
    val=dict(
        type='NuScenesDataset',
        data_root='/data0/yy_datasets/nuscenes_bevdet/',
        ann_file=
        '/data0/yy_datasets/nuscenes_bevdet/bevdetv2sweep10-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset',
        data_root='/data0/yy_datasets/nuscenes_bevdet/',
        ann_file=
        '/data0/yy_datasets/nuscenes_bevdet/bevdetv2sweep10-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=1,
    pipeline=test_pipeline)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='BEVDET_FUSION_TF',
    fuse_bev=True,
    img_backbone=dict(
        type='CBSwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    img_neck=dict(
        type='FPNC',
        final_dim=(896, 1600),
        downsample=8,
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        outC=256,
        use_adp=True,
        num_outs=5),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=dict(
            x=[-54.0, 54.0, 0.6],
            y=[-54.0, 54.0, 0.6],
            z=[-5, 3, 8],
            depth=[1.0, 60.0, 0.5]),
        input_size=(896, 1600),
        in_channels=256,
        out_channels=256,
        downsample=8),
    fuser=dict(type='SEFuser', in_channels=768, out_channels=512),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.075, 0.075, 0.2],
        max_voxels=(90000, 120000),
        point_cloud_range=[-54, -54, -5.0, 54, 54, 3.0]),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            voxel_size=[0.075, 0.075],
            out_size_factor=8,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0)),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=8,
            pc_range=[-54.0, -54.0],
            voxel_size=[0.075, 0.075],
            nms_type=None)))
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
share_data_config = dict(
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=True,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet')
test_data_config = dict(
    pipeline=[
        dict(
            type='PrepareImageInputs',
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(512, 1408),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            is_train=False),
        dict(
            type='LoadAnnotationsBEVDepth',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            is_train=False),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=9,
            use_dim=[0, 1, 2, 3, 4],
            file_client_args=dict(backend='disk'),
            pad_empty_sweeps=True,
            remove_close=True),
        dict(type='PointToEGO', downsample=1, grid_config=None),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(type='Collect3D', keys=['points', 'img_inputs'])
            ])
    ],
    ann_file=
    '/data0/yy_datasets/nuscenes_bevdet/bevdetv2sweep10-nuscenes_infos_val.pkl',
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=True,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet')
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[2, 3])
runner = dict(type='EpochBasedRunner', max_epochs=10)
work_dir = '/home/yy/bevdet/work_dir/bevdet_tf_depthlss'
load_from = '/home/yy/bevdet/work_dir/bevdet_tf_nodepthlss/epoch_7.pth'
find_unused_parameters = True
gpu_ids = range(0, 3)
