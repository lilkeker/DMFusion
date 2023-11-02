point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset'
data_root = '/data0/yy_datasets/nuscenes_bevdet/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root+'bevdetv2-nuscenes_dbinfos_train.pkl',
    # info_path=data_root+'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk')))
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4]),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='Coordinate_change', downsample=1, grid_config=None),
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
    # dict(type='ObjectSample', db_sampler=db_sampler),
##### 测试加入数据增强后是否对齐
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]
    ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
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
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4]),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='Coordinate_change', downsample=1, grid_config=None),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
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
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk'),
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='CBGSDataset',
        data_root=data_root,
        ann_file=data_root + 'bevdetv2sweep10-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            data_root=data_root,
            ann_file=data_root + 'bevdetv2sweep10-nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            load_interval=1,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR',
            type=dataset_type,
            modality=input_modality,
            img_info_prototype='bevdet4d',
            multi_adj_frame_id_cfg=(1, 9, 1)
        )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2sweep10-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 9, 1)),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2sweep10-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 9, 1)))


dist_params = dict(backend='nccl')
log_level = 'INFO'

opencv_num_threads = 0
mp_start_method = 'fork'
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
voxel_size = [0.075, 0.075, 0.2]
numC_Trans = 80
multi_adj_frame_id_cfg = (1, 9, 1)
model = dict(
    type='TransFusionDetector',
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.075, 0.075, 0.2],
        max_voxels=(120000, 160000),
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
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
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(1, 9, 1))
test_data_config = dict(
    pipeline=test_pipeline,
    ann_file= data_root + 'bevdetv2sweep10-nuscenes_infos_val.pkl',
    type=dataset_type,
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
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(1, 9, 1))
key = 'test'
optimizer = dict(
    type='AdamW',
    lr=0.000002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=5)

work_dir = "/home/yy/bevdet/work_dir/transfusion_best_resume"
load_from = '/home/yy/bevdet/work_dir/transfusion_best_resume/best_transfusion.pth'
resume_from = None
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(
    interval=1,
    pipeline=test_pipeline)
