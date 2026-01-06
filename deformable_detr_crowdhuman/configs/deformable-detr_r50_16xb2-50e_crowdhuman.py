# 基于官方 Deformable DETR COCO 配置，改成 CrowdHuman 版本
# - 数据集：CrowdHuman（COCO 格式 train.json / val.json）
# - 迁移学习：从 COCO 预训练的 Deformable DETR 初始化
# - 主要设置：
#   * 随机多尺度训练（原始 Deformable DETR 标配）
#   * AdamW + Warmup + MultiStepLR
#   * AMP 混合精度训练
#   * EMA 权重滑动平均
#   * 对 backbone / sampling_offsets / reference_points 使用更小 LR

data_root = '/home/didu/datasets/CrowdHuman/'
backend_args = None  # 显式定义，避免 _base_ 相关问题

_base_ = [
    '../_base_/datasets/crowdhuman.py',   # 自己写的 CrowdHuman 数据集配置
    '../_base_/default_runtime.py'
]

# ======================
# 1. 模型结构
# ======================

model = dict(
    type='DeformableDETR',
    num_queries=300,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,                     # 冻结第一阶段，迁移学习更稳
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.1))),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=1,  # CrowdHuman 只有 person 一个类别
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # 训练/测试配置
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100)
)

# ======================
# 2. 训练 pipeline（多尺度 + 随机裁剪）
# ======================

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),

    dict(
        type='RandomChoice',
        transforms=[
            # 分支 1：标准多尺度 Resize（安全稳定）
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            # 分支 2：大尺度 + 适度裁剪（比原版更保守）
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(600, 1333), (700, 1333), (800, 1333)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=False),  # ❗ 不允许裁空图
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),

    dict(type='PackDetInputs')
]

# ======================
# 3. 训练 dataloader（只覆盖 train，val/test 用 base 里的）
# ======================

train_dataloader = dict(
    batch_size=2,
    num_workers=8,   # 13900ES + 32G 内存，8 已经很稳
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='Images/'),
        metainfo=dict(
            classes=('person', ),
            palette=[(220, 20, 60)]
        ),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args)
)

# ======================
# 4. 优化器 / 学习率 / 训练循环
# ======================

max_epochs = 80  # 大约 2 天左右（3090，batch=2）

optim_wrapper = dict(
    type='AmpOptimWrapper',  # AMP 混合精度训练
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }),
    loss_scale='dynamic'  # AMP 动态 loss scale
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 学习率调度：
# - 前 1000 iter 线性 warmup（从 1e-3 * lr 到 lr）
# - 之后使用 MultiStepLR，在 40 / 55 / 65 epoch 衰减
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        begin=0,
        end=1000,            # 按 iter 计数，前 1000 iter warmup
        by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40, 55, 65],
        gamma=0.1)
]

# ======================
# 5. EMA Hook
# ======================

custom_hooks = [
    dict(
        type='EMAHook',
        momentum=2e-4,
        priority='ABOVE_NORMAL')
]

# ======================
# 6. 自动 LR 缩放（用不到也没关系）
# ======================

auto_scale_lr = dict(base_batch_size=32)

# ======================
# 7. 从 COCO 预训练权重迁移
# ======================

load_from = '/home/didu/checkpoints/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth'



