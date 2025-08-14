from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_200,
)

# Checkpoint paths
weight = '/content/drive/MyDrive/sonata/Checkpoint/pretrain-sonata-v1m1-0-base.pth'
resume = False

# Training settings
evaluate = False
test_only = False
seed = 28434507
num_worker = 2
batch_size = 3
batch_size_val = None
batch_size_test = None
epoch = 800
eval_epoch = 800
clip_grad = 3.0
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = True
find_unused_parameters = False
enable_wandb = True
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8

# Parameter-specific learning rates
param_dicts = [dict(keyword='block', lr=0.0004)]

# Hooks configuration
hooks = [
    dict(
        type='CheckpointLoader',
        keywords='module.student.backbone',
        replacement='module.backbone'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(
        type='InsSegEvaluator',
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=5),
    dict(type='PreciseEvaluator', test_last=False)
]

# Trainer configuration
train = dict(type='DefaultTrainer')
test = dict(
    type='InsSegTester',
    verbose=False,
    segment_ignore_index=(-1, 0, 1),
    instance_ignore_index=-1)

# IMPORTANT: Use ScanNet200 class information
num_classes = 200  # Changed from 20 to 200
class_names = CLASS_LABELS_200  # Use the imported 200 class labels
segment_ignore_index = (-1, 0, 1)

# Model configuration - PointGroup with Sonata (PT-v3m2) backbone
model = dict(
    type='PG-v1m2',
    backbone=dict(
        type='PT-v3m2',
        in_channels=9,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=True),
    backbone_out_channels=64,
    semantic_num_classes=200,  # Changed from 20 to 200
    semantic_ignore_index=-1,
    segment_ignore_index=(-1, 0, 1),
    instance_ignore_index=-1,
    cluster_thresh=2.2,
    cluster_closed_points=300,
    cluster_propose_points=100,
    cluster_min_points=50,
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])

# Optimizer configuration
optimizer = dict(type='AdamW', lr=0.005, weight_decay=0.02)

# Scheduler configuration
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)

# Dataset settings
dataset_type = "ScanNet200Dataset"
data_root = "data/scannet"

# Data configuration for ScanNet200
data = dict(
    num_classes=200,  # Changed from 20 to 200
    ignore_index=-1,
    names=CLASS_LABELS_200,  # Use the imported 200 class labels
    train=dict(
        type='ScanNet200Dataset',
        split='train',
        data_root='data/scannet',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomDropout',
                dropout_ratio=0.0,
                dropout_application_ratio=1.0),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.1),
            dict(type='RandomJitter', sigma=0.009, clip=0.02),
            dict(
                type='ElasticDistortion',
                distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=35768, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, 0, 1),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance',
                      'instance_centroid', 'bbox'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False,
        loop=3),
    val=dict(
        type='ScanNet200Dataset',
        split='val',
        data_root='data/scannet',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    segment='origin_segment',
                    instance='origin_instance')),
            dict(
                type='GridSample',
                grid_size=0.009,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='SphereCrop', point_max=38768, mode='center'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, 0, 1),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance',
                      'origin_coord', 'origin_segment', 'origin_instance',
                      'instance_centroid', 'bbox', 'inverse'),
                feat_keys=('coord', 'color', 'normal'),
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False),
    test=dict(
        type='ScanNet200Dataset',
        split='val',
        data_root='data/scannet',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    segment='origin_segment',
                    instance='origin_instance')),
            dict(
                type='GridSample',
                grid_size=0.009,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=38768, mode='center'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, 0, 1),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance',
                      'origin_coord', 'origin_segment', 'origin_instance',
                      'instance_centroid', 'bbox', 'name'),
                feat_keys=('coord', 'color', 'normal'),
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False))

start_epoch = 681