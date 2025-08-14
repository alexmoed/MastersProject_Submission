_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 2  # bs: total bs in all gpus
num_worker = 3
mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
enable_amp = True
evaluate = False

weight = '/content/drive/MyDrive/Pointcept/exp/scannet/insseg-pointgroup-sonata_v2/model/epoch_740.pth'
resume = '/content/drive/MyDrive/Pointcept/exp/scannet/insseg-pointgroup-sonata_v2/model/epoch_740.pth'
# Define class names and instance segmentation specific params
class_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]
num_classes = 20
segment_ignore_index = (-1, 0, 1)  # Instance segmentation specific

# model settings - Merged SONATA backbone with PointGroup head
model = dict(
    type="PG-v1m2",  # Changed from DefaultSegmentorV2 to PointGroup
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
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
        freeze_encoder=True,
    ),
    backbone_out_channels=64,
    # PointGroup specific parameters
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=300,
    cluster_propose_points=100,
    cluster_min_points=50,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings - Using SONATA's settings
epoch = 800
eval_epoch = 800
optimizer = dict(type="AdamW", lr=0.004, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "data/scannet"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=1.0
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=49768, mode="random"),  # Using SONATA's larger point count
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # Instance segmentation specific transform
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",  # Instance specific
                    "instance_centroid",  # Instance specific
                    "bbox",  # Instance specific
                ),
                feat_keys=("coord", "color", "normal"),  # Using SONATA's feature keys
            ),
        ],
        test_mode=False,
        loop=3,  # From SONATA config
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",  # Instance specific
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,  # From SONATA
            ),
            dict(type="SphereCrop", point_max=49768, mode='center'),  # Keep original for val
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "inverse",  # Added from SONATA
                ),
                feat_keys=("coord", "color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=49768, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "name",
                ),
                feat_keys=("coord", "color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
)

# hooks - Using SONATA's checkpoint loader
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",  # Instance segmentation evaluator
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=5),
    dict(type="PreciseEvaluator", test_last=False),
]

# Tester
test = dict(
    type="InsSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=False,
)

start_epoch = 741
