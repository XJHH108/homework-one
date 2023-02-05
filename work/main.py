model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

data =dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train = dict (
        data_prefix= 'data/flower/train',
        ann_file='data/flower/train.txt',
        classes='data/flower/classes.txt'
    ),
    val=dict(
        data_prefix= 'data/flower/val',
        ann_file='data/flower/val.txt',
        classes='data/flower/classes.txt'
    )
)

optimizer = dict(type='SGD', lr=0.0001, momentum=0.8,weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    step=[1]
)

runner = dict (type = 'EpochBasedRunner', max_epochs=100)

load_from = '/run/mmclassification/configs/resnet18/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'