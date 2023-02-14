# dataset settings
dataset_type = 'CityscapesDataset'# 数据集类型，这将被用来定义数据集。
data_root = 'data/cityscapes/'# 数据的根路径。
img_norm_cfg = dict(# 图像归一化配置，用来归一化输入的图像。
    mean=[123.675, 116.28, 103.53], # 预训练里用于预训练主干网络模型的平均值。
    std=[58.395, 57.12, 57.375],# 预训练里用于预训练主干网络模型的标准差。
    to_rgb=True)# 预训练里用于预训练主干网络的图像的通道顺序。
crop_size = (512, 1024)# 训练时的裁剪大小
train_pipeline = [#训练流程
    dict(type='LoadImageFromFile'),# 第1个流程，从文件路径里加载图像。
    dict(type='LoadAnnotations'),# 第2个流程，对于当前图像，加载它的注释信息。
    dict(type='Resize', # 变化图像和其注释大小的数据增广的流程。
         img_scale=(2048, 1024),# 图像的最大规模。
         ratio_range=(0.5, 2.0)),# 数据增广的比例范围。
    dict(type='RandomCrop', # 随机裁剪当前图像和其注释大小的数据增广的流程。
         crop_size=crop_size,# 随机裁剪图像生成 patch 的大小。
         cat_max_ratio=0.75), # 单个类别可以填充的最大区域的比例。
    dict(type='RandomFlip',# 翻转图像和其注释大小的数据增广的流程。
         prob=0.5),# 翻转图像的概率
    dict(type='PhotoMetricDistortion'),# 光学上使用一些方法扭曲当前图像和其注释的数据增广的流程。
    dict(type='Normalize',# 归一化当前图像的数据增广的流程。
         **img_norm_cfg),
    dict(type='Pad',# 填充当前图像到指定大小的数据增广的流程。
         size=crop_size,# 填充的图像大小。
         pad_val=0,# 图像的填充值。
         seg_pad_val=255),# 'gt_semantic_seg'的填充值。
    dict(type='DefaultFormatBundle'), # 流程里收集数据的默认格式捆。
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),# 决定数据里哪些键被传递到分割器里的流程。
]
test_pipeline = [
    dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像。
    dict(
        type='MultiScaleFlipAug',# 封装测试时数据增广(test time augmentations)。
        img_scale=(2048, 1024),# 决定测试时可改变图像的最大规模。用于改变图像大小的流程。
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,# 测试时是否翻转图像。
        transforms=[
            dict(type='Resize',# 使用改变图像大小的数据增广。
                 keep_ratio=True),# 是否保持宽和高的比例，这里的图像比例设置将覆盖上面的图像规模大小的设置。
            dict(type='RandomFlip'),# 考虑到 RandomFlip 已经被添加到流程里，当 flip=False 时它将不被使用。
            dict(type='Normalize', **img_norm_cfg),# 归一化配置项，值来自 img_norm_cfg。
            dict(type='ImageToTensor', keys=['img']), # 将图像转为张量
            dict(type='Collect', keys=['img']),# 收集测试时必须的键的收集流程。
        ])
]
data = dict(
    samples_per_gpu=2,# 单个 GPU 的 Batch size
    workers_per_gpu=2,# 单个 GPU 分配的数据加载线程数
    train=dict(# 训练数据集配置
        type=dataset_type,
        data_root=data_root, # 数据集的根目录。
        img_dir='leftImg8bit/train',# 数据集图像的文件夹。
        ann_dir='gtFine/train',# 数据集注释的文件夹。
        pipeline=train_pipeline), #流程， 由之前创建的 train_pipeline 传递进来。
    val=dict(# 验证数据集的配置
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),# 由之前创建的 test_pipeline 传递的流程。
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
