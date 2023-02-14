# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)# 分割框架通常使用 SyncBN
model = dict(
    type='EncoderDecoder',# 分割器(segmentor)的名字
    pretrained='open-mmlab://resnet50_v1c',# 将被加载的 ImageNet 预训练主干网络
    backbone=dict(
        type='ResNetV1c',# 主干网络的类别。 可用选项请参考 mmseg/models/backbones/resnet.py
        depth=50,# 主干网络的深度。通常为 50 和 101。
        num_stages=4,# 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(0, 1, 2, 3), # 每个状态产生的特征图输出的索引。
        dilations=(1, 1, 2, 4), # 每一层(layer)的空心率(dilation rate)。
        strides=(1, 2, 1, 1), # 每一层(layer)的步长(stride)。
        norm_cfg=norm_cfg,
        # norm_cfg=dict(  # 归一化层(norm layer)的配置项。
        #    type='SyncBN',  # 归一化层的类别。通常是 SyncBN。
        #    requires_grad=True),   # 是否训练归一化里的 gamma 和 beta。

        norm_eval=False,# 是否冻结 BN 里的统计项。
        style='pytorch',# 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
        contract_dilation=True),# 当空洞 > 1, 是否压缩第一个空洞层。
    decode_head=dict(
        type='PSPHead',# 解码头(decode head)的类别。 可用选项请参考 mmseg/models/decode_heads。
        in_channels=2048,# 解码头的输入通道数。
        in_index=3,# 被选择的特征图(feature map)的索引。
        channels=512,# 解码头中间态(intermediate)的通道数。
        pool_scales=(1, 2, 3, 6),# PSPHead 平均池化(avg pooling)的规模(scales)。 细节请参考文章内容。
        dropout_ratio=0.1,# 进入最后分类层(classification layer)之前的 dropout 比例。
        num_classes=19,# 分割前景的种类数目。 通常情况下，cityscapes 为19，VOC为21，ADE20k 为150。
        norm_cfg=norm_cfg,  # 归一化层的配置项。
        align_corners=False,# 解码里调整大小(resize)的 align_corners 参数。
        loss_decode=dict(# 解码头(decode_head)里的损失函数的配置项。
            type='CrossEntropyLoss',# 在分割里使用的损失函数的类别。
            use_sigmoid=False,# 在分割里是否使用 sigmoid 激活。
            loss_weight=1.0)), # 解码头里损失的权重。
    auxiliary_head=dict(
        type='FCNHead',# 辅助头(auxiliary head)的种类。可用选项请参考 mmseg/models/decode_heads。
        in_channels=1024,# 辅助头的输入通道数。
        in_index=2,# 被选择的特征图(feature map)的索引。
        channels=256,# 辅助头中间态(intermediate)的通道数。
        num_convs=1,# FCNHead 里卷积(convs)的数目. 辅助头里通常为1。
        concat_input=False,# 在分类层(classification layer)之前是否连接(concat)输入和卷积的输出。
        dropout_ratio=0.1,# 进入最后分类层(classification layer)之前的 dropout 比例。
        num_classes=19,# 分割前景的种类数目。 通常情况下，cityscapes 为19，VOC为21，ADE20k 为150。
        norm_cfg=norm_cfg,# 归一化层的配置项。
        align_corners=False,# 解码里调整大小(resize)的 align_corners 参数。
        loss_decode=dict(# 辅助头(auxiliary head)里的损失函数的配置项。
            type='CrossEntropyLoss',# 在分割里使用的损失函数的类别。
            use_sigmoid=False, # 在分割里是否使用 sigmoid 激活。
            loss_weight=0.4)),# 辅助头里损失的权重。默认设置为0.4。
    # model training and testing settings
    train_cfg=dict(),# train_cfg 当前仅是一个占位符。
    test_cfg=dict(mode='whole')) # 测试模式， 选项是 'whole' 和 'sliding'. 'whole': 整张图像全卷积(fully-convolutional)测试。 'sliding': 图像上做滑动裁剪窗口(sliding crop window)。
