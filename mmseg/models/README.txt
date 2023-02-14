MMSegmentation 中将语义分割模型定义为 segmentor， 一般包括 backbone、neck、head、loss 4 个核心组件，每个模块的功能如下：

预处理后的数据输入到 backbone（ 如 ResNet 和 Swin Transformer ）中进行编码并提取特征。
输出的单尺度或者多尺度特征图输入到 neck 模块中进行特征融合或者增强，典型的 neck 是 特征金字塔 (Feature Pyramid Networks， FPN)。
上述多尺度特征最终输入到 head 部分，一般包括 decoder head，auxiliary head 以及 cascade decoder head，用以预测分割结果（它们的区别我们会在下文具体介绍）。
最后一步是计算 pixel 分类的 loss，进行训练。

builder.py中使用了注册器，具体可参考https://github.com/open-mmlab/mmcv/blob/master/docs/zh_cn/understand_mmcv/registry.md
之后对于具体的类的实现就可以使用@NECKS.register_module()，从而实现字符串和类或者函数之间的映射