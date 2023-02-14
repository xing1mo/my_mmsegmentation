./mmseg 里面是 MMSegmentation 的算法库，包括核心组件、数据集处理、分割模型代码和面向用户的 API 接口。

MMSegmentation 的算法库有 3 个关键组件：
1../mmseg/apis/，用于训练和测试的接口
2../mmseg/models/，用于分割网络模型的具体实现
3../mmseg/datasets/，用于数据集处理

介绍算法模型相关的代码，因此涉及内容主要在 ./mmseg/models 里面。