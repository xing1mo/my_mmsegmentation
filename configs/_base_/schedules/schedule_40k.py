# optimizer
optimizer = dict( # 用于构建优化器的配置文件。支持 PyTorch 中的所有优化器，同时它们的参数与PyTorch里的优化器参数一致。
    type='SGD',# 优化器种类，更多细节可参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13。
    lr=0.01,# 优化器的学习率，参数的使用细节请参照对应的 PyTorch 文档。
    momentum=0.9,# 动量 (Momentum)
    weight_decay=0.0005)# SGD 的衰减权重 (weight decay)。
optimizer_config = dict()# 用于构建优化器钩 (optimizer hook) 的配置文件，执行细节请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8。
# learning policy
lr_config = dict(policy='poly',# 调度流程的策略，同样支持 Step, CosineAnnealing, Cyclic 等. 请从 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9 参考 LrUpdater 的细节。
                 power=0.9,# 多项式衰减 (polynomial decay) 的幂。
                 min_lr=1e-4,# 用来稳定训练的最小学习率。
                 by_epoch=False)# 是否按照每个 epoch 去算学习率。
# runtime settings
runner = dict(type='IterBasedRunner',# 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner)。
              max_iters=40000)# 全部迭代轮数大小，对于 EpochBasedRunner 使用 `max_epochs` 。
checkpoint_config = dict(# 设置检查点钩子 (checkpoint hook) 的配置文件。执行时请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py。
    by_epoch=False,# 是否按照每个 epoch 去算 runner。
    interval=4000)# 保存的间隔
evaluation = dict(# 构建评估钩 (evaluation hook) 的配置文件。细节请参考 mmseg/core/evaluation/eval_hook.py。
    interval=4000,# 评估的间歇点
    metric='mIoU',# 评估的指标
    pre_eval=True)
