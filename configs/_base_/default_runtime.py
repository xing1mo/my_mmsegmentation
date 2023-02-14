# yapf:disable
log_config = dict(# 注册日志钩 (register logger hook) 的配置文件。
    interval=50,# 打印日志的间隔
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')# 同样支持 Tensorboard 日志
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')# 用于设置分布式训练的参数，端口也同样可被设置。
log_level = 'INFO'# 日志的级别。
load_from = None# 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间。
resume_from = None# 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train', 1)]# runner 的工作流程。 [('train', 1)] 意思是只有一个工作流程而且工作流程 'train' 仅执行一次。根据 `runner.max_iters` 工作流程训练模型的迭代轮数为40000次。
cudnn_benchmark = True# 是否是使用 cudnn_benchmark 去加速，它对于固定输入大小的可以提高训练速度
