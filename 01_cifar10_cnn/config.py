# config.py

import torch

# 训练配置
DEVICE = torch.device("cuda:3")  # 使用GPU 3
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 32        # 批处理大小
EPOCHS = 10            # 训练轮数
MOMENTUM = 0.9         # SGD 优化器的动量参数

# 数据集配置
# CIFAR-10 数据集的均值和标准差（用于归一化）
# 这三个值分别是 R, G, B 三个通道的
DATA_MEAN = (0.5, 0.5, 0.5)
DATA_STD = (0.5, 0.5, 0.5)

# 类别名称
CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')