# 深度学习入门项目

这是一个深度学习入门项目，包含了两个经典的机器学习任务实现。

## 项目结构

- `01_cifar10_cnn/` - CIFAR-10图像分类（CNN）
- `02_mnist_mlp/` - MNIST手写数字识别（MLP）
- `shared/` - 共享工具和资源

## 环境要求

```bash
pip install torch torchvision matplotlib numpy
```

## 使用方法

### CIFAR-10 CNN分类
```bash
cd 01_cifar10_cnn
python train.py
```

### MNIST MLP
查看 `02_mnist_mlp/notebooks/` 中的Jupyter笔记本

## 注意事项

由于模型文件和数据集较大，没有包含在仓库中。运行时会自动下载数据集。