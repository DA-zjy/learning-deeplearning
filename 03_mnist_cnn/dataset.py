# dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, DATASET_PATH

def get_dataloaders():
    """返回训练和测试的数据加载器"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root=DATASET_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.MNIST(root=DATASET_PATH, train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader