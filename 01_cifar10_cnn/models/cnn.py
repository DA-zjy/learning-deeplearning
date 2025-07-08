# models/cnn.py

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        # 卷积层和池化层
        self.conv_layers = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # -> 16x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 16x16x16
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # -> 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> 32x8x8
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 卷积和池化
        x = self.conv_layers(x)
        # 展平: [batch_size, 32, 8, 8] -> [batch_size, 32*8*8]
        x = x.view(x.size(0), -1)
        # 全连接层分类
        x = self.fc_layers(x)
        return x