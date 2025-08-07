import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷积模块1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # 输入通道数 (灰度图为1)
                out_channels=16,    # 输出通道数
                kernel_size=5,      # 卷积核大小
                stride=1,           # 步长
                padding=2,          # 填充 (为了保持图片尺寸不变)
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 池化层，图片尺寸减半
        )

        # 卷积模块2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 全连接分类器
        # 经过两次2x2的池化，28x28的图片变成了 7x7
        # 32是最后一个卷积层的输出通道数
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
        
        