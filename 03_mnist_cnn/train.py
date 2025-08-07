# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm # 引入tqdm，一个强大的进度条工具

# 从我们自己的模块中导入
import config
from model import SimpleCNN
from dataset import get_dataloaders

def train_one_epoch(model, device, train_loader, optimizer, criterion):
    """训练一个epoch"""
    model.train() # 设置为训练模式
    running_loss = 0.0
    # 使用tqdm来显示进度条
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate(model, device, test_loader, criterion):
    """在测试集上评估模型"""
    model.eval() # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def main():
    """主函数"""
    # 确保模型保存目录存在
    if not os.path.exists(config.CHECKPOINT_PATH):
        os.makedirs(config.CHECKPOINT_PATH)
        
    # 获取数据加载器
    train_loader, test_loader = get_dataloaders()
    
    # 实例化模型并移动到设备
    model = SimpleCNN().to(config.DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print("开始训练...")
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, config.DEVICE, train_loader, optimizer, criterion)
        test_loss, accuracy = evaluate(model, config.DEVICE, test_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:.2f}%")
        
        # 保存模型
        torch.save(model.state_dict(), f"{config.CHECKPOINT_PATH}/mnist_cnn_epoch_{epoch+1}.pth")

    print("训练完成！")

# 这是一个Python脚本的入口点。
# 当你直接运行 `python train.py` 时，__name__ 的值就是 "__main__"，于是 main() 函数就会被调用。
# 如果这个文件被其他脚本作为模块导入，__name__ 就不是 "__main__"，main() 就不会被执行。
if __name__ == '__main__':
    main()