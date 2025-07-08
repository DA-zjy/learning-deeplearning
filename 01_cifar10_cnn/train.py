# train.py

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 从我们自己的模块中导入
import config
from models.cnn import SimpleCNN
from utils.training import train_one_epoch, evaluate
from utils.visualization import TrainingLogger

def main():
    # 1. 加载和预处理数据
    print("正在准备数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.DATA_MEAN, config.DATA_STD)
    ])

    # 加载完整的训练集
    print("正在分割数据集...")
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 拆分训练集为训练集和验证集
    train_size = int(0.8 * len(full_trainset))  # 80% 用于训练 (40,000张)
    val_size = len(full_trainset) - train_size   # 20% 用于验证 (10,000张)
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)  # 验证集不需要打乱
    
    # 测试集保持不变
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"数据集准备完成！")
    print(f"训练集大小: {len(trainset)}")
    print(f"验证集大小: {len(valset)}")
    print(f"测试集大小: {len(testset)}")

    # 2. 实例化模型、损失函数和优化器
    print("正在创建模型...")
    model = SimpleCNN(num_classes=len(config.CLASSES)).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    print("模型创建完成！")

    # 3. 创建训练记录器
    logger = TrainingLogger(save_dir="./results")
    
    # 4. 训练和评估循环
    print(f"开始在 {config.DEVICE} 上训练...")
    best_val_accuracy = 0.0  # 记录最佳验证准确率
    
    for epoch in range(config.EPOCHS):
        print(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        # 训练一个epoch
        train_one_epoch(model, trainloader, criterion, optimizer, config.DEVICE)
        
        # 在验证集上评估
        val_accuracy = evaluate(model, valloader, config.DEVICE)
        print(f"Epoch {epoch+1} 在验证集上的准确率: {val_accuracy:.2f} %")
        
        # 记录训练结果
        logger.log_epoch(epoch, train_loss, val_accuracy)
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> 新的最佳模型已保存！验证准确率: {val_accuracy:.2f}%")
    
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%")

    # 5. 最终在测试集上评估
    print("正在加载最佳模型进行最终测试...")
    model.load_state_dict(torch.load("best_model.pth"))
    test_accuracy = evaluate(model, testloader, config.DEVICE)
    print(f"最终测试准确率: {test_accuracy:.2f}%")

    # 6. 保存最终模型
    model_save_path = "cifar10_cnn_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到 {model_save_path}")

    # 7. 绘制训练曲线保存记录
    logger.plot_training_curves()
    logger.save_history()
    logger.print_summary()

if __name__ == '__main__':
    main()