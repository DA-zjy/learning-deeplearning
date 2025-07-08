# utils/training.py

import torch

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch 的函数"""
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0
    total_batches = 0
    
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_batches += 1
        
        if i % 500 == 499: # 每 500 个 batch 打印一次 loss
            print(f'    - Batch {i + 1}, Loss: {running_loss / 500:.4f}')
            running_loss_for_print = running_loss / 500
            running_loss = 0.0  # 重置用于打印的损失
    
    # 返回整个epoch的平均损失
    return running_loss / total_batches if total_batches > 0 else 0.0

def evaluate(model, dataloader, device):
    """在测试集上评估模型的函数"""
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad(): # 在评估阶段，不需要计算梯度
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy