# utils/visualization.py

import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

class TrainingLogger:
    """训练过程记录器"""
    
    def __init__(self, save_dir="./results"):
        self.save_dir = save_dir
        self.history = {
            'train_loss': [],
            'val_accuracy': [],
            'epochs': [],
            'best_val_accuracy': 0.0,
            'best_epoch': 0
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
    def log_epoch(self, epoch, train_loss, val_accuracy):
        """记录单个epoch的结果"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_accuracy'].append(val_accuracy)
        
        # 更新最佳结果
        if val_accuracy > self.history['best_val_accuracy']:
            self.history['best_val_accuracy'] = val_accuracy
            self.history['best_epoch'] = epoch
            
        print(f"  -> 已记录 Epoch {epoch} 结果")
        
    def plot_training_curves(self, save_plot=True):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制训练损失
        ax1.plot(self.history['epochs'], self.history['train_loss'], 'b-', label='训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练损失曲线')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制验证准确率
        ax2.plot(self.history['epochs'], self.history['val_accuracy'], 'r-', label='验证准确率')
        ax2.axhline(y=self.history['best_val_accuracy'], color='g', linestyle='--', 
                   label=f'最佳准确率: {self.history["best_val_accuracy"]:.2f}%')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('验证准确率曲线')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.save_dir, f"training_curves_{timestamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存到: {plot_path}")
            
        plt.show()
        
    def save_history(self):
        """保存训练历史到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(self.save_dir, f"training_history_{timestamp}.json")
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
            
        print(f"训练历史已保存到: {history_path}")
        
    def print_summary(self):
        """打印训练总结"""
        print("\n" + "="*60)
        print("🎯 训练总结")
        print("="*60)
        print(f"总训练轮数: {len(self.history['epochs'])}")
        print(f"最佳验证准确率: {self.history['best_val_accuracy']:.2f}%")
        print(f"最佳模型出现在: Epoch {self.history['best_epoch']}")
        print(f"最终验证准确率: {self.history['val_accuracy'][-1]:.2f}%")
        print("="*60)

def load_training_history(file_path):
    """加载训练历史"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_training_runs(history_files, labels=None):
    """比较多次训练运行的结果"""
    plt.figure(figsize=(12, 8))
    
    for i, file_path in enumerate(history_files):
        history = load_training_history(file_path)
        label = labels[i] if labels else f"Run {i+1}"
        
        plt.plot(history['epochs'], history['val_accuracy'], 
                label=f"{label} (最佳: {history['best_val_accuracy']:.2f}%)")
    
    plt.xlabel('Epoch')
    plt.ylabel('验证准确率 (%)')
    plt.title('多次训练运行比较')
    plt.legend()
    plt.grid(True)
    plt.show()