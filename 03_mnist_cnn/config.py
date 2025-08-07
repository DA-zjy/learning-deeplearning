# config.py

import torch

# --- 训练配置 ---
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 128 #Batch如何选择？
LEARNING_RATE = 1e-3 # 0.001

# --- 数据集配置 ---
DATASET_PATH = "./data"

# --- 模型保存配置 ---
CHECKPOINT_PATH = "./checkpoints"