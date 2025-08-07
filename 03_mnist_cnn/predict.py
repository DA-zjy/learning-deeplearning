# predict.py

import torch
from torchvision import transforms
from PIL import Image # Pillow 库，用于图像处理。如果未安装，请 pip install Pillow

# 从我们自己的模块中导入
import config
from model import SimpleCNN

def predict(model_path, image_path):
    """加载模型并对单张图片进行预测"""
    
    # 1. 设置设备
    device = torch.device(config.DEVICE)
    
    # 2. 实例化模型结构
    model = SimpleCNN().to(device)
    
    # 3. 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 4. 设置为评估模式
    model.eval()
    
    # 5. 图像预处理
    # 注意：这里的预处理必须和训练时完全一致！
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # 确保是单通道灰度图
        transforms.Resize((28, 28)),                 # 确保尺寸是 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 打开图片并进行预处理
    image = Image.open(image_path).convert("L") # .convert("L") 确保是灰度模式
    image_tensor = transform(image).unsqueeze(0).to(device) # unsqueeze(0) 增加一个 batch 维度，即1（batch）*1（通道）*28*28
    
    # 6. 进行预测
    with torch.no_grad():
        outputs = model(image_tensor)
        # 使用 softmax 将输出转换为概率分布
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # 找到概率最高的类别
        predicted_class = torch.argmax(probabilities, dim=1)
        top_prob = torch.max(probabilities)

    return predicted_class.item(), top_prob.item()

if __name__ == '__main__':
    # 找到你保存的最好的模型文件，比如最后一个epoch的
    MODEL_TO_LOAD = f"{config.CHECKPOINT_PATH}/mnist_cnn_epoch_10.pth"
    
    # 准备一张你要测试的图片
    # 你可以自己用画图工具画一个数字，保存成 my_digit.png
    # 或者从网上下载一张数字图片
    IMAGE_TO_TEST = "test_images/image.png" # <--- !!! 把这里换成你自己的图片路径 !!!
    
    try:
        predicted_digit, confidence = predict(MODEL_TO_LOAD, IMAGE_TO_TEST)
        print(f"模型预测这张图片是数字: {predicted_digit}")
        print(f"预测的置信度: {confidence*100:.2f}%")
    except FileNotFoundError:
        print(f"错误: 找不到图片文件 '{IMAGE_TO_TEST}'。")
        print("请用你自己的图片路径替换 'path/to/your/image.png'。")