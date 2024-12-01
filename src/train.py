# src/train.py
import numpy as np
import pandas as pd
from PIL import Image
import os
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from model import VGG13
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformer import TrainTransforms,ValidTransforms


import os
from torch.utils.tensorboard import SummaryWriter

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, 
          transform_name="default_transform", model_name="default_model"):
    """
    训练模型，并根据 transform 和模型名字保存最优模型和 TensorBoard 日志。
    
    参数：
    - model: 待训练的模型
    - train_loader: 训练数据加载器
    - valid_loader: 验证数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数
    - transform_name: 数据增强（transform）名字，用于标记日志和保存路径
    - model_name: 模型名字，用于标记日志和保存路径
    """
    # 创建与 transform 和模型名字相关的日志和模型存储路径
    log_dir = f'runs/{transform_name}_{model_name}'
    best_model_path = f'models/{transform_name}_{model_name}_best.pth'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 清除旧的梯度信息

            # 向前传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()

        train_acc = 100 * train_correct / train_total

        # 记录训练阶段的损失和准确率到 TensorBoard
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

        # 验证阶段
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 计算验证准确率
                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                valid_loss += loss.item()

        valid_acc = 100 * valid_correct / valid_total

        # 记录验证阶段的损失和准确率到 TensorBoard
        writer.add_scalar('Loss/valid', valid_loss / len(valid_loader), epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {valid_loss/len(valid_loader):.4f}, Valid Accuracy: {valid_acc:.2f}%")

        # 保存最佳模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_valid_acc:.2f}% at {best_model_path}")

    # 训练完成后关闭 TensorBoard writer
    writer.close()




if __name__ == '__main__':
    # 更新数据目录路径
    train_data_dir = 'data/train_data'
    test_data_dir = 'data/test_data'  # 新增测试数据路径

    BATCH_SIZE = 48  


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    data_train = torchvision.datasets.ImageFolder(root='data/train', transform=TrainTransforms())
    data_valid = torchvision.datasets.ImageFolder(root='data/valid', transform=ValidTransforms())

    # 创建数据加载器
    train_loader = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=data_valid, batch_size=BATCH_SIZE, shuffle=False)

        # 示例：打印训练和验证集的大小
    print(f"训练集样本数: {len(data_train)}")
    print(f"验证集样本数: {len(data_valid)}")

        # 假设你的模型是 MultiChannelCNN
    # model =ResNet1D(BasicBlock1D, [2, 2, 2, 2])
    model=VGG13()
    # 创建一个存储每层输入输出形状的函数
    def print_layer_shapes(module, input, output):
        print(f"{module.__class__.__name__}:")
        print(f"  输入形状: {tuple(input[0].shape)}")  # 输入是一个 tuple
        print(f"  输出形状: {tuple(output.shape)}")
        print("-" * 30)

    # 注册钩子函数
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear, nn.MaxPool1d, nn.ReLU, nn.Sigmoid)):
            hooks.append(layer.register_forward_hook(print_layer_shapes))

    # 创建一个模拟输入 (batch_size=4, channels=4, sequence_length=2560)
    inputs = torch.randn(1, 1, 48,48)  # 随机生成输入数据
    outputs = model(inputs)

    # 移除钩子
    for hook in hooks:
        hook.remove()


    # 应用初始化
    # model.apply(init_weights)
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # 使用SGD优化器
    num_epochs=50
    
    train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, 
          transform_name="VGG13", model_name="VGG13")
   