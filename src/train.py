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
from .model import MLP1,ResNet18
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .transformer import TrainTransforms,ValidTransforms


import os
from torch.utils.tensorboard import SummaryWriter

def load_data(train_data_dir, valid_data_dir, batch_size, train_transform, valid_transform):
    """
    加载训练和验证数据集，并创建数据加载器

    参数:
    train_data_dir (str): 训练数据集目录路径
    valid_data_dir (str): 验证数据集目录路径
    batch_size (int): 批量大小
    train_transform (callable): 训练数据的变换
    valid_transform (callable): 验证数据的变换

    返回:
    train_loader (DataLoader): 训练数据加载器
    valid_loader (DataLoader): 验证数据加载器
    """
    # 加载数据集
    data_train = torchvision.datasets.ImageFolder(root=train_data_dir, transform=train_transform)
    data_valid = torchvision.datasets.ImageFolder(root=valid_data_dir, transform=valid_transform)

    # 创建数据加载器
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=data_valid, batch_size=batch_size, shuffle=False)

    # 打印训练和验证集的大小
    print(f"训练集样本数: {len(data_train)}")
    print(f"验证集样本数: {len(data_valid)}")

    return train_loader, valid_loader

def print_model_layer_shapes(model, output_func=print):
    """
    打印模型每层的输入输出形状

    参数:
    model (nn.Module): 要检查的模型
    output_func (callable): 输出函数，默认为 print
    """
    def print_layer_shapes(module, input, output):
        output_func(f"{module.__class__.__name__}:")
        output_func(f"  输入形状: {tuple(input[0].shape)}")
        output_func(f"  输出形状: {tuple(output.shape)}")
        output_func("-" * 30)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Dropout)):
            hooks.append(layer.register_forward_hook(print_layer_shapes))

    inputs = torch.randn(1, 1, 48, 48)
    outputs = model(inputs)

    for hook in hooks:
        hook.remove()

def train(model, train_loader, valid_loader, criterion, optimizer, epochs, transform_name="default_transform", model_name="model", output_func=print):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        output_func(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                valid_loss += loss.item()

        valid_acc = 100 * valid_correct / valid_total
        output_func(f"Epoch [{epoch+1}/{epochs}], Valid Loss: {valid_loss/len(valid_loader):.4f}, Valid Accuracy: {valid_acc:.2f}%")

if __name__ == '__main__':
 # 更新数据目录路径
    train_data_dir = 'data/train'
    valid_data_dir = 'data/valid'  # 新增验证数据路径

    BATCH_SIZE = 48  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_loader, valid_loader = load_data(train_data_dir, valid_data_dir, BATCH_SIZE, TrainTransforms(), ValidTransforms())

        # 假设你的模型是 MultiChannelCNN
    # model =ResNet1D(BasicBlock1D, [2, 2, 2, 2])
    model=MLP1()
    print_model_layer_shapes(model)

    # 应用初始化
    # model.apply(init_weights)
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # 使用SGD优化器
    num_epochs=50
    
    train(model, train_loader, valid_loader, criterion, optimizer,num_epochs, transform_name="default_transform", model_name="model")
   