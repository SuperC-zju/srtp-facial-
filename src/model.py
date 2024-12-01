import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG 块的构建函数
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(out_channels))  # 批归一化
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 最大池化
    return nn.Sequential(*layers)

# 通用 VGG 模型构建函数
def vgg(conv_arch, fc_features, fc_hidden_units, num_classes=7):
    conv_blks = []
    in_channels = 1  # 假设输入为灰度图像 (1 通道)
    
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, num_classes)
    )

# 定义特定的 VGG 模型类
class VGG13(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG13, self).__init__()
        conv_arch = ((2, 64), (2, 128), (2, 256), (2, 512))
        fc_features = 512 * 3 * 3  # 对应的全连接输入特征
        fc_hidden_units = 1024
        self.model = vgg(conv_arch, fc_features, fc_hidden_units, num_classes)

    def forward(self, x):
        return self.model(x)

class VGGCustom(nn.Module):
    def __init__(self, num_classes=7):
        super(VGGCustom, self).__init__()
        conv_arch = ((1, 32), (1, 64), (2, 128))
        fc_features = 128 * 6 * 6  # 对应的全连接输入特征
        fc_hidden_units = 1024
        self.model = vgg(conv_arch, fc_features, fc_hidden_units, num_classes)

    def forward(self, x):
        return self.model(x)
    

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 每个路径的定义
        self.p1_1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.p2_1 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.BatchNorm2d(c2[0]),
            nn.ReLU()
        )
        self.p2_2 = nn.Sequential(
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c2[1]),
            nn.ReLU()
        )
        self.p3_1 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU()
        )
        self.p3_2 = nn.Sequential(
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU()
        )
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Sequential(
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            nn.ReLU()
        )

    def forward(self, x):
        # 拼接不同路径的输出
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=7):
        super(GoogleNet, self).__init__()

        # 主干网络部分
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (18, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 最后的分类层
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 顺序连接各个部分
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.fc(x)
        return x


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)  # 第一个卷积层
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)  # 第二个卷积层
        
        if use_1x1conv:  # 如果需要调整输入输出通道，则使用1x1卷积
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm2d(num_channels)  # 第一个批归一化层
        self.bn2 = nn.BatchNorm2d(num_channels)  # 第二个批归一化层
        self.relu = nn.ReLU(inplace=True)  # 激活函数（ReLU）
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))  # 第一个卷积、批归一化和激活
        Y = self.bn2(self.conv2(Y))  # 第二个卷积和批归一化
        if self.conv3:  # 如果需要1x1卷积调整输入输出通道
            X = self.conv3(X)
        Y += X  # 残差连接
        return F.relu(Y)  # 最后通过ReLU激活


class ResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet, self).__init__()

        # 第一个残差模块
        self.b1 = nn.Sequential(
            Residual(1, 64, use_1x1conv=False, strides=1),
            Residual(64, 64, use_1x1conv=False, strides=1)
        )

        # 第二个残差模块
        self.b2 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, strides=2),
            Residual(128, 128, use_1x1conv=False, strides=1)
        )

        # 第三个残差模块
        self.b3 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, strides=2),
            Residual(256, 256, use_1x1conv=False, strides=1)
        )

        # 第四个残差模块
        self.b4 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True, strides=2),
            Residual(512, 512, use_1x1conv=False, strides=1)
        )

        # 分类全连接部分
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 前向传播通过每个模块
        x = self.b1(x)  # 第一个模块
        x = self.b2(x)  # 第二个模块
        x = self.b3(x)  # 第三个模块
        x = self.b4(x)  # 第四个模块

        # 分类部分
        x = self.flatten(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
