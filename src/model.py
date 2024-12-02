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



# 定义基本残差块（用于 ResNet18 和 ResNet34）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 下采样
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 定义瓶颈残差块（用于 ResNet50）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 卷积
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # 下采样
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 定义 ResNet 主体
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入通道数为 1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        # 第一个残差块可能需要下采样
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        # 后续的残差块
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 构建 ResNet18 模型
def ResNet18(num_classes=7):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# 构建 ResNet34 模型
def ResNet34(num_classes=7):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

# 构建 ResNet50 模型
def ResNet50(num_classes=7):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

import torch
import torch.nn as nn

# MLP 模型 1
class MLP1(nn.Module):
    def __init__(self, num_classes=7,batch_size=48):
        super(MLP1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(48 * 48, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# MLP 模型 2
class MLP2(nn.Module):
    def __init__(self, num_classes=7):
        super(MLP2, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(48 * 48, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# MLP 模型 3
class MLP3(nn.Module):
    def __init__(self, num_classes=7):
        super(MLP3, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(48 * 48, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x