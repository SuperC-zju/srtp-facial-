from PyQt5.QtWidgets import QMessageBox
import subprocess
import sys
import os
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

# 导入模型和打印模型层结构的函数
from src.model import MLP1, MLP2, MLP3, ResNet18, ResNet34, ResNet50, VGG13, VGGCustom, GoogleNet
from src.train import print_model_layer_shapes, train, load_data, TrainTransforms, ValidTransforms

class EventHandlers:
    def __init__(self, ui):
        self.ui = ui
        self.training_process = None  # 用于存储训练进程
        self.train_loader = None
        self.valid_loader = None

    def on_pushButton_clicked(self):
        print("退出按钮被点击")
        self.ui.close()

    def on_pushButton_2_clicked(self):
        print("进行数据预处理按钮被点击")
        self.ui.textEdit.append("进行数据预处理...")
        self.run_preprocess_data()

    def on_pushButton_3_clicked(self):
        print("打印模型结构按钮被点击")
        model_name = self.get_selected_item(self.ui.listWidget)
        self.ui.textEdit.append(f"打印模型结构: {model_name}...")
        model = self.get_model_instance(model_name)
        if model:
            self.print_model_layer_shapes_to_textedit(model)
        else:
            self.ui.textEdit.append("未选择有效的模型")

    def on_pushButton_4_clicked(self):
        print("清空输出按钮被点击")
        self.ui.textEdit.clear()

    def on_pushButton_5_clicked(self):
        print("开始训练按钮被点击")
        batch_size = self.get_batch_size(self.ui.listWidget_2)
        learning_rate = self.get_learning_rate(self.ui.listWidget_3)
        epochs = self.get_epochs(self.ui.listWidget_4)
        optimizer_name = self.get_optimizer(self.ui.listWidget_5)
        model_name = self.get_selected_item(self.ui.listWidget)
        self.ui.textEdit.append(f"开始训练...\n批次大小: {batch_size}\n学习率: {learning_rate}\n训练轮次: {epochs}\n优化器: {optimizer_name}\n模型: {model_name}")
        model = self.get_model_instance(model_name)
        if model:
            self.run_training_process(model, batch_size, learning_rate, epochs, optimizer_name, model_name)
        else:
            self.ui.textEdit.append("未选择有效的模型")

    def on_pushButton_6_clicked(self):
        print("载入数据集按钮被点击")
        self.ui.textEdit.append("载入数据集...")
        self.load_data()

    def on_pushButton_7_clicked(self):
        print("停止训练按钮被点击")
        self.ui.textEdit.append("停止训练...")
        self.stop_training_process()

    def get_selected_item(self, list_widget):
        selected_items = list_widget.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return "未选择"

    def get_model_instance(self, model_name):
        if model_name == "MLP模型：输出：128，Relu激活，输出：7":
            return MLP1()
        elif model_name == "MLP模型：输出：256，Relu激活，输出：64，Relu激活，输出：7":
            return MLP2()
        elif model_name == "MLP模型：输出：512，Relu激活，丢弃层：0.5，输出：,256，Relu激活，丢弃层：0.5，输出：7":
            return MLP3()
        elif model_name == "Resnet模型：Resnet18:  残差层结构：BasicBlock, [2, 2, 2, 2] 输出：7":
            return ResNet18()
        elif model_name == "Resnet模型：Resnet34:  残差层结构：BasicBlock, [3, 4, 6, 3] 输出：7":
            return ResNet34()
        elif model_name == "VGG模型：VGG13，VGG块参数：((2, 64), (2, 128), (2, 256), (2, 512))，全连接输出：1024：输出：7":
            return VGG13()
        elif model_name == "VGG模型：VGG块参数：（(1, 32), (1, 64), (2, 128)），全连接输出：1024，输出：7":
            return VGGCustom()
        elif model_name == "Geoglenet模型：经典inception块构建的模型":
            return GoogleNet()
        else:
            return None

    def get_batch_size(self, list_widget):
        selected_item = self.get_selected_item(list_widget)
        if selected_item == "Batchsize=32":
            return 32
        elif selected_item == "Batchsize=48":
            return 48
        elif selected_item == "Batchsize=64":
            return 64
        else:
            return 32  # 默认值

    def get_learning_rate(self, list_widget):
        selected_item = self.get_selected_item(list_widget)
        if selected_item == "Learningrate=0.0001":
            return 0.0001
        elif selected_item == "Learningrate=0.001":
            return 0.001
        elif selected_item == "Learningrate=0.01":
            return 0.01
        else:
            return 0.001  # 默认值

    def get_epochs(self, list_widget):
        selected_item = self.get_selected_item(list_widget)
        if selected_item == "Epochs=32":
            return 32
        elif selected_item == "Epochs=64":
            return 64
        elif selected_item == "Epochs=150":
            return 150
        else:
            return 32  # 默认值

    def get_optimizer(self, list_widget):
        selected_item = self.get_selected_item(list_widget)
        if selected_item == "SGD":
            return "SGD"
        elif selected_item == "Adam":
            return "Adam"
        else:
            return "SGD"  # 默认值

    def run_preprocess_data(self):
        process = subprocess.Popen(['python', 'scripts/preprocess_data.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        self.ui.textEdit.append(stdout.decode('utf-8'))
        if stderr:
            self.ui.textEdit.append("Error:\n" + stderr.decode('utf-8'))

    def print_model_layer_shapes_to_textedit(self, model):
        def output_func(text):
            self.ui.textEdit.append(text)
        print_model_layer_shapes(model, output_func)

    def load_data(self):
        train_data_dir = 'data/train'
        valid_data_dir = 'data/valid'
        batch_size = self.get_batch_size(self.ui.listWidget_2)
        self.train_loader, self.valid_loader = load_data(train_data_dir, valid_data_dir, batch_size, TrainTransforms(), ValidTransforms())
        self.ui.textEdit.append(f"训练集样本数: {len(self.train_loader.dataset)}")
        self.ui.textEdit.append(f"验证集样本数: {len(self.valid_loader.dataset)}")

    def run_training_process(self, model, batch_size, learning_rate, epochs, optimizer_name, model_name):
        # 将模型移动到设备上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        if optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

        # 训练模型
        def output_func(text):
            self.ui.textEdit.append(text)
        train(model, self.train_loader, self.valid_loader, criterion, optimizer, epochs, transform_name="default_transform", model_name=model_name, output_func=output_func)

    def stop_training_process(self):
        if self.training_process:
            self.training_process.terminate()
            self.training_process = None
            self.ui.textEdit.append("训练进程已终止")