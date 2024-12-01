import os
import pandas as pd
import numpy as np
from PIL import Image

def make_dir(train_path, valid_path):
    """
    创建训练和验证数据的目录结构，每个类别一个文件夹。
    """
    for i in range(0, 7):
        p1 = os.path.join(train_path, str(i))
        p2 = os.path.join(valid_path, str(i))
        if not os.path.exists(p1):
            os.makedirs(p1)
        if not os.path.exists(p2):
            os.makedirs(p2)
    print(f"Directories created under {train_path} and {valid_path}.")

def save_images(data_path, train_path, valid_path):
    """
    读取CSV文件，将图像数据保存为图片文件，分别放入训练和验证目录。
    """
    df = pd.read_csv(data_path)
    t_i = [1 for _ in range(0, 7)]  # 训练集文件计数器
    v_i = [1 for _ in range(0, 7)]  # 验证集文件计数器

    for index in range(len(df)):
        emotion = int(df.iloc[index, 0])  # 第一列：emotion (类别)
        image = df.iloc[index, 1]        # 第二列：pixels (像素数据)
        usage = df.iloc[index, 2]        # 第三列：Usage (用途)
        
        # 将像素数据转换为图像
        data_array = list(map(float, image.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)
        im = Image.fromarray(image).convert('L')  # 转换为8位灰度图像
        
        if usage == 'Training':
            t_p = os.path.join(train_path, str(emotion), f'{t_i[emotion]}.jpg')
            im.save(t_p)
            t_i[emotion] += 1
        else:
            v_p = os.path.join(valid_path, str(emotion), f'{v_i[emotion]}.jpg')
            im.save(v_p)
            v_i[emotion] += 1
    print(f"Images saved to {train_path} and {valid_path}.")

if __name__ == "__main__":
    # 设置路径
    train_path = 'data/train'
    valid_path = 'data/valid'
    data_path = 'data/no_duplicates_file.csv'

    # 执行目录创建和图片保存
    make_dir(train_path, valid_path)
    save_images(data_path, train_path, valid_path)
    print("successful processed")
