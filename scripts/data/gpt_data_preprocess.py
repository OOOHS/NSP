"""
生成 GPT 的训练数据，即将原始图像转换为绘画动作序列。
"""

import os
import json
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm  # 用于显示进度条
from torch.nn import DataParallel
from src.DRL.actor import ResNet
from src.Renderer.stroke_gen import *
from src.Renderer.model import *

# 配置和参数初始化

DATA_DIR = './data/img_align_celeba/' # 待转换的数据集存放路径
TENSOR_DIR = './data/tensor_align_celeba/' # 转换后的数据集存放路径
NUM_IMAGES = 200000
NUM_STEPS = 50
STROKES_PER_STEP = 5
STROKE_DIM = 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--actor', default='./actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--width', default=128, type=int, help='image width')
args = parser.parse_args()

# 实例化模型并加载权重
actor = ResNet(9, 18, 65)  # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.actor))
# actor = DataParallel(actor).to(device).eval()  # 使用多GPU
actor = actor.to(device).eval() 

Decoder = FCN()
Decoder.load_state_dict(torch.load(args.renderer))
# Decoder = DataParallel(Decoder).to(device).eval()  # 使用多GPU
Decoder = Decoder.to(device).eval()

# 创建 data 和 tensor 目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TENSOR_DIR, exist_ok=True)

# 设置一些常量
width = args.width
canvas_cnt = 4 * 4  # 假设divide=4
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)
coord = coord.to(device)  # Coordconv

# 将动作序列渲染为图像，不过这份代码仅用于序列生成，不会渲染图像
def decode(x, canvas):
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res

# 获取已处理图像的列表
processed_files = set(f.split('.')[0] for f in os.listdir(TENSOR_DIR) if f.endswith('.pt'))

# 遍历所有图像，添加进度条
with tqdm(total=NUM_IMAGES, desc="处理进度") as pbar:
    for i in range(NUM_IMAGES):
        img_id = '%06d' % (i + 1)
        tensor_path = os.path.join(TENSOR_DIR, f'{img_id}.pt')
        if img_id in processed_files:  # 跳过已处理的图像
            pbar.update(1)  # 更新进度条
            continue

        img_path = os.path.join(DATA_DIR, f'{img_id}.jpg')

        # 1. 按序读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"图像 {img_path} 加载失败，跳过...")
            pbar.update(1)  # 更新进度条
            continue
        img = cv2.resize(img, (width, width))
        img = img.reshape(1, width, width, 3)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.tensor(img).to(device).float() / 255.

        # 2. 生成actions序列并保存
        canvas = torch.zeros([1, 3, width, width]).to(device)
        actions = []
        with torch.no_grad():
            for step in range(NUM_STEPS):
                stepnum = T * step / NUM_STEPS
                action_bundle = actor(torch.cat([canvas, img, stepnum, coord], 1))
                canvas, res = decode(action_bundle, canvas)
                temp = action_bundle.cpu().numpy().reshape(-1, STROKE_DIM)
                actions.append(temp)

        actions = np.array(actions).reshape(-1, STROKE_DIM)  # (250, 13)
        torch.save(torch.from_numpy(actions), tensor_path)
        # print(f"处理完成: {tensor_path}")

        pbar.update(1)  # 更新进度条

print("所有图像处理完成!")
