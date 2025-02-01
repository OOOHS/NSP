# config.py
import random
import numpy as np
import torch
import os

class Config:
    def __init__(self, is_training=True):
        """
        通用配置类，支持训练和推理配置。

        参数:
            is_training: 是否是训练模式，用于切换训练和推理特定的参数。
        """
        # 基本路径配置
        self.tensor_dir = './data/tensor_align_celeba/'  # 序列数据路径
        self.attr_file = './data/list_attr_celeba.txt'  # 属性数据路径
        self.save_dir = './model_checkpoints'           # 模型保存路径
        self.output_dir = './output'                   # 推理结果保存路径
        self.renderer_path = './model_checkpoints/RL/renderer.pkl'          # 渲染器模型路径
        
        # GPT 模型配置
        self.attr_dim = 40  # 属性的维度
        self.action_dim = 13  # 输出动作的维度
        
        self.trunc_num = 200 # 截断的动作数
        
        # self.hidden_dim = 768
        # self.num_heads = 12 
        self.hidden_dim = 1024
        self.num_heads = 16
        # self.num_layers = 8
        self.num_layers = 12
        
        self.gaussian_num = 24 # 高斯分布数量
        
        self.scale = 12.0  # 缩放比例

        # 画布参数
        self.canvas_size = 128  # 画布尺寸

        # 训练配置
        self.is_training = is_training  # 区分训练和推理模式
        if is_training:
            self.use_amp = True 
            self.batch_size = 220
            self.epochs = 2000
            self.lr = 1e-4
            self.weight_decay = 1e-2  # AdamW 的权重衰减
            self.kl_weight = 0.1  # KL 散度损失的权重
            
        # 随机种子
        self.seed = 42
        
    def get_gpt_config(self):
        return {
            'attr_dim': self.attr_dim,
            'hidden_dim': self.hidden_dim,
            'seq_len': self.trunc_num,
            'action_dim': self.action_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'scale': self.scale,
            'gaussian_num': self.gaussian_num
        }

# 随机种子设置函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
