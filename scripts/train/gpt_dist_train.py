import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.GPT.config.config import Config, set_seed  # 导入配置类
# from src.GPT.model import GPTModel
from src.GPT.model_dist import GPTModel, MDNLoss
from src.utils.util import preprocess_attributes
# 数据集定义
class CelebADataset(Dataset):
    def __init__(self, tensor_dir, attr_file, trunc_num=100):
        self.tensor_dir = tensor_dir
        self.attr_file = attr_file
        self.trunc_num = trunc_num
        self.image_ids = []
        self.attributes = []

        # 加载前20万行属性数据
        with open(self.attr_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 200000:  # 读取到第20万行后停止
                    break
                parts = line.strip().split()
                img_id = parts[0].split('.')[0]  # 获取图片编号
                # attr = preprocess_attributes(parts)
                attr = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                self.image_ids.append(img_id)
                self.attributes.append(attr)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        tensor_path = os.path.join(self.tensor_dir, f'{img_id}.pt')
        sequence = torch.load(tensor_path)  # 加载序列数据

        # 只取前 trunc_num 行
        sequence = sequence[:self.trunc_num, :]

        # 属性数据是作为起始 token，序列是随后的 tokens
        attribute = self.attributes[idx]
        attribute = torch.tensor(attribute).unsqueeze(0)  # 确保属性是 1x40 的张量
        return attribute, sequence

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, config, rank):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for i, (attributes, sequence) in enumerate(dataloader):
        attributes, sequence = attributes.to(device), sequence.to(device)
        sequence = sequence * config.scale  # 缩放序列数据
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():  # 开启半精度计算
            pi, mu, full_cov = model(attributes, sequence) 
            target = sequence.to(pi.dtype)
            loss = criterion(pi, mu, full_cov, target)

        # 使用 GradScaler 进行梯度缩放
        scaler.scale(loss).backward()  # 反向传播时缩放梯度
        scaler.step(optimizer)  # 更新模型参数
        scaler.update()  # 更新 GradScaler 状态

        total_loss += loss.item()

        if rank == 0:  # 只在 rank 0 上打印进度
            tqdm.write(f"Batch [{i + 1}/{num_batches}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


# 分布式训练入口
def main(rank, world_size):
    # 初始化分布式环境
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 设置当前进程使用的 GPU
    device = torch.device("cuda", rank)

    # 加载配置
    config = Config(is_training=True)
    set_seed(config.seed)

    # 数据集和分布式数据加载器
    dataset = CelebADataset(config.tensor_dir, config.attr_file, config.trunc_num)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)

    # 初始化模型
    model = GPTModel(**config.get_gpt_config()).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # 损失函数
    criterion = MDNLoss()

    # 创建 GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # 创建保存目录
    if rank == 0:
        os.makedirs(config.save_dir, exist_ok=True)

    # 用于记录每个 epoch 的损失
    loss_history = []

    # 训练循环
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)  # 设置分布式采样器的 epoch

        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, config, rank)

        if rank == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {avg_loss:.4f}")
            loss_history.append(avg_loss)

            # 保存模型（每10个 epoch 保存一次）
            if (epoch + 1) % 10 == 0:
                model_path = os.path.join(config.save_dir, f"model_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")

            # 保存损失曲线
            plt.plot(range(1, epoch + 2), loss_history, label="Train Loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.savefig(os.path.join(config.save_dir, f'loss_curve_epoch.png'))
            plt.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    world_size = 3
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

