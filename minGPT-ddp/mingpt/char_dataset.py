import torch
from torch.utils.data import Dataset
import fsspec
from dataclasses import dataclass

"""
Adapted from https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
"""

@dataclass
class DataConfig:
    path: str = None
    block_size: int = None      # 输入序列长度    
    train_split: float = None   # 训练集和测试集划分
    truncate: float = 1.0       # 用于训练的数据占全体数据的比例

class CharDataset(Dataset):

    def __init__(self, data_cfg: DataConfig): #data_path: str, block_size):
        # 加载所需比例的数据
        data = fsspec.open(data_cfg.path).open().read().decode('utf-8')
        data = data[ : int(len(data) * data_cfg.truncate)]

        # Set 去重，转 list 后排序得到数据集中的唯一字符列表作为词表
        chars = sorted(list(set(data))) 
        data_size, vocab_size = len(data), len(chars)
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))

        # 得到字符和词表索引之间的双射
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 词表索引
        self.itos = {i: ch for i, ch in enumerate(chars)}   # 词表索引 -> 字符
        
        self.block_size = data_cfg.block_size   
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
