# 使用 DistributedDataParallel 进行单机多卡训练
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# 对 python 多进程的一个 pytorch 包装
import torch.multiprocessing as mp

# 这个 sampler 可以把采样的数据分散到各个 CPU 上                                      
from torch.utils.data.distributed import DistributedSampler     

# 实现分布式数据并行的核心类        
from torch.nn.parallel import DistributedDataParallel as DDP         

# DDP 在每个 GPU 上运行一个进程，其中都有一套完全相同的 Trainer 副本（包括model和optimizer）
# 各个进程之间通过一个进程池进行通信，这两个方法来初始化和销毁进程池
from torch.distributed import init_process_group, destroy_process_group 


def ddp_setup(rank, world_size):
    """
    setup the distribution process group

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # MASTER Node（运行 rank0 进程，多机多卡时的主机）用来协调各个 Node 的所有进程之间的通信
    os.environ["MASTER_ADDR"] = "localhost" # 由于这里是单机实验所以直接写 localhost
    os.environ["MASTER_PORT"] = "12355"     # 任意空闲端口
    init_process_group(
        backend="nccl",                     # Nvidia CUDA CPU 用这个 "nccl"
        rank=rank,                          
        world_size=world_size
    )
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every                    # 指定保存 ckpt 的周期
        self.model = DDP(model, device_ids=[gpu_id])    # model 要用 DDP 包装一下

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)        # 在各个 epoch 入口调用 DistributedSampler 的 set_epoch 方法是很重要的，这样才能打乱每个 epoch 的样本顺序
        for source, targets in self.train_data: 
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()            # 由于多了一层 DDP 包装，通过 .module 获取原始参数 
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # 各个 GPU 上都在跑一样的训练进程，这里指定 rank0 进程保存 ckpt 以免重复保存
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,                      # 设置了新的 sampler，参数 shuffle 要设置为 False 
        sampler=DistributedSampler(dataset) # 这个 sampler 自动将数据分块后送个各个 GPU，它能避免数据重叠
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    # 初始化进程池
    ddp_setup(rank, world_size)

    # 进行训练
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
   
    # 销毁进程池
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total-epochs', type=int, default=50, help='Total epochs to train the model')
    parser.add_argument('--save-every', type=int, default=10, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    
    # 利用 mp.spawn，在整个 distribution group 的 nprocs 个 GPU 上生成进程来执行 fn 方法，并能设置要传入 fn 的参数 args
    # 注意不需要 fn 的 rank 参数，它由 mp.spawn 自动分配
    mp.spawn(
        fn=main, 
        args=(world_size, args.save_every, args.total_epochs, args.batch_size), 
        nprocs=world_size
    )