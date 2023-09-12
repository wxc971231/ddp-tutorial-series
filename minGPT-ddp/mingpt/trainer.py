"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Optional, Any, Dict
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import fsspec
import torch.distributed as dist

@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int

class Trainer:
    def __init__(self, trainer_config: TrainerConfig, model, optimizer, train_dataset, test_dataset=None):
        self.config = trainer_config
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"]) # 在所有node的所有进程中当前GPU进程的rank
        self.global_rank = int(os.environ["RANK"])      # 在当前node中当前GPU进程的rank
        
        # data stuff
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None
        
        # initialize train states
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer        
        self.save_every = self.config.save_every

        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()

        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])

        # torch.cuda.amp.GradScaler 是一个用于自动混合精度训练的 PyTorch 工具，它可以帮助加速模型训练并减少显存使用量
        # 具体来说，GradScaler 可以将梯度缩放到较小的范围，以避免数值下溢或溢出的问题，同时保持足够的精度以避免模型的性能下降
        if self.config.use_amp: 
            self.scaler = torch.cuda.amp.GradScaler()

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset)                 # 这个 sampler 自动将数据分块后送个各个 GPU，它能避免数据重叠
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)   # fsspec 为各种后端存储系统提供统一的 Python 接口，可以用相同的语法打开本地、AWS S3 和 GCS 等各种云存储平台的文件
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return 
    
        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch
        )
        # save snapshot
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.config.snapshot_path)
        print(f"Snapshot saved at epoch {epoch}")

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(self.config.use_amp)):
            _, loss = self.model(source, targets)
        
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_amp: 
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()
        
        #return loss.item()
        return loss

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        for iter, (source, targets) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets, train)
            if iter % 100 == 0:
                #print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss.item():.5f}")
                if train:
                    print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss.item():.5f}")
                else:
                    eval_loss_list = [torch.zeros_like(batch_loss) for _ in range(int(os.environ['WORLD_SIZE']))]
                    dist.gather(
                        batch_loss,
                        eval_loss_list if self.local_rank == 0 else None, 
                        dst=0
                    )
                    if self.local_rank == 0:
                        for i, loss in enumerate(eval_loss_list):
                            print(f"[GPU{i}] Epoch {epoch} | Iter {iter} | {step_type} Loss {loss.item():.5f}")

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1
            
            # train for one epoch
            self._run_epoch(epoch, self.train_loader, train=True)

            # 各个 GPU 上都在跑一样的训练进程，这里指定 rank0 进程保存 snapshot 以免重复保存
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)
