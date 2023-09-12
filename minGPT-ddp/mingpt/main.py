import os
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from trainer import Trainer, TrainerConfig
from char_dataset import CharDataset, DataConfig
from omegaconf import DictConfig
import hydra


def ddp_setup():
    os.environ["MASTER_ADDR"] = "localhost" # 由于这里是单机实验所以直接写 localhost
    os.environ["MASTER_PORT"] = "12355"     # 任意空闲端口
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_train_objs(gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)
    
    return model, optimizer, train_set, test_set
 
@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):
    # 初始化进程池
    ddp_setup()

    # 从 yaml 文件读取超参数
    gpt_cfg = GPTConfig(**cfg['gpt_config'])
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg = DataConfig(**cfg['data_config'])
    trainer_cfg = TrainerConfig(**cfg['trainer_config'])

    # 创建训练对象
    model, optimizer, train_data, test_data = get_train_objs(gpt_cfg, opt_cfg, data_cfg)
    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
    
    # 开始训练
    trainer.train()

    # 训练完成后，销毁进程池
    destroy_process_group()


if __name__ == "__main__":
    main()

'''
运行命令: 
    CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=gpu main.py
'''