# ddp-tutorial-series
Follow the pytorch official tutorial to learn how to use `nn.parallel.DistributedDataParallel` to speed up training

# distributed-pytorch

Code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Each code file extends upon the previous one. The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.

## Files

- [single_gpu.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py): Non-distributed training script
- [multigpu.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py): DDP on a single node
- [multigpu_torchrun.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py): DDP on a single node using Torchrun
- [multinode.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py): DDP on multiple nodes using Torchrun (and optionally Slurm)
  - [slurm/setup_pcluster_slurm.md](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/setup_pcluster_slurm.md): instructions to set up an AWS cluster
  - [slurm/config.yaml.template](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/config.yaml.template): configuration to set up an AWS cluster
  - [slurm/sbatch_run.sh](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/sbatch_run.sh): slurm script to launch the training job