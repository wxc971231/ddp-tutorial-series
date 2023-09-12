# minGPT-DDP

Code accompanying the tutorial at https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html for training a GPT-like model with Distributed Data Parallel (DDP) in PyTorch.

Files marked with an asterisk (*) are adapted from the minGPT repo (https://github.com/karpathy/minGPT).

- [trainer.py](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/trainer.py) includes the Trainer class that runs the distributed training iterations on the model with the provided dataset.
- [model.py *](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/model.py) defines the model architecture.
- [char_dataset.py *](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/char_dataset.py) contains the `Dataset`class for a character-level dataset.
- [gpt2_train_cfg.yaml](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/gpt2_train_cfg.yaml) contains the configurations for data, model, optimizer and training run.
- [main.py](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/main.py) is the entry point to the trainig job. It sets up the DDP process group, reads all the configurations and runs the training job.
- [slurm/](https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/slurm) contains files for setting up an AWS cluster and the slurm script to run multinode training.