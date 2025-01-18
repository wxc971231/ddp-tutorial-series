[<img src="https://api.gitsponsors.com/api/badge/img?id=686616963" height="20">](https://api.gitsponsors.com/api/badge/link?p=KoJzzoI5V0U5cxCo3lEj9srdIMTw7IoiFPToGomZRFp9HNMVxC2tGRy4n5Chm6M03jA9RjbezjCIKyoQVFxp7yN3+IexpNGKeaLHWoqwrp/6C6BjFgQf7A9QnfnJcs9D)
# ddp-tutorial-series
Follow the [pytorch official tutorial](https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=youtube&utm_medium=organic_social&utm_campaign=tutorial) to learn how to use `nn.parallel.DistributedDataParallel` to speed up training

# distributed-pytorch

Code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Each code file extends upon the previous one. The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.

## Files

- [single_gpu.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py): Non-distributed training script
- [multigpu.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py): DDP on a single node
- [multigpu_torchrun.py](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py): DDP on a single node using Torchrun
- minGPT-ddp:  training a GPT-like model (from the minGPT repo [https://github.com/karpathy/minGPT](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGo2ZjQtMzFlQ2pJQmV6VV9yenFpdmlXVGItd3xBQ3Jtc0tueWdxVFZsYTNPRXFTSW5xejhUajZ1OVYydjNraENoZzNka05ZLWMtZXJkM1VjaFd5cENUMld0TEc5N3VkRFV2bzM2aWdvWVRjTU01TmFfZE9mdXVBTFczWDJZMnU2TjA4Z0tCd25LX2sxOFJLMWtsMA&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2FminGPT&v=XFsFDGKZHh4)) with DDP. 



## 我的笔记

- [Pytorch 多卡并行（1）—— 原理简介和 DDP 并行实践](https://blog.csdn.net/wxc971231/article/details/132816104)

- [Pytorch 多卡并行（2）—— 使用 torchrun 进行容错处理](https://blog.csdn.net/wxc971231/article/details/132827787)

- [Pytorch 多卡并行（3）—— 使用 DDP 加速 minGPT 训练](https://blog.csdn.net/wxc971231/article/details/132829661)

