
import torch
from torch.utils.data.distributed import DistributedSampler
from data.data_utils import build_loaders, make_dfs
from .learner import supervised_train
from evaluation.test import test


def torch_main(config, rank=0, world_size=1):
    train_df, test_df, validation_df = make_dfs(config)
    
    # 为DDP创建数据加载器
    train_loader = build_loaders(config, train_df, mode="train", 
                                distributed=(world_size > 1), rank=rank, world_size=world_size)
    validation_loader = build_loaders(config, validation_df, mode="validation", 
                                     distributed=(world_size > 1), rank=rank, world_size=world_size)
    test_loader = build_loaders(config, test_df, mode="test", 
                               distributed=(world_size > 1), rank=rank, world_size=world_size)

    supervised_train(config, train_loader, validation_loader, rank, world_size)
    
    # 只在主进程中进行测试
    if rank == 0:
        test(config, test_loader)





