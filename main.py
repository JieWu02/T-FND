import argparse
import os
import torch
import torch.multiprocessing as mp

from evaluation import test_main
from training import torch_main
from data.weibo.config import WeiboConfig
from utils import make_directory
from utils.ddp_utils import setup_ddp, cleanup_ddp

import warnings
warnings.filterwarnings("ignore")


def main_worker(rank, world_size, args):
    """DDP worker进程"""
    if world_size > 1:
        setup_ddp(rank, world_size, args.port)
    
    # 设置设备
    device = torch.device(f'cuda:{rank}')
    
    # 创建配置
    if args.data == 'weibo':
        config = WeiboConfig()
    else:
        raise Exception('Enter a valid dataset name', args.data)

    # 设置设备
    config.device = device
    
    if args.batch:
        config.batch_size = args.batch
    if args.epoch:
        config.epochs = args.epoch
    if args.first_stage_epochs:
        config.first_stage_epochs = args.first_stage_epochs
    if args.second_stage_epochs:
        config.second_stage_epochs = args.second_stage_epochs

    config.output_path += 'logs/' + args.data + '_' + str(args.extra)

    # 只在主进程中创建目录
    if rank == 0:
        make_directory(config.output_path)
    
    # 同步所有进程
    if world_size > 1:
        torch.distributed.barrier()

    if args.just_test is not None:
        test_main(config, args.just_test)
    else:
        torch_main(config, rank, world_size)
    
    # 清理DDP
    if world_size > 1:
        cleanup_ddp()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--just_test', type=int, required=False)
    parser.add_argument('--batch', type=int, required=False)
    parser.add_argument('--epoch', type=int, required=False)
    parser.add_argument('--extra', type=str, required=False)
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs separated by comma')
    parser.add_argument('--port', type=int, default=None, help='DDP master port')
    parser.add_argument('--first_stage_epochs', type=int, default=None, help='Number of epochs for first stage (full model training)')
    parser.add_argument('--second_stage_epochs', type=int, default=None, help='Max number of epochs for second stage (teacher fine-tuning)')

    args = parser.parse_args()
    
    # 解析GPU列表
    gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(',')]
    world_size = len(gpu_ids)
    
    if world_size == 1:
        # 单卡训练
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        main_worker(0, 1, args)
    else:
        # 多卡训练
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
