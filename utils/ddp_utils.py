import os
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def find_free_port():
    """查找空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_ddp(rank, world_size, port=None):
    """初始化DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    
    # 设置NCCL环境变量以提高稳定性（使用新的TORCH_NCCL前缀）
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时（秒）
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'  # 启用阻塞等待
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 启用异步错误处理
    os.environ['NCCL_DEBUG'] = 'WARN'  # 设置调试级别
    
    # 如果没有指定端口，使用默认端口
    if port is None:
        port = find_free_port()
    os.environ['MASTER_PORT'] = str(port)
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()


def is_main_process():
    """检查是否是主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """获取当前进程的rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """获取总进程数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor):
    """在所有进程间规约tensor"""
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存checkpoint (只在主进程中保存)"""
    if is_main_process():
        if isinstance(model, DDP):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """加载checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf') 