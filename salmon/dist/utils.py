import torch.distributed as dist

def print_rank_n(x, rank=0):
    if dist.get_rank() == rank:
        print(x)