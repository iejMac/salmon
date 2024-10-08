import torch
import torch.distributed as dist

from salmon.dist.utils import print_rank_n


class Reducer:
    """
    Wraps a modules parameters and clusters neighboring gradient all_reduce operations
    to improve communication efficiency.

    It does this by creating gradient buffers for the flattened, concatenated gradients of each "bucket"
    and calling asynchronous all_reduce on those vs. the actual param.grad
    """
    def __init__(self, named_parameters, world_size, bucket_cap_mb=25.0):
        self.bucket_cap_mb = bucket_cap_mb
        self.world_size = world_size

        self._init_params(named_parameters)
        self._init_buckets(bucket_cap_mb)

    def _init_params(self, named_parameters):
        self.params = {}  # access to param and grad
        self.p_ready = {}  # if accumulated grad or not yet
        self.p_bucket = {}  # which bucket p belongs to
        self.p_gb_slice = {}  # which indices of gradient buffer hold p.grad
        for k, p in named_parameters:
            self.params[k] = p
            self.p_ready[k] = False
            post_grad_hook = lambda g, k=k: self._post_grad_hook(k)
            p.register_post_accumulate_grad_hook(post_grad_hook)

    def _init_buckets(self, bucket_cap_mb):
        self.b_params = []  # which params are in the bucket
        self.b_work = []  # async work object for reducing buffer
        self.grad_buffers = []  # buffer tensors

        # assign params to buckets
        self.b_params = []
        b_i_mb, cur_bucket = 0.0, []
        for k in list(self.p_ready.keys())[::-1]:  # reverse order since thats usually order of grad computation
            b_i_mb += self.params[k].data.numel() * self.params[k].data.element_size() / (1024 ** 2)  # parameter size in MB
            cur_bucket.append(k)
            self.p_bucket[k] = len(self.b_params)  # param k belongs to bucket i
            if b_i_mb > bucket_cap_mb:  # start new bucket
                self.b_params.append(cur_bucket)
                b_i_mb, cur_bucket = 0.0, []
                continue
        if len(cur_bucket) > 0:  # remainder in new bucket
            self.b_params.append(cur_bucket)

        # create buffers
        for b_ps in self.b_params:
            buffer_size = 0
            for k in b_ps:
                n_weights = self.params[k].numel()
                start, end = buffer_size, buffer_size + n_weights
                self.p_gb_slice[k] = slice(start, end)  # grad_buffer[buf_size:buf_size+n_w] = p.grad
                buffer_size += n_weights
                device, dtype = self.params[k].device, self.params[k].dtype  # ASSUMPTION: all have the same device and dtype
            self.grad_buffers.append(torch.zeros((buffer_size,), dtype=dtype, device=device))
        self.b_work = []

    def _post_grad_hook(self, k):
        self.p_ready.update({k: True})  # ready up param

        b_i = self.p_bucket[k]
        p_sl = self.p_gb_slice[k]
        # TODO: maybe we can somehow automatically accumulate into buffer?
        self.grad_buffers[b_i][p_sl].copy_(self.params[k].grad.view(-1))  # copy the flattened grad into buffer

        if all([self.p_ready[k] for k in self.b_params[b_i]]):
            self.grad_buffers[b_i].div_(self.world_size)
            # use this https://github.com/pytorch/pytorch/blob/main/torch/distributed/_functional_collectives.py
            work_b = dist.all_reduce(self.grad_buffers[b_i], async_op=True)
            self.b_work.append(work_b)

        if len(self.b_work) == len(self.b_params):
            for b_i, bw in enumerate(self.b_work):
                bw.wait()
                for k in self.b_params[b_i]:  # copy grad back into param.grad
                    p_sl = self.p_gb_slice[k]  # slice out correct grad
                    g_sh = self.params[k].grad.shape # inflate back to normal shape via .view
                    self.params[k].grad.copy_(self.grad_buffers[b_i][p_sl].view(g_sh))

    def reset_state(self):
        self.b_work = []
        for k in self.p_ready.keys():
            self.p_ready[k] = False
        for gb in self.grad_buffers: # NOTE: not needed since all ops are memcpy, do we need it?
            gb.zero_()