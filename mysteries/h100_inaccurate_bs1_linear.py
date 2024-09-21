import torch
torch.manual_seed(0)

lin = torch.nn.Linear(32, 2).cuda()
x = torch.randn(4, 32).cuda() * 100.0

i = 1
print(lin(x)[:i])
print(lin(x[:i]))

"""
on H100 will produce diff output for i=1 and i>1
probably something to do with bigger matmuls being more accurate?
status: unsolved
"""