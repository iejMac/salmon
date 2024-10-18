import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

class CIFAR10Dataset:
    def __init__(self, batch_size, train=True, device="cpu", root="./data"):
        self.batch_size = batch_size
        self.device = device

        self.transform = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.X = (torch.tensor(dataset.data).float() / 255).permute(0, 3, 1, 2)
        self.Y = torch.tensor(dataset.targets)

    def __iter__(self):
        while True:
            idx = torch.randint(0, self.X.shape[0], (self.batch_size,))
            X = self.transform(self.X[idx])
            X = X.view(X.shape[0], -1)  # flatten for MLP
            y = self.Y[idx]
            yield X.to(self.device), y.to(self.device)


if __name__ == "__main__":
    ds = CIFAR10Dataset(4, True)
    for X, y in ds:
        print(X.shape, y.shape)
        break