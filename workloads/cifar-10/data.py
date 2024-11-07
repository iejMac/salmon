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
        self.X = self.transform(self.X).view(self.X.shape[0], -1).to(device)
        self.Y = torch.tensor(dataset.targets).to(device)

    def __iter__(self):
        while True:
            idx = torch.randint(0, self.X.shape[0], (self.batch_size,))
            X = self.X[idx]
            y = self.Y[idx]
            yield X, y

class SyntheticNormalDataset:
    def __init__(self, batch_size, n=16, device="cpu", dataset_size=None):
        self.batch_size = batch_size
        self.device = device
        self.n = n
        self.dataset_size = dataset_size or (n ** 2 + n)  # Default size as per the paper

        # Generate synthetic data and labels from N(0, 1)
        self.X = torch.randn(self.dataset_size, n).to(device)
        self.Y = torch.randn(self.dataset_size, 1).to(device)

    def __iter__(self):
        while True:
            idx = torch.randint(0, self.X.shape[0], (self.batch_size,))
            X = self.X[idx]
            y = self.Y[idx]
            yield X, y


if __name__ == "__main__":
    ds = CIFAR10Dataset(4, True)
    for X, y in ds:
        print(X.shape, y.shape)
        break