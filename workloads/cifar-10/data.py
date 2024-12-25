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
    def __init__(self, dataset_size, batch_size, width, device="cpu", resample=True, signal_strength=0.0):
        self.batch_size = batch_size
        self.device = device
        self.width = width
        self.dataset_size = dataset_size
        self.resample = resample

        # Generate synthetic data and labels from N(0, 1)
        self.X = torch.randn(self.dataset_size, width).to(device)

        # Generate weights for a linear relationship and add Gaussian noise to Y
        true_weights = torch.randn(width, 1).to(device)
        noise = torch.randn(self.dataset_size, 1).to(device)
        self.Y = self.X @ true_weights * signal_strength + noise * (1.0 - signal_strength)


    def __iter__(self):
        while True:
            if self.resample:
                idx = torch.randint(0, self.X.shape[0], (self.batch_size,))
            else:
                idx = torch.randperm(self.X.shape[0])[:self.batch_size]
                if self.batch_size == self.dataset_size:
                    idx = torch.arange(0, self.X.shape[0])
            X = self.X[idx]
            y = self.Y[idx]
            yield X, y


if __name__ == "__main__":
    ds = CIFAR10Dataset(4, True)
    for X, y in ds:
        print(X.shape, y.shape)
        break