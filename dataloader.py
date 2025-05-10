from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(dataset_name, batch_size=128):
    DATA_LOC = './data'
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(
            DATA_LOC, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(
            DATA_LOC, train=False, download=True, transform=transform)
        in_channels = 1
        num_classes = 10
        learning_rate = 0.001
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_set = datasets.FashionMNIST(
            DATA_LOC, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(
            DATA_LOC, train=False, download=True, transform=transform)
        in_channels = 1
        num_classes = 10
        learning_rate = 0.001
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])
        train_set = datasets.CIFAR10(
            DATA_LOC, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(
            DATA_LOC, train=False, download=True, transform=transform)
        in_channels = 3
        num_classes = 10
        learning_rate = 0.001
    elif dataset_name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])
        train_set = datasets.CIFAR100(
            DATA_LOC, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(
            DATA_LOC, train=False, download=True, transform=transform)
        in_channels = 3
        num_classes = 100
        learning_rate = 0.001
    else:
        raise ValueError('Unknown dataset')
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, in_channels, num_classes, learning_rate
