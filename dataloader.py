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
    elif dataset_name == 'Flowers102':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_set = datasets.Flowers102(
            DATA_LOC, split="train", download=True, transform=transform)
        test_set = datasets.Flowers102(
            DATA_LOC, split="test", download=True, transform=transform)
        in_channels = 3
        num_classes = 102
    else:
        raise ValueError('Unknown dataset')
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, in_channels, num_classes
