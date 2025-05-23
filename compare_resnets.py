# flake8: noqa: E501

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
import argparse

from custom_models.resnet_sse import ResNet18_sSE
from custom_models.resnet_arc import ResNet18_ArcFace
from custom_models.resnet_sse_arc import ResNet18_sSE_ArcFace
from custom_models.resnet import get_standard_resnet18
from dataloader import get_dataloaders


def set_seed(seed:int = 42) -> None:
    """
    Sets the seed for the random number generators.

    Parameters:
        seed (int, optional): The value to set the seed to. Defaults to 42
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
        model: nn.Module, 
        device: torch.device, 
        train_loader: torch.utils.data.DataLoader, 
        optimizer: optim.Adam,
        criterion: nn.CrossEntropyLoss, 
        use_arcface: bool = False) -> Tuple[float, float]:
    """
    Runs a epoch of training on the model.

    Parameters:
        model (nn.Module): The model that is to be trained.
        device (torch.device): Device to run the models on (CPU or GPU).
        train_loader (torch.utils.data.DataLoader): The PyTorch dataloader.
        optimizer (optim.Adam): The optimizer for the model.
        criterion (nn.CrossEntropyLoss): The loss function of the model.
        use_arcface (bool, optional): A flag for how the model should be ran based on the 
        requirements in ArcFace loss.
        
    Returns:
        float: The training loss.
        float: The training accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if use_arcface:
            output = model(data, target)
        else:
            output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += data.size(0)
    return running_loss / total, correct / total


def test(
        model: nn.Module, 
        device: torch.device, 
        test_loader: torch.utils.data.DataLoader, 
        criterion: nn.CrossEntropyLoss, 
        use_arcface: bool = False) -> Tuple[float, float]:
    """
    Runs the test set on the model.

    Parameters:
        model (nn.Module): The model that is to be tested.
        device (torch.device): Device to run the models on (CPU or GPU).
        test_loader (torch.utils.data.DataLoader): The PyTorch dataloader.
        criterion (nn.CrossEntropyLoss): The loss function of the model.
        use_arcface (bool, optional): A flag for how the model should be ran based on the 
        requirements in ArcFace loss.
        
    Returns:
        float: The test loss.
        float: The test accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if use_arcface:
                output = model(data, target)
            else:
                output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += data.size(0)
    return test_loss / total, correct / total


def run_experiment(dataset_name: str, device: torch.device, epochs: int = 5) -> Dict[str, float]:
    """
    Runs all the models on the provided dataset and saves the data to the log

    Parameters:
        dataset_name (str): Name of the dataset to use for training and testing
        device (torch.device): Device to run the models on (CPU or GPU)
        epochs (int, optional): Number of training epochs. Defaults to 5
        
    Returns:
        Dict[str, float]: Dictionary containing final test accuracies for each model variant
    """
    print(f"\n===== Dataset: {dataset_name} =====")
    train_loader, test_loader, in_channels, num_classes, lr = get_dataloaders(dataset_name) # noqa
    results = {}
    # Model 1: Standard ResNet18
    model1 = get_standard_resnet18(num_classes, in_channels).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr, weight_decay=0.0001)
    criterion1 = nn.CrossEntropyLoss()
    # Model 2: ResNet18 + sSE
    model2 = ResNet18_sSE(num_classes, in_channels).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr, weight_decay=0.0001)
    criterion2 = nn.CrossEntropyLoss()
    # Model 3: ResNet18 + ArcFace
    model3 = ResNet18_ArcFace(num_classes, in_channels).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr, weight_decay=0.0001)
    criterion3 = nn.CrossEntropyLoss()
    # Model 4: ResNet18 + sSE + ArcFace
    model4 = ResNet18_sSE_ArcFace(num_classes, in_channels).to(device)
    optimizer4 = optim.Adam(model4.parameters(), lr, weight_decay=0.0001)
    criterion4 = nn.CrossEntropyLoss()
    models_list = [model1, model2, model3, model4]
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
    criterions = [criterion1, criterion2, criterion3, criterion4]
    names = ["Standard", "sSE", "ArcFace", "sSE+ArcFace"]
    arcface_flags = [False, False, True, True]
    # Prepare log directory
    log_dir = f"logs/logs_{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    all_logs = {}
    for i, (model, optimizer, criterion, name, use_arcface) in enumerate(zip(models_list, optimizers, criterions, names, arcface_flags)):  # noqa
        print(f"\n--- Training {name} ---")
        log = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, device, train_loader, optimizer, criterion, use_arcface)
            test_loss, test_acc = test(
                model, device, test_loader, criterion, use_arcface)
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}") # noqa
            log["train_loss"].append(train_loss)
            log["train_acc"].append(train_acc)
            log["test_loss"].append(test_loss)
            log["test_acc"].append(test_acc)
        # Save log to file
        with open(os.path.join(log_dir, f"{name}_log.json"), "w") as f:
            json.dump(log, f)
        all_logs[name] = log
        results[name] = log["test_acc"][-1]
    print(f"\nResults for {dataset_name}:")
    for name in names:
        print(f"{name}: {results[name]:.4f}")
    # Visualization
    plot_metrics(all_logs, dataset_name, log_dir)
    return results


def plot_metrics(all_logs: dict, dataset_name: str, log_dir: str) -> None:
    """
    Plots the testing and training accuracy and loss for each model and saves plots as PNGs.

    Parameters:
        all_logs (dict): The data of the tests.
        dataset_name (str): The name of the dataset to log.
        log_dir (str): The path to where the images should be saved to.
    """
    metrics = ["train_acc", "test_acc", "train_loss", "test_loss"]
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for name, log in all_logs.items():
            plt.plot(log[metric], label=name)
        plt.xlabel("Epoch")
        if "acc" in metric:
            plt.ylabel("Accuracy")
        else:
            plt.ylabel("Loss")
        plt.title(f"{metric.replace('_', ' ').title()} on {dataset_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"{dataset_name}_{metric}.png"))
        plt.close()


def parse_arguments() -> argparse.Namespace:
    """
    Small function used to parse the arguments inputted when running the file.

    Returns:
        argparse.NameSpace: The arguments as variables inside an object.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate various ResNet models.')
    parser.add_argument('--datasets', nargs='+', type=str, 
                        default=["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"],
                        help='List of datasets to test (e.g., MNIST FashionMNIST CIFAR10)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    """
    The main function of the file, running all tests passed in 
    as arguments, or all tests if the command line argument was omitted.
    """
    args = parse_arguments()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}
    for dataset in args.datasets:
        all_results[dataset] = run_experiment(dataset, device, epochs=args.epochs)
    print("\n===== Summary =====")
    for dataset in args.datasets:
        print(f"\n{dataset}:")
        for name, acc in all_results[dataset].items():
            print(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    main()
