# Modified ResNet Exploration

Tests sSE and ArcFace loss on ResNet for classical object recognition tasks.

Original Inspiration from [A Face Recognition Algorithm Based on Improved Resnet](https://drpress.org/ojs/index.php/fcis/article/view/1100) by Jing et al. 2022

## Results
### MNIST
#### Accuracy
- Standard ResNet Test Acc: `0.991`
- sSE ResNet Test Acc: `0.9898`
- ArcFace Loss ResNet Test Acc: `0.9813`
- sSE + ArcFace Loss ResNet Test Acc: `0.9696`

#### Loss

<img src="logs/logs_MNIST/MNIST_train_loss.png" width="640" height="400" alt="Loss of model variants for MNIST">

### Fashion-MNIST
#### Accuracy
- Standard ResNet Test Acc: `0.9034`
- sSE ResNet Test Acc: `0.904`
- ArcFace Loss ResNet Test Acc: `0.8315`
- sSE + ArcFace Loss ResNet Test Acc: `0.8422`

#### Loss

<img src="logs/logs_FashionMNIST/FashionMNIST_train_loss.png" width="640" height="400" alt="Loss of model variants for Fashion-MNIST">

### CIFAR10
#### Accuracy
- Standard ResNet Test Acc: `0.7195`
- sSE ResNet Test Acc: `0.7333`
- ArcFace Loss ResNet Test Acc: `0.9952`
- sSE + ArcFace Loss ResNet Test Acc: `0.9846`

#### Loss

<img src="logs/logs_CIFAR10/CIFAR10_train_loss.png" width="640" height="400" alt="Loss of model variants for CIFAR10">

### CIFAR100
#### Accuracy
- Standard ResNet Test Acc: `0.4089`
- sSE ResNet Test Acc: `0.4127`
- ArcFace Loss ResNet Test Acc: `0.9904`
- sSE + ArcFace Loss ResNet Test Acc: `0.986`

#### Loss

<img src="logs/logs_CIFAR100/CIFAR100_train_loss.png" width="640" height="400" alt="Loss of model variants for CIFAR100">

## Development
### Setup Steps
- `python -m venv venv` - Create a venv for the project
- `venv\Scripts\activate` - Activate the venv (Windows)
- `pip install -r requirements.txt` - Install the required packages
- `python compare_resnets.py` - Runs the trainer on all models
  - `--datasets` - Specify which datasets you want to train on (MNIST, FashionMNIST, CIFAR10, CIFAR100) (default all)
  - `--epochs` - How many epochs (default 5)
  - `--seed` - The seed for the random number generator (default 42)

Development was done on Windows, and all training was done on Google Colab T4.

### File Structure
- `custom_models/` - All of the model code for the project
  - `components/` - The custom components to test the ResNet changes
    - `arc_face_head.py` - The head to calculate the loss for ArcFace
    - `sse_block.py` - The Squeeze and Excitation Block for sSE
  - `resnet_arc.py` - ResNet with ArcFace loss
  - `resnet_sse_arc.py` - ResNet with both sSE and ArcFace loss
  - `resnet_sse.py` - ResNet with sSE
  - `resnet.py` - Standard ResNet
- `logs/` - The log folders for each dataset
- `compare_resnets.py` - Trains all models on the provided datasets
- `dataloader.py` - Loads the datasets from PyTorch
- `requirements.txt` - The required libraries to install to run the project

## Links
Based on:
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://ieeexplore.ieee.org/document/8953658) by Deng et al. 2019
- [Recalibrating Fully Convolutional Networks With Spatial and Channel “Squeeze and Excitation” Blocks](https://ieeexplore.ieee.org/document/8447284) by Roy et al. 2018

Tested on the following datasets:
- [MNIST](https://ieeexplore.ieee.org/document/726791)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR10 & CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
