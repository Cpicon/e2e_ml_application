from enum import Enum
import torch
from torch import nn

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def activation_layer(
    activation: str = "relu", alpha: float = 0.1, inplace: bool = True
) -> nn.Module:
    """Activation layer wrapper for LeakyReLU and ReLU activation functions

    Args:
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)

    Returns:
        torch.Tensor: activation layer
    """
    activation_function = {
        "relu": nn.ReLU(inplace=inplace),
        "leaky_relu": nn.LeakyReLU(negative_slope=alpha, inplace=inplace),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
    }
    return activation_function[activation]


class NeuralNetworkRELU(nn.Module):
    """A simple feedforward neural network with ReLU activations.

    This network consists of two hidden layers, each with 512 neurons,
    followed by ReLU activation functions, and an output layer with 10 neurons.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            activation_layer("relu"),
            nn.Linear(512, 512),
            activation_layer("relu"),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetworkELU(nn.Module):
    """A simple feedforward neural network with ELU activations.

    This network consists of three hidden layers, each with 512 neurons,
    followed by ELU activation functions, and an output layer with 10 neurons.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            activation_layer("elu"),
            nn.Linear(512, 512),
            activation_layer("elu"),
            nn.Linear(512, 512),
            activation_layer("elu"),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Net(nn.Module):
    """A convolutional neural network with ReLU activations and dropout.

    This network consists of three convolutional layers, followed by
    fully connected layers with ReLU activations and dropout.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            activation_layer("relu"),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            activation_layer("relu"),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3),
            activation_layer("relu"),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 50),  # Adjusted input size
            activation_layer("relu"),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.net(x)


class CNN(nn.Module):
    """A simple convolutional neural network with ReLU activations and dropout.

    This network consists of one convolutional layer, followed by ReLU
    activations, max pooling, dropout, and two fully connected layers.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2),
            activation_layer("relu"),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(27 * 27 * 10, 128),
            activation_layer("relu"),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.net(x)


class LeNet5(nn.Module):
    """Implementation of the LeNet-5 architecture with Tanh activations.

    LeNet-5 is a classical convolutional neural network architecture
    originally designed for handwritten and machine-printed character recognition.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            activation_layer("tanh"),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            activation_layer("tanh"),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            activation_layer("tanh"),
            nn.Flatten(),
            nn.Linear(120 * 1 * 1, 84),
            activation_layer("tanh"),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.net(x)


class ModelClassifierEnum(str, Enum):
    NeuralNetworkRELU = "NeuralNetworkRELU"
    NeuralNetworkELU = "NeuralNetworkELU"
    CNN = "CNN"
    LeNet5 = "LeNet5"
    Net = "Net"
