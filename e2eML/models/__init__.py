from torch import nn
from .classifiers import (
    ModelClassifierEnum,
    NeuralNetworkRELU,
    NeuralNetworkELU,
    device,
    CNN,
    LeNet5,
    Net,
)

__all__ = ["get_model", "ModelClassifierEnum"]


def get_model(model_name) -> nn.Module:
    """
    Factory method to get the model
    Args:
        model_name: Name of the model to get
    Returns:
        nn.Module: The model
    """
    model_registry = {
        ModelClassifierEnum.NeuralNetworkRELU: NeuralNetworkRELU,
        ModelClassifierEnum.NeuralNetworkELU: NeuralNetworkELU,
        ModelClassifierEnum.CNN: CNN,
        ModelClassifierEnum.LeNet5: LeNet5,
        ModelClassifierEnum.Net: Net,
    }
    return model_registry[model_name]().to(device)
