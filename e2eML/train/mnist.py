import torch
import torch.nn as nn
import torch.optim as optim
from dagster import (
    MaterializeResult,
    MetadataValue,
    AssetExecutionContext,
    AssetOut,
    multi_asset,
    OpExecutionContext,
)
from torch.utils.data import DataLoader

from ..ingest import get_data_loader
from ..models import ModelClassifierEnum, get_model
from ..pipeline_configs import MNISTConfig


def train(
    context: OpExecutionContext | AssetExecutionContext,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    n_epochs: int,
    trainloader: DataLoader,
    valoader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Trains the model on the training data and evaluates it on the validation data.

    Args:
        context (OpExecutionContext | AssetExecutionContext): The execution context provided by Dagster.
        model (nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer used for training the model.
        loss_fn (nn.Module): The loss function used to compute the loss.
        n_epochs (int): The number of epochs to train the model.
        trainloader (DataLoader): DataLoader providing training batches.
        valoader (DataLoader): DataLoader providing validation batches.
        device (torch.device): The device to which tensors and the model should be moved (CPU or GPU).

    Returns:
        dict[str, float]: A dictionary where keys are epoch identifiers and values are accuracy percentages.
    """
    acc_per_epoch: dict[str, float] = {}
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in trainloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        count = 0
        acc = 0
        with torch.no_grad():
            for X_batch, y_batch in valoader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
                count += len(y_batch)
        acc = acc / count
        acc = float(acc * 100)
        acc_per_epoch[f"epoch_{epoch}_accuracy"] = acc
        context.log.info("Epoch %d: model accuracy %.2f%%" % (epoch, acc))
    return acc_per_epoch


def get_device() -> str:
    """Determines the device to be used for training (CUDA, MPS, or CPU).

    Returns:
        torch.device: The device to be used for training.
    """
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    """Creates an Adam optimizer for the given model.

    Args:
        model (nn.Module): The PyTorch model for which the optimizer is created.

    Returns:
        optim.Optimizer: The Adam optimizer configured for the model's parameters.
    """
    return optim.Adam(model.parameters())


@multi_asset(
    outs={member.name: AssetOut(is_required=False) for member in ModelClassifierEnum},
)
def train_model(context: AssetExecutionContext, config: MNISTConfig):
    """Trains multiple models on the MNIST dataset and yields results for each model.

    Args:
        context (AssetExecutionContext): The execution context provided by Dagster.
        config (MNISTConfig): Configuration settings including paths, filenames, number of epochs, and model names.

    Yields:
        MaterializeResult: Contains metadata for each model including accuracy, optimizer, and loss function details.
    """
    n_epochs = config.epochs
    test_dataloader, val_dataloader, train_dataloader = get_data_loader(context, config)
    device = get_device()
    accuracies = {}
    model_names = config.ml_model_names
    for model_name in model_names:
        # context.log.info(f"Training model {key}")
        model = get_model(model_name)
        optimizer = get_optimizer(model)
        loss_fn = nn.CrossEntropyLoss()
        acc = train(
            context,
            model,
            optimizer,
            loss_fn,
            n_epochs,
            train_dataloader,
            val_dataloader,
            device,
        )
        accuracies[model_name] = acc

        yield MaterializeResult(
            asset_key=model_name,
            metadata={
                "ml_model_name": MetadataValue.text(
                    model_name.name
                ),  # Metadata can be any key-value pair
                "accuracy": MetadataValue.json(acc),
                "optimizer": MetadataValue.text(optimizer.__class__.__name__),
                "loss": MetadataValue.text(loss_fn.__class__.__name__),
            },
        )
