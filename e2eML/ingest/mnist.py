import base64
import os
from io import BytesIO
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from dagster import (
    MaterializeResult,
    MetadataValue,
    asset,
    AssetExecutionContext,
    Config,
    AssetOut,
    multi_asset,
)


class MNISTConfig(Config):
    train_folder_path: str = "data/MNIST/train/"
    test_folder_path: str = "data/MNIST/test/"
    train_filename: str = "mnist_train_data.pt"
    test_filename: str = "mnist_test_data.pt"
    batch_size: int = 64
    shuffle: bool = True


@asset
def train_MNIST_data(
    context: AssetExecutionContext, config: MNISTConfig
) -> MaterializeResult:
    """Downloads, processes, and saves the MNIST training data.

    Args:
        context (AssetExecutionContext): The execution context provided by Dagster.
        config (MNISTConfig): Configuration settings including paths and filenames.

    Returns:
        MaterializeResult: The result of materializing the training data, including metadata such as
            the number of records, a preview of the data, and the label mappings.
    """
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    context.log.info(f"Training data: {training_data}")
    context.log.info(f"Raw data save to: {training_data.raw_folder}")
    os.makedirs(config.train_folder_path, exist_ok=True)
    save_path = os.path.join(config.train_folder_path, config.train_filename)
    torch.save(training_data, save_path)
    context.log.info(f"Data saved to {save_path}")

    img, target = training_data[0]
    plt.figure(figsize=(10, 6))
    plt.title(f"Label: {target}")
    plt.imshow(img[0], cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    # Convert the image to a saveable format
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    image_data = base64.b64encode(buffer.getvalue())

    # Convert the image to Markdown to preview it within Dagster
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"

    return MaterializeResult(
        metadata={
            "num_records": len(training_data),  # Metadata can be any key-value pair
            "info": MetadataValue.text(str(training_data)),
            "preview": MetadataValue.md(md_content),
            "labels": MetadataValue.json(training_data.class_to_idx),
        }
    )


@asset
def test_MNIST_data(
    context: AssetExecutionContext, config: MNISTConfig
) -> MaterializeResult:
    """Downloads, processes, and saves the MNIST test data.

    Args:
        context (AssetExecutionContext): The execution context provided by Dagster.
        config (MNISTConfig): Configuration settings including paths and filenames.

    Returns:
        MaterializeResult: The result of materializing the test data, including metadata such as
            the number of records, a preview of the data, and the label mappings.
    """
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    context.log.info(f"Training data: {test_data}")
    context.log.info(f"Raw data save to {test_data.raw_folder}")

    os.makedirs(config.test_folder_path, exist_ok=True)
    save_path = os.path.join(config.test_folder_path, config.test_filename)
    torch.save(test_data, save_path)
    context.log.info(f"Data saved to {save_path}")

    img, target = test_data[0]
    plt.figure(figsize=(10, 6))
    plt.title(f"Label: {target}")
    plt.imshow(img[0], cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    # Convert the image to a saveable format
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    image_data = base64.b64encode(buffer.getvalue())

    # Convert the image to Markdown to preview it within Dagster
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"

    return MaterializeResult(
        metadata={
            "num_records": len(test_data.data),  # Metadata can be any key-value pair
            "info": MetadataValue.text(str(test_data)),
            "preview": MetadataValue.md(md_content),
            "labels": MetadataValue.json(test_data.class_to_idx),
        }
    )


@multi_asset(
    outs={
        "test_dataloader": AssetOut(),
        "train_dataloader": AssetOut(),
    },
    deps=[train_MNIST_data, test_MNIST_data],
)
def get_data_loader(context: AssetExecutionContext, config: MNISTConfig):
    """Loads the MNIST training and test data and returns DataLoaders.

    Args:
        context (AssetExecutionContext): The execution context provided by Dagster.
        config (MNISTConfig): Configuration settings including paths, filenames, and batch size.

    Returns:
        tuple: A tuple containing the test and train DataLoader objects.
    """
    training_path = os.path.join(config.train_folder_path, config.train_filename)
    training_data = torch.load(training_path)
    context.log.info(f"Training data loaded: {training_data}")
    train_dataloader = DataLoader(training_data, batch_size=config.batch_size, shuffle=config.shuffle)

    test_path = os.path.join(config.test_folder_path, config.test_filename)
    test_data = torch.load(test_path)
    context.log.info(f"Test data loaded: {test_data}")
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    return test_dataloader, train_dataloader


@asset
def print_data(train_dataloader) -> MaterializeResult:
    """Generates and logs a visual preview of the first batch of MNIST training data.

    Args:
        train_dataloader (DataLoader): The DataLoader for the MNIST training data.

    Returns:
        MaterializeResult: The result of materializing the data preview, including metadata
            such as the shape and dtype of the data, and a visual preview of the images.
    """
    image_data = None
    for X, y in train_dataloader:
        # Create a figure with subplots arranged in a grid
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns

        for i in range(10):
            ax = axes[i // 5, i % 5]  # Determine the position in the grid
            ax.imshow(X[i][0], cmap="gray")  # Display the image
            ax.set_title(f"Label: {y[i].item()}")  # Set the title as the label
            ax.axis("off")  # Hide the axes

        plt.tight_layout()  # Adjust subplots to fit in the figure area

        # Convert the image to a saveable format
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        image_data = base64.b64encode(buffer.getvalue())
        break

    if image_data is not None:
        md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    else:
        md_content = "No data to display"

    return MaterializeResult(
        metadata={
            "preview": MetadataValue.md(md_content),
        },
    )
