import base64
import json
from io import BytesIO

import requests
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor

from dagster import (
    MaterializeResult,
    MetadataValue,
    asset, AssetExecutionContext, DataVersion, Config
)


class MNISTConfig(Config):
    training_save_path: str = "data/MNIST/train/mnist_training_data.pt"
    test_save_path: str = "data/MNIST/test/mnist_test_data.pt"


@asset
def train_MNIST_data(context: AssetExecutionContext, config : MNISTConfig) -> MaterializeResult:
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    context.log.info(f"Training data: {training_data}")
    context.log.info(f"Raw data save to: {training_data.raw_folder}")
    torch.save(training_data, config.training_save_path)
    context.log.info(f"Data saved to {config.training_save_path}")

    # Make a bar chart of the top 25 words
    img, target = training_data[0]
    plt.figure(figsize=(10, 6))
    plt.title(f"Label: {target}")
    plt.imshow(img[0], cmap='gray')
    plt.axis('off')
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
def test_MNIST_data(context: AssetExecutionContext, config: MNISTConfig) -> MaterializeResult:
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    context.log.info(f"Training data: {test_data}")
    context.log.info(f"Raw data save to {test_data.raw_folder}")
    torch.save(test_data, config.test_save_path)
    context.log.info(f"Data saved to {config.test_save_path}")

    # Make a bar chart of the top 25 words
    img, target = test_data[0]
    plt.figure(figsize=(10, 6))
    plt.title(f"Label: {target}")
    plt.imshow(img[0], cmap='gray')
    plt.axis('off')
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



