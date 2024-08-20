import os
import logging
from logging import getLogger, StreamHandler
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from e2eML.clients.mlserver import MlServerClient, MlServerClientConfig
from e2eML.models import device

# Set up logging
logger = getLogger(__name__)


def initialize_mlserver_client() -> MlServerClient:
    """
    Initialize the MLServer client with configuration.
    """
    config = MlServerClientConfig(
        base_uri="http://0.0.0.0:9595",
        model_name="Net",
    )
    return MlServerClient.from_config(config)


def prepare_data_loader(batch_size: int = 100) -> DataLoader:
    """
    Prepare the DataLoader for the MNIST test dataset.

    Args:
        batch_size (int): The number of samples per batch.

    Returns:
        DataLoader: The DataLoader for the test set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


def evaluate_model(mlserver_client: MlServerClient, test_loader: DataLoader):
    """
    Evaluate the model using the MLServer client.

    Args:
        mlserver_client (MlServerClient): The initialized MLServer client.
        test_loader (DataLoader): The DataLoader for the test dataset.
    """
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            correct_predictions = 0

            for idx, sample in enumerate(X_batch):
                X = sample.to(device).unsqueeze(0)
                y_true = y_batch[idx].to(device)

                y_pred_response = mlserver_client.post(X)
                y_pred = int(y_pred_response["prediction"][0].argmax())

                correct_predictions += int(y_pred == y_true)

                if y_pred != y_true:
                    logger.info(f"Sample {idx} failed")
                    logger.info(f"Predicted: {y_pred}")
                    logger.info(f"Expected: {y_true}")

            accuracy = correct_predictions / len(y_batch)
            logger.info(f"Accuracy per batch: {accuracy * 100:.2f}%")
            break


if __name__ == "__main__":
    #check if env variables are set
    assert os.getenv("AWS_ACCESS_KEY_ID"), "AWS_ACCESS_KEY_ID not set"
    assert os.getenv("AWS_SECRET_ACCESS_KEY"), "AWS_SECRET_ACCESS_KEY not set"
    assert os.getenv("MLFLOW_S3_ENDPOINT_URL"), "MLFLOW_S3_ENDPOINT_URL not set"

    # Set up logging to print to stdout
    logging.basicConfig(level=logging.INFO, handlers=[StreamHandler()])
    mlserver_client = initialize_mlserver_client()
    test_loader = prepare_data_loader(batch_size=100)
    evaluate_model(mlserver_client, test_loader)
