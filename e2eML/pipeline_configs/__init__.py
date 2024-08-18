from dagster import Config

from ..models import ModelClassifierEnum


class MNISTConfig(Config):
    """Configuration class for MNIST dataset processing and model training.

    Attributes:
        train_folder_path (str): Path to the folder containing the MNIST training data.
        test_folder_path (str): Path to the folder containing the MNIST test data.
        train_filename (str): Filename of the MNIST training data file.
        test_filename (str): Filename of the MNIST test data file.
        validation_percentage (float): Percentage of the training data to be used for validation.
        epochs (int): Number of epochs for training the model.
        batch_size (int): Number of samples per batch during training and evaluation.
        shuffle (bool): Whether to shuffle the training data.
        ml_model_names (List[str]): List of model names to be used for training, derived from ModelClassifierEnum.
    """

    train_folder_path: str = "data/MNIST/train/"
    test_folder_path: str = "data/MNIST/test/"
    train_filename: str = "mnist_train_data.pt"
    test_filename: str = "mnist_test_data.pt"
    validation_percentage: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    shuffle: bool = True
    ml_model_names: list[ModelClassifierEnum] = [
        member.name for member in ModelClassifierEnum
    ]
