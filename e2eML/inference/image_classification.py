import os
from logging import getLogger
import mlflow
import torch
from mlflow.pytorch import load_model
from mlserver import MLModel, ModelSettings
from mlserver.codecs import NumpyRequestCodec
from mlserver.types import InferenceRequest, InferenceResponse
from torch import nn

logger = getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_image(model: nn.Module, x_test) -> int:
    """
    This function is responsible for making predictions using the loaded model.
    Args:
        model: The loaded model.
        x_test: The input data for the prediction.
    Returns:
        int: The prediction results.
    """
    x_test = torch.tensor(x_test)
    x_test = x_test.to(device)
    y_pred = model(x_test)
    return y_pred.cpu().numpy()


class MinstModel(MLModel):
    def __init__(self, settings: ModelSettings):
        super().__init__(settings)
        self.model_name = None
        # we need to name the logger mlserver to match the logger configuration in the mlserver package
        LoggerName = "mlserver"
        self.logger = getLogger(LoggerName)

    async def load(self):
        """
        This method is responsible for loading the model
        It reads a model and a scaler from files. The model path needs to be an
        MLflow "YAML model representation", or the directory containing that
        file. See the definition here:
        <https://mlflow.org/docs/latest/models.html#fields-in-the-mlmodel-format>

        Returns:
            bool: True if the model and scaler are loaded, False otherwise.
        """

        # we can't use orquestra.sdk.mlflow.get_tracking_uri because we don't have $HOME/.orquestra/config.json in build time for docker image
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5005")
        mlflow.set_tracking_uri(tracking_uri)
        self.model_name = self._settings.name
        alias = os.getenv("MODEL_ALIAS", "stage")
        # Load the model and its scaler
        model_path = f"models:/{self.model_name}@{alias}"

        print(f"model name: {self.model_name}")
        model = load_model(model_path)

        self.logger.info(f"model name: {self.model_name}")

        self._model = model

        if not self._model:
            raise RuntimeError("Failed to load model")

        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """
        This method is responsible for making predictions using the loaded
        model.

        Args:
            payload (InferenceRequest): The input data for the prediction.

        Returns:
            InferenceResponse: The prediction results.
        """
        x_test = NumpyRequestCodec.decode_request(payload)
        # convert to torch tensor
        self._model.eval()
        with torch.no_grad():
            y_predict = infer_image(
                model=self._model,
                x_test=x_test,
            )
        logger.info(f"Predicted image: {y_predict}")
        reponse = NumpyRequestCodec.encode_response(
            model_name=self.model_name,
            payload=y_predict,
            id=payload.id,
        )
        logger.info(f"Response: {reponse}")
        return reponse
