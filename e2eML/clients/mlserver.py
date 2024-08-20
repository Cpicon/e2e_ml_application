from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from requests import Response, post
import torch
from logging import getLogger

from mlserver.codecs import NumpyCodec
from mlserver.types import InferenceRequest, InferenceResponse

log = getLogger(__name__)


@dataclass
class MlServerClientConfig:
    model_name: str
    base_uri: str = "http://0.0.0.0:9597"


class MlServerClient:
    """Encapsulates the API contract with MLServer for MNIST models."""

    def __init__(self, base_uri: str, model_name: str):
        self._base_uri = base_uri
        self._model_name = model_name

    @classmethod
    def from_config(cls, conf: MlServerClientConfig) -> "MlServerClient":
        return cls(
            base_uri=conf.base_uri,
            model_name=conf.model_name,
        )

    def post(
        self,
        image: torch.Tensor,
    ) -> dict[str, Any]:
        request_model = self._create_request(image)
        log.info(f"Sending request to {self._infer_uri}")
        resp = self._post_request(body_model=request_model)
        log.info(f"Received response from {self._infer_uri}")
        output_values = self._parse_response(response=resp)
        return output_values

    @staticmethod
    def _create_request(
        x: torch.Tensor,
    ) -> InferenceRequest:
        inference_request = InferenceRequest(
            id=str(uuid4()),
            inputs=[
                NumpyCodec.encode_input(name="x_hist_data", payload=x.cpu().numpy()),
            ],
        )
        return inference_request

    def _post_request(self, body_model: BaseModel) -> Response:
        resp = post(
            url=self._infer_uri,
            data=body_model.model_dump_json(),
            headers={
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp

    @property
    def _infer_uri(self) -> str:
        return f"{self._base_uri}/v2/models/{self._model_name}/infer"

    @staticmethod
    def _parse_response(response: Response) -> dict[str, Any]:
        response_model = InferenceResponse.model_validate(response.json())
        return {
            "prediction": NumpyCodec.decode_output(output)
            for output in response_model.outputs
        }
