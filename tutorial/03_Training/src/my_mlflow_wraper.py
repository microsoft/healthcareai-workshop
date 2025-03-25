import base64

import mlflow
from mlflow.types.schema import Schema, ColSpec

import numpy as np
import pandas as pd
import torch

from my_model import XRVDenseNetLightning

from utils import build_transforms, xrv_read_dicom_and_normalize_image

def build_signature():
    input_schema = Schema([
    ColSpec("binary", "image",required=False),  # Binary data for the image
    ])

    # Define the schema for the output data
    output_schema = Schema([
        ColSpec("string", "predictions",required=False),  # String data for image features
    ])
    # Create the model signature with both input and output schemas
    return mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema, params=None)

class MyMLFlowWraper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_dir=None):
        self._model = None
        self._model_dir = model_dir

    @staticmethod
    def load_model(model_dir):
        return XRVDenseNetLightning.load_from_checkpoint(model_dir)
        
    def load_context(self, context):
        # Load the model from the specified model directory
        model_dir = context.artifacts["model_dir"]
        self.model = self.load_model(model_dir)

        # Define image transformation pipeline
        self.transform = build_transforms()

        if torch.cuda.is_available():
            self._device = torch.device(type="cuda", index=0)
        else:
            self._device = torch.device(type="cpu")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Decodes base64 images (in a column named 'image'), parses comma-separated labels
        (in a column named 'text'), then runs inference on the loaded model.
        Returns a DataFrame with columns 'probs' and 'labels'.
        """
        # Decode base64 images
        decoded_images = input_data['image'].apply(lambda x: base64.b64decode(x))
        predictions = self.run_inference_batch(decoded_images)
        df_result = pd.DataFrame({
            "predictions": [p.tolist() for p in predictions],
        })

        return df_result
    
    def run_inference_batch(self, img_list):

        imgs = [xrv_read_dicom_and_normalize_image(data) for data in img_list]
        input_batch = np.stack([self.transform(i) for i in imgs])
        input_batch = torch.tensor(input_batch).float()
        with torch.no_grad():
            logits = self.model(input_batch)

        return logits.detach().numpy()
    


        
