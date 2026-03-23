from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
import onnxruntime
from transformers import RobertaTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "roberta-sequence-classification-9.onnx"
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
EXPERIMENT_NAME = "roberta-sentiment-api"


class RobertaSentimentPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.session = onnxruntime.InferenceSession(context.artifacts["onnx_model"])

    def predict(self, context, model_input):
        if "text" not in model_input.columns:
            raise ValueError("Input must contain a 'text' column")

        records = []
        for text in model_input["text"].fillna("").astype(str):
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            input_ids = np.asarray([token_ids[:512]], dtype=np.int64)
            inputs = {self.session.get_inputs()[0].name: input_ids}
            logits = self.session.run(None, inputs)[0]

            predicted_class = int(np.argmax(logits, axis=1)[0])
            shifted = logits[0] - np.max(logits[0])
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            confidence = float(probabilities[predicted_class])
            label = "positive" if predicted_class == 1 else "negative"

            records.append(
                {
                    "label": label,
                    "positive": label == "positive",
                    "confidence": round(confidence, 4),
                }
            )

        return pd.DataFrame(records)


def main():
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    input_example = pd.DataFrame(
        {"text": ["This movie was excellent and very enjoyable."]}
    )
    output_example = pd.DataFrame(
        [{"label": "positive", "positive": True, "confidence": 0.9953}]
    )
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(run_name="roberta-pyfunc-model") as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RobertaSentimentPyFunc(),
            artifacts={"onnx_model": str(MODEL_PATH)},
            input_example=input_example,
            signature=signature,
        )
        mlflow.log_param("model_type", "pyfunc")
        mlflow.log_param("base_model_file", MODEL_PATH.name)
        print(run.info.run_id)


if __name__ == "__main__":
    main()
