import json
from pathlib import Path
import tempfile

from flask import Flask, request, jsonify
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from transformers import RobertaTokenizer
import onnxruntime


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "roberta-sequence-classification-9.onnx"
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
EXPERIMENT_NAME = "roberta-sentiment-api"
tokenizer = None
session = None
model_load_error = None

MLRUNS_DIR.mkdir(exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")


def ensure_experiment():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=MLRUNS_DIR.as_uri(),
        )

    mlflow.set_experiment(EXPERIMENT_NAME)


ensure_experiment()


def load_model_components():
    global tokenizer, session, model_load_error

    if tokenizer is not None and session is not None:
        return tokenizer, session

    if not MODEL_PATH.exists():
        model_load_error = f"Missing model file: {MODEL_PATH}"
        return None, None

    try:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        session = onnxruntime.InferenceSession(str(MODEL_PATH))
        model_load_error = None
    except Exception as exc:
        model_load_error = str(exc)
        tokenizer = None
        session = None

    return tokenizer, session


@app.route("/")
def home():
    return "<h2>RoBERTa sentiment analysis API</h2>"


def log_prediction_artifacts(text, token_ids, logits, label, confidence):
    artifact_payload = {
        "input_text": text,
        "input_tokens": token_ids[:512],
        "token_count": len(token_ids[:512]),
        "logits": logits[0].tolist(),
        "prediction": {
            "label": label,
            "positive": label == "positive",
            "confidence": round(confidence, 4),
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "input.txt").write_text(text, encoding="utf-8")
        (temp_path / "prediction.json").write_text(
            json.dumps(artifact_payload, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(temp_path / "input.txt"), artifact_path="prediction")
        mlflow.log_artifact(
            str(temp_path / "prediction.json"),
            artifact_path="prediction",
        )


@app.route("/runs", methods=["GET"])
def list_runs():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        return jsonify({"experiment": EXPERIMENT_NAME, "runs": []})

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10,
    )

    records = []
    for _, row in runs.iterrows():
        records.append(
            {
                "run_id": row["run_id"],
                "status": row["status"],
                "start_time": row["start_time"].isoformat(),
                "label": row.get("tags.prediction_label"),
                "endpoint": row.get("tags.endpoint"),
                "confidence": row.get("metrics.confidence"),
                "input_preview": row.get("params.input_preview"),
            }
        )

    return jsonify({"experiment": EXPERIMENT_NAME, "runs": records})


@app.route("/predict", methods=["POST"])
def predict():
    tokenizer_instance, session_instance = load_model_components()

    if tokenizer_instance is None or session_instance is None:
        return (
            jsonify(
                {
                    "error": "Model is not available on this machine",
                    "details": model_load_error,
                }
            ),
            503,
        )

    payload = request.get_json(silent=True)

    if isinstance(payload, dict):
        text = payload.get("text", "")
    elif isinstance(payload, list) and payload:
        text = payload[0]
    else:
        return (
            jsonify({"error": "Send JSON like {'text': 'your sentence here'}"}),
            400,
        )

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "The 'text' value must be a non-empty string"}), 400

    token_ids = tokenizer_instance.encode(text, add_special_tokens=True)
    input_ids = np.asarray([token_ids[:512]], dtype=np.int64)
    inputs = {session_instance.get_inputs()[0].name: input_ids}
    logits = session_instance.run(None, inputs)[0]

    predicted_class = int(np.argmax(logits, axis=1)[0])
    shifted = logits[0] - np.max(logits[0])
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    confidence = float(probabilities[predicted_class])
    label = "positive" if predicted_class == 1 else "negative"
    truncated_text = text[:250]

    with mlflow.start_run(nested=True):
        mlflow.log_param("model_path", str(MODEL_PATH.name))
        mlflow.log_param("tokenizer", "roberta-base")
        mlflow.log_param("max_sequence_length", 512)
        mlflow.log_param("input_characters", len(text))
        mlflow.log_param("input_tokens", len(token_ids[:512]))
        mlflow.log_param("input_preview", truncated_text)
        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("predicted_class", predicted_class)
        mlflow.set_tag("prediction_label", label)
        mlflow.set_tag("endpoint", "/predict")
        log_prediction_artifacts(text, token_ids, logits, label, confidence)

    return jsonify(
        {
            "label": label,
            "positive": label == "positive",
            "confidence": round(confidence, 4),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
