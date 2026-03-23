# RoBERTa Sentiment API with MLflow

This project exposes a Flask API for sentiment prediction using an ONNX RoBERTa model and uses MLflow for:

- tracking prediction runs
- storing prediction artifacts
- packaging a serveable model

This project is doing sentiment classification, not summarization.

## Project Structure

- `web_app/app.py`: Flask API for prediction and MLflow run tracking
- `web_app/log_model.py`: logs a serveable MLflow `pyfunc` model
- `web_app/roberta-sequence-classification-9.onnx`: ONNX model used for inference
- `web_app/mlflow.db`: MLflow tracking database
- `web_app/mlruns/`: MLflow artifacts
- `Makefile`: local developer and CI commands

## End-to-End Flow

There are two main paths in this project:

1. Get an answer from the API
2. Log and serve a model with MLflow

### Path 1: Get an Answer from the API

#### 1. Install dependencies

```bash
make install
```

#### 2. Start the Flask API

```bash
make run
```

The API starts on:

```text
http://127.0.0.1:5000
```

#### 3. Send text to the model

Use this command in another terminal:

```bash
make predict
```

Or call the endpoint directly:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This product was excellent and very easy to use."}'
```

Example response:

```json
{
  "label": "positive",
  "positive": true,
  "confidence": 0.993
}
```

#### 4. What happens behind the scenes

When you call `/predict`, the app:

- tokenizes the input text
- runs the ONNX model
- returns the sentiment result
- creates an MLflow run
- logs params, metrics, tags, and artifacts for that prediction

#### 5. See tracked runs from the API

```bash
make runs
```

Or:

```bash
curl http://127.0.0.1:5000/runs
```

## Path 2: See the Tracking Data in MLflow

#### 1. Start the MLflow UI

```bash
make mlflow-ui
```

Then open:

```text
http://127.0.0.1:5001
```

#### 2. What you will see

Inside MLflow UI, open the experiment:

```text
roberta-sentiment-api
```

For each prediction run, you will see:

- params:
  - model name
  - tokenizer
  - input size
  - input preview
- metrics:
  - confidence
  - predicted class
- tags:
  - endpoint
  - prediction label
- artifacts:
  - `prediction/input.txt`
  - `prediction/prediction.json`

## Path 3: Log a Serveable MLflow Model

Prediction tracking and model packaging are separate steps in this project.

The API logs inference runs.
The model packaging script logs a real MLflow model artifact.

#### 1. Log the model

```bash
make log-model
```

Or:

```bash
python web_app/log_model.py
```

This prints a `run_id`.

Example:

```text
78447675102e40d38a3b5d4267717b11
```

#### 2. What this creates

This script creates a new MLflow run that contains:

- a `model` artifact
- the ONNX model packaged inside the MLflow model
- input/output signature information
- example serving input

This is the run type you can use with `mlflow models serve`.

## Path 4: Serve the Logged Model

Use the run id printed by `make log-model`.

Start model serving with:

```bash
make serve-model
```

By default, the `Makefile` uses:

- tracking DB: `sqlite:///web_app/mlflow.db`
- model run id: `78447675102e40d38a3b5d4267717b11`
- serving port: `5003`

If you want to override the run id:

```bash
make serve-model MODEL_RUN_ID=YOUR_RUN_ID
```

Equivalent direct command:

```bash
cd web_app
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve \
  -m runs:/YOUR_RUN_ID/model \
  -p 5003 \
  --env-manager local
```

## Path 5: Call the Served MLflow Model

Once the model server is running, call it with:

```bash
curl -X POST http://127.0.0.1:5003/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"text": "This product was excellent and very easy to use."}
    ]
  }'
```

Example response:

```json
[
  {
    "label": "positive",
    "positive": true,
    "confidence": 0.993
  }
]
```

## Local CI

Run the local CI-style checks with:

```bash
make ci
```

This runs:

- Python compile checks
- lightweight project validation
- Flask API smoke tests

## GitHub Actions

This project includes a GitHub Actions workflow:

- `.github/workflows/ci.yml`

It runs on:

- pushes to `main` or `master`
- pull requests

And executes:

```bash
make install
make ci
```

## Useful Make Commands

```bash
make help
make install
make run
make predict
make runs
make mlflow-ui
make log-model
make serve-model
make ci
```

## Tracking Files

- MLflow database: `web_app/mlflow.db`
- MLflow artifacts: `web_app/mlruns`

## Repository Notes

Large local files are not committed to git:

- `web_app/roberta-sequence-classification-9.onnx`
- `web_app/mlruns/`
- `web_app/mlflow.db`

Keep those files locally when running the project. GitHub rejects files larger than 100 MB, so the model binary and generated MLflow outputs must stay outside the repository history.

In CI, the project runs without the local ONNX file. The app now handles that case gracefully:

- `/` still works
- `/runs` still works
- `/predict` returns `503` until the model file is present locally
# RoBERTa-Sentiment-Analysis-API-with-MLflow-Tracking-and-Model-Serving
