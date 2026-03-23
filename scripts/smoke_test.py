import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("webapp", "web_app/app.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

client = module.app.test_client()
model_exists = Path("web_app/roberta-sequence-classification-9.onnx").exists()

home = client.get("/")
assert home.status_code == 200, home.status_code

predict = client.post(
    "/predict",
    json={"text": "This product was excellent and easy to use."},
)
payload = predict.get_json()
if model_exists:
    assert predict.status_code == 200, predict.get_data(as_text=True)
    assert "label" in payload and "confidence" in payload and "positive" in payload, payload
else:
    assert predict.status_code == 503, predict.get_data(as_text=True)
    assert payload["error"] == "Model is not available on this machine", payload

runs = client.get("/runs")
assert runs.status_code == 200, runs.get_data(as_text=True)

print("Smoke tests passed.")
