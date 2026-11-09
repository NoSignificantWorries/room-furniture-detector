# room-furniture-detector

Automatic furniture detector that extracts bounding boxes with class labels from room photos and outputs them as JSON. Built for real-estate systems to automate furniture description.

## Setup

### Web
```bash
uv sync --extra web

source ./.venv/bin/activate

cd web
npm install

# uv run pre-commit install
```

### Service
```bash
uv sync --extra serve

source ./.venv/bin/activate

# cd web
# npm install
```

## Train

### Prepare the dataset

Download and prepare the dataset:

```bash
uv run --locked --extra train room-detector download_data
```

The dataset source and expected MD5 are stored in
`data/homeobjects-3K.zip.dvc`. If the archive is missing, the downloader
retrieves it from GitHub Releases using `dvc get-url` and verifies its MD5.
An existing archive is reused.

The downloader extracts `images/` and `labels/` and checks the required
training and validation directories. Google Drive authorization and a DVC
remote are not required for this workflow.

Dataset preparation is also called automatically before training.

### Configure training

Training uses Hydra configurations from `configs/`. Application settings
are defined in `src/room_furniture_detector/config.py` and can be
overridden using environment variables with the same names.

Examples of configurable settings:

- `PROJECT_ROOT`
- `HYDRA_CONFIG_DIR`
- `HYDRA_CONFIG_NAME`
- `DATA_ROOT`
- `DATA_ARCHIVE_PATH`
- `DVC_TARGET`
- `DVC_EXECUTABLE`
- `TRAIN_OUTPUT_DIR`
- `TRAIN_INITIAL_WEIGHTS`
- `TRAIN_RUN_NAME`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`

Hydra configurations reference application settings using the
`${app:SETTING_NAME}` resolver.

Print the resolved configuration without downloading data or starting
training:

```bash
uv run --locked --extra train room-detector train --dry_run=True
```

Hydra overrides can be supplied as a quoted string:

```bash
uv run --locked --extra train room-detector train --dry_run=True --overrides="training.epochs=1"
```

### Start MLflow

The default tracking URI is `http://127.0.0.1:8080`.

Start a local tracking server in a separate terminal if one is not already
running:

```bash
uv run --locked --extra train mlflow server --host 127.0.0.1 --port 8080
```

### Start training

```bash
uv run --locked --extra train room-detector train
```

Training prepares the dataset and split manifests, creates the dataset
configuration, and starts the model training run. The integration records
the resolved configuration, Git metadata, and split artifacts in MLflow.

## Production preparation

### Download model artifacts

Model artifacts are stored outside Git. The downloader retrieves files
from Google Drive and verifies their SHA-256 checksums.

Download the artifact required for the selected backend:

```bash
uv run --locked --extra artifacts download-models --backend onnx
```

```bash
uv run --locked --extra artifacts download-models --backend tensorrt
```

Download PT weights for prediction or export:

```bash
uv run --locked --extra artifacts download-models --backend pt
```

Download all artifacts:

```bash
uv run --locked --extra artifacts download-models --backend all
```

Existing files are checked against the configured SHA-256 values.
A checksum mismatch raises an error instead of silently replacing a file.

The TensorRT selection downloads both the engine and its metadata.
The downloader checks the engine and source ONNX hashes recorded in the
metadata.

Paths, Google Drive file IDs, and expected checksums are configured in
`config.py` and can be overridden through environment variables.

### Export ONNX

Export the PT model configured by `MODEL_PT_PATH`:

```bash
uv run --locked --extra train --extra export-onnx export-onnx
```

Relevant settings:

- `MODEL_PT_PATH`
- `MODEL_ONNX_PATH`
- `EXPORT_IMGSZ`
- `EXPORT_OPSET`
- `EXPORT_DEVICE`

The exporter creates a static model with batch size 1 and without
embedded NMS. It validates the ONNX graph and checks that ONNX Runtime
can load it.

### Export TensorRT

Build a TensorRT engine from the ONNX model configured by
`MODEL_ONNX_PATH`:

```bash
uv run --locked --extra train --extra export-tensorrt export-tensorrt
```

Relevant settings:

- `MODEL_ONNX_PATH`
- `MODEL_TENSORRT_PATH`
- `MODEL_TENSORRT_METADATA_PATH`
- `TENSORRT_DEVICE`
- `TENSORRT_WORKSPACE_GB`

The exporter requires an available CUDA GPU. It builds the engine,
checks deserialization, and writes build metadata including model
hashes, TensorRT version, CUDA version, and GPU information.

TensorRT engines are environment-dependent. An engine built on one
platform or GPU configuration must not be assumed compatible with a
different deployment environment. Rebuild from ONNX when necessary.

Both exporters refuse to overwrite existing output files. To create
another export, configure a different destination or deliberately move
the previous artifacts.

### Run tests

```bash
uv run --locked --extra serve pytest -q
```

Model weights and dataset archives must remain outside Git. The
repository ignores `.pt`, `.onnx`, `.engine`, `.plan`, and dataset archive
artifacts.

## Infer

### Start the ONNX service

Download the ONNX model first if it is not already available.

The default `MODEL_BACKEND` is `onnx`. The default ONNX backend uses CPU
execution.

Explicitly select the ONNX backend if another backend was previously
selected in the current terminal.

PowerShell:

```powershell
$env:MODEL_BACKEND = "onnx"
```

Bash:

```bash
export MODEL_BACKEND=onnx
```

Start the service:

```bash
uv run --locked --extra serve uvicorn room_furniture_detector.api:app --host 127.0.0.1 --port 8000 --workers 1
```

### Start the TensorRT service

Set `MODEL_BACKEND=tensorrt` before starting the service.

PowerShell:

```powershell
$env:MODEL_BACKEND = "tensorrt"
```

Bash:

```bash
export MODEL_BACKEND=tensorrt
```

Then run:

```bash
uv run --locked --extra serve --extra train --extra export-tensorrt uvicorn room_furniture_detector.api:app --host 127.0.0.1 --port 8000 --workers 1
```

The TensorRT backend requires a compatible engine and an available CUDA
GPU.

Use one worker: batch records and the processing queue are stored in
process memory. Results do not survive a service restart.

The `triton` backend is currently a placeholder and raises
`NotImplementedError`.

### API

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Check service availability |
| POST | `/batches` | Submit images for processing |
| GET | `/batches/{batch_id}` | Retrieve batch status and results |

`POST /batches` accepts multipart form data:

- `files`: one or more JPEG, PNG, or WebP images.
- `batch_id`: optional UUID; generated by the service if omitted.

The response contains `batch_id`, `image_ids`, and `status: "accepted"`.
Image IDs follow the order of the submitted files.

Poll `GET /batches/{batch_id}` until `ready` is `true`.

Batch states:

- `processing`: `ready=false`, `results=[]`.
- `done`: `ready=true`, results are available.
- `failed`: `ready=true`, `error` describes the failure.

Each image result contains:

- `image_id`
- `width` and `height`
- `boxes`, with `cls`, `conf`, and normalized `xyxy_norm` coordinates

Images within an API batch are processed sequentially. The exported
models use tensor batch size 1.

Completed results are retained according to `API_RESULT_TTL_SECONDS`.
After expiration and cleanup, requesting the batch returns `404`.

Relevant service settings include:

- `MODEL_BACKEND`
- `MODEL_ONNX_PATH`
- `MODEL_TENSORRT_PATH`
- `MODEL_CLASS_NAMES`
- `INFERENCE_IMGSZ`
- `INFERENCE_CONF`
- `INFERENCE_IOU`
- `INFERENCE_MAX_DET`
- `API_MAX_FILES`
- `API_MAX_UPLOAD_BYTES`
- `API_MAX_BATCH_BYTES`
- `API_QUEUE_SIZE`
- `API_MAX_STORED_BATCHES`
- `API_RESULT_TTL_SECONDS`

### Smoke test

With the service running, execute this script using the project
environment. It uses `httpx` from the development dependency group.

```python
import time
from pathlib import Path

import httpx

images = [
    Path("data/images/val/living_room_10.jpg"),
    Path("data/images/val/living_room_1000.jpg"),
]

with httpx.Client(
    base_url="http://127.0.0.1:8000",
    timeout=120,
    trust_env=False,
) as client:
    files = [
        ("files", (path.name, path.read_bytes(), "image/jpeg"))
        for path in images
    ]

    response = client.post("/batches", files=files)
    response.raise_for_status()
    accepted = response.json()
    print(accepted)

    batch_id = accepted["batch_id"]

    for _ in range(120):
        response = client.get(f"/batches/{batch_id}")
        response.raise_for_status()
        result = response.json()

        if result["ready"]:
            print(result)
            if result["status"] != "done":
                raise RuntimeError(result.get("error"))
            break

        time.sleep(1)
    else:
        raise TimeoutError(f"Batch is not ready: {batch_id}")
```

If Swagger UI does not display a file picker for the optional file-array
field, use a multipart HTTP client such as the script above.
