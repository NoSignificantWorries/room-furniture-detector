import os
from pathlib import Path


def _env_path(name: str, default: Path, base: Path) -> Path:
    value = Path(os.environ.get(name, str(default))).expanduser()
    if not value.is_absolute():
        value = base / value
    return value.resolve()


_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = (
    Path(os.environ.get("PROJECT_ROOT", str(_DEFAULT_ROOT)))
    .expanduser()
    .resolve()
)

HYDRA_CONFIG_DIR = _env_path(
    "HYDRA_CONFIG_DIR",
    PROJECT_ROOT / "configs",
    PROJECT_ROOT,
)
HYDRA_CONFIG_NAME = os.environ.get("HYDRA_CONFIG_NAME", "config")

DATA_ROOT = _env_path(
    "DATA_ROOT",
    PROJECT_ROOT / "data",
    PROJECT_ROOT,
)
DATA_ARCHIVE_PATH = _env_path(
    "DATA_ARCHIVE_PATH",
    DATA_ROOT / "homeobjects-3K.zip",
    PROJECT_ROOT,
)
DVC_TARGET = os.environ.get(
    "DVC_TARGET",
    "data/homeobjects-3K.zip.dvc",
)
DVC_REMOTE = os.environ.get("DVC_REMOTE", "")
DVC_EXECUTABLE = os.environ.get("DVC_EXECUTABLE", "dvc")

MODEL_PT_PATH = _env_path(
    "MODEL_PT_PATH",
    PROJECT_ROOT
    / "outputs"
    / "yolo11s_balanced832_nomosaic_ft"
    / "weights"
    / "best.pt",
    PROJECT_ROOT,
)

MODEL_ONNX_PATH = _env_path(
    "MODEL_ONNX_PATH",
    PROJECT_ROOT / "models" / "furniture_detector.onnx",
    PROJECT_ROOT,
)

EXPORT_IMGSZ = int(os.environ.get("EXPORT_IMGSZ", "832"))
EXPORT_OPSET = int(os.environ.get("EXPORT_OPSET", "17"))
EXPORT_DEVICE = os.environ.get("EXPORT_DEVICE", "cpu")

MODEL_TENSORRT_PATH = _env_path(
    "MODEL_TENSORRT_PATH",
    PROJECT_ROOT / "models" / "furniture_detector.engine",
    PROJECT_ROOT,
)

TENSORRT_DEVICE = int(os.environ.get("TENSORRT_DEVICE", "0"))
TENSORRT_WORKSPACE_GB = float(os.environ.get("TENSORRT_WORKSPACE_GB", "1"))

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "onnx").lower()

INFERENCE_IMGSZ = int(os.environ.get("INFERENCE_IMGSZ", "832"))
INFERENCE_CONF = float(os.environ.get("INFERENCE_CONF", "0.43"))
INFERENCE_IOU = float(os.environ.get("INFERENCE_IOU", "0.7"))
INFERENCE_MAX_DET = int(os.environ.get("INFERENCE_MAX_DET", "300"))
INFERENCE_MAX_NMS = int(os.environ.get("INFERENCE_MAX_NMS", "30000"))

MODEL_CLASS_NAMES = tuple(
    name.strip()
    for name in os.environ.get(
        "MODEL_CLASS_NAMES",
        "bed,sofa,chair,table,lamp,tv,laptop,wardrobe,"
        "window,door,potted plant,photo frame",
    ).split(",")
)

API_MAX_FILES = int(os.environ.get("API_MAX_FILES", "10"))
API_MAX_UPLOAD_BYTES = int(
    os.environ.get("API_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))
)
API_MAX_BATCH_BYTES = int(
    os.environ.get("API_MAX_BATCH_BYTES", str(50 * 1024 * 1024))
)
API_MAX_IMAGE_PIXELS = int(os.environ.get("API_MAX_IMAGE_PIXELS", "25000000"))
API_MAX_STORED_BATCHES = int(os.environ.get("API_MAX_STORED_BATCHES", "100"))
API_QUEUE_SIZE = int(os.environ.get("API_QUEUE_SIZE", "8"))
API_RESULT_TTL_SECONDS = int(os.environ.get("API_RESULT_TTL_SECONDS", "3600"))

API_ALLOWED_CONTENT_TYPES = tuple(
    value.strip()
    for value in os.environ.get(
        "API_ALLOWED_CONTENT_TYPES",
        "image/jpeg,image/png,image/webp",
    ).split(",")
)

API_ALLOWED_FORMATS = tuple(
    value.strip().upper()
    for value in os.environ.get(
        "API_ALLOWED_FORMATS",
        "JPEG,PNG,WEBP",
    ).split(",")
)

ANALYZE_DATA_DIR = _env_path(
    "ANALYZE_DATA_DIR",
    DATA_ROOT,
    PROJECT_ROOT,
)

ANALYZE_OUTPUT_DIR = _env_path(
    "ANALYZE_OUTPUT_DIR",
    PROJECT_ROOT / "reports" / "dataset",
    PROJECT_ROOT,
)

PREDICT_OUTPUT_PATH = _env_path(
    "PREDICT_OUTPUT_PATH",
    PROJECT_ROOT / "outputs" / "predictions" / "room.json",
    PROJECT_ROOT,
)

PREDICT_CONF = float(os.environ.get("PREDICT_CONF", str(INFERENCE_CONF)))

PREDICT_IMGSZ = int(os.environ.get("PREDICT_IMGSZ", str(INFERENCE_IMGSZ)))

PREDICT_DEVICE = os.environ.get("PREDICT_DEVICE", "cpu")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be one of: 1, true, yes, on, 0, false, no, off."
    )


PREDICT_SAVE_IMAGE = _env_bool("PREDICT_SAVE_IMAGE", False)

TRAIN_OUTPUT_DIR = _env_path(
    "TRAIN_OUTPUT_DIR",
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT,
)

TRAIN_INITIAL_WEIGHTS = os.environ.get(
    "TRAIN_INITIAL_WEIGHTS",
    "yolo11n.pt",
)

TRAIN_RUN_NAME = os.environ.get(
    "TRAIN_RUN_NAME",
    "yolo11n_hydra_smoke",
)

DATA_SPLIT_OUTPUT_DIR = _env_path(
    "DATA_SPLIT_OUTPUT_DIR",
    DATA_ROOT / "splits",
    PROJECT_ROOT,
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:8080",
)

MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "room-furniture-detector",
)

EVALUATION_CONF = float(os.environ.get("EVALUATION_CONF", "0.001"))

EVALUATION_IOU = float(os.environ.get("EVALUATION_IOU", "0.7"))

EVALUATION_SAVE_JSON = _env_bool(
    "EVALUATION_SAVE_JSON",
    False,
)

EVALUATION_PLOTS = _env_bool(
    "EVALUATION_PLOTS",
    True,
)

MODEL_PT_GDRIVE_ID = os.environ.get(
    "MODEL_PT_GDRIVE_ID",
    "1Iw_tVnTv9NOJbdX9nxg6K8yyVv7d9UQd",
)
MODEL_PT_SHA256 = os.environ.get(
    "MODEL_PT_SHA256",
    "2d4e71b15cb6eba55ca72574a08af3303b2e0fec60f0b223354feeea6b935250",
)

MODEL_ONNX_GDRIVE_ID = os.environ.get(
    "MODEL_ONNX_GDRIVE_ID",
    "1-Q-G1ozQKuYCrhb5jXx5M8vZ-t_JQaTq",
)
MODEL_ONNX_SHA256 = os.environ.get(
    "MODEL_ONNX_SHA256",
    "9c21a07a6116ae856adf372d0ada89550b387206fc951d172aae5c6b74eb1d9a",
)

MODEL_TENSORRT_GDRIVE_ID = os.environ.get(
    "MODEL_TENSORRT_GDRIVE_ID",
    "1O0tJFG5BpaeWcOlbPELxe4WU9QEYOwp2",
)
MODEL_TENSORRT_SHA256 = os.environ.get(
    "MODEL_TENSORRT_SHA256",
    "ab32f012bd53925e5f51f562c2f5e3f5db20a156cbb0cf11f205c07220847229",
)

MODEL_TENSORRT_METADATA_PATH = _env_path(
    "MODEL_TENSORRT_METADATA_PATH",
    MODEL_TENSORRT_PATH.with_suffix(".engine.json"),
    PROJECT_ROOT,
)
MODEL_TENSORRT_METADATA_GDRIVE_ID = os.environ.get(
    "MODEL_TENSORRT_METADATA_GDRIVE_ID",
    "1LqZLT_y-yi1-5fSW0Rp38Q9FGmzE7UdN",
)
MODEL_TENSORRT_METADATA_SHA256 = os.environ.get(
    "MODEL_TENSORRT_METADATA_SHA256",
    "9e5ca7cd18a2cff234c38da5d75b9fd3b3dc25968d5c12574b0e9c812a48f568",
)
