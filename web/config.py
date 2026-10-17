"""Настройки web-части. Любое значение можно переопределить переменной окружения."""

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_pair(
    name: str, default: tuple[float, float]
) -> tuple[float, float]:
    """Формат переменной: "1.0,3.0"."""
    value = os.getenv(name)
    if not value:
        return default
    low, high = (float(part) for part in value.split(","))
    return (min(low, high), max(low, high))


BASE_DIR = Path(__file__).resolve().parent

# --- Сервер -----------------------------------------------------------------
HOST = os.getenv("WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("WEB_PORT", "8000"))

# --- Режимы -----------------------------------------------------------------
# USE_MOCK=true: к model-service не обращаемся, боксы генерируются случайно.
USE_MOCK = _env_bool("USE_MOCK", True)
# CHECK_HEALTH=true: проверять GET /health у model-service (пока сервиса нет — false).
CHECK_HEALTH = _env_bool("CHECK_HEALTH", False)

# --- model-service ----------------------------------------------------------
MODEL_SERVICE_URL = os.getenv(
    "MODEL_SERVICE_URL", "http://localhost:8001"
).rstrip("/")
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "30"))

# --- Батчи и опрос (отдаются в браузер через /api/config) -------------------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))  # картинок в одном батче
POLL_INTERVAL_MS = int(
    os.getenv("POLL_INTERVAL_MS", "1000")
)  # период опроса результата
POLL_TIMEOUT_SEC = int(
    os.getenv("POLL_TIMEOUT_SEC", "300")
)  # сколько ждём один батч
HEALTH_INTERVAL_MS = int(
    os.getenv("HEALTH_INTERVAL_MS", "5000")
)  # период проверки /health

# --- Загрузка файлов --------------------------------------------------------
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

# --- Mock-режим -------------------------------------------------------------
MOCK_DELAY_SEC = _env_float_pair(
    "MOCK_DELAY_SEC", (1.0, 3.0)
)  # задержка ответа на батч
MOCK_BOXES_RANGE = (1, 6)  # сколько боксов на картинку
MOCK_FAIL_RATE = float(
    os.getenv("MOCK_FAIL_RATE", "0")
)  # доля упавших батчей (0..1)
MOCK_MAX_STORED_BATCHES = 500

# --- Классы -----------------------------------------------------------------
# Индекс в списке = поле "cls" в ответе сервиса. Заменить на реальные классы модели.
CLASS_NAMES = [
    "bed",
    "sofa",
    "chair",
    "table",
    "wardrobe",
    "tv",
    "desk",
    "shelf",
]


def public_config() -> dict:
    """Часть настроек, нужная браузеру."""
    return {
        "use_mock": USE_MOCK,
        "check_health": CHECK_HEALTH,
        "batch_size": BATCH_SIZE,
        "poll_interval_ms": POLL_INTERVAL_MS,
        "poll_timeout_sec": POLL_TIMEOUT_SEC,
        "health_interval_ms": HEALTH_INTERVAL_MS,
        "allowed_extensions": ALLOWED_EXTENSIONS,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "class_names": CLASS_NAMES,
    }
