import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


USE_MOCK = _env_bool("USE_MOCK", True)
CHECK_HEALTH = _env_bool("CHECK_HEALTH", False)

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:8001")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
POLL_INTERVAL_MS = int(os.getenv("POLL_INTERVAL_MS", "1500"))

MOCK_DELAY_SECONDS = float(os.getenv("MOCK_DELAY_SECONDS", "3.0"))
MOCK_MIN_BOXES = int(os.getenv("MOCK_MIN_BOXES", "0"))
MOCK_MAX_BOXES = int(os.getenv("MOCK_MAX_BOXES", "5"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30.0"))
