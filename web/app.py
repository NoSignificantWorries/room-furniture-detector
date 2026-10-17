"""Web-сервис: раздаёт фронт и проксирует батчи в model-service (или отвечает mock-ом).

Запуск:  cd web && uvicorn app:app --reload --port 8000
"""

import io
import mimetypes
import random
import time
import uuid
from pathlib import Path
from typing import Annotated

import config
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

app = FastAPI(title="Room Furniture Detector — web")

# batch_id -> состояние mock-батча (только при USE_MOCK=true)
_mock_batches: dict[str, dict] = {}


# --- helpers ----------------------------------------------------------------
def _httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail="Для работы с model-service нужен httpx (добавьте в extra 'web').",
        ) from exc
    return httpx


async def _read_files(files: list[UploadFile]) -> list[dict]:
    if not files:
        raise HTTPException(status_code=400, detail="Нет файлов в запросе")
    if len(files) > config.BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"В батче максимум {config.BATCH_SIZE} изображений",
        )

    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    result = []
    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Неподдерживаемый формат: {upload.filename}",
            )
        data = await upload.read()
        if not data:
            raise HTTPException(
                status_code=400, detail=f"Пустой файл: {upload.filename}"
            )
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Файл больше {config.MAX_FILE_SIZE_MB} МБ: {upload.filename}",
            )
        content_type = (
            upload.content_type
            if upload.content_type and upload.content_type.startswith("image/")
            else mimetypes.types_map.get(ext, "application/octet-stream")
        )
        result.append(
            {
                "image_id": str(uuid.uuid4()),
                "ext": ext,
                "content_type": content_type,
                "data": data,
            }
        )
    return result


# --- mock -------------------------------------------------------------------
def _image_size(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception:
        return (1280, 720)


def _random_boxes() -> list[dict]:
    boxes = []
    for _ in range(random.randint(*config.MOCK_BOXES_RANGE)):
        x1 = random.uniform(0.02, 0.7)
        y1 = random.uniform(0.02, 0.7)
        x2 = min(0.98, x1 + random.uniform(0.1, 0.3))
        y2 = min(0.98, y1 + random.uniform(0.1, 0.3))
        boxes.append(
            {
                "cls": random.randrange(len(config.CLASS_NAMES)),
                "conf": round(random.uniform(0.35, 0.99), 2),
                "xyxy_norm": [round(v, 4) for v in (x1, y1, x2, y2)],
            }
        )
    return boxes


def _mock_create(batch_id: str, items: list[dict]) -> None:
    while len(_mock_batches) >= config.MOCK_MAX_STORED_BATCHES:
        _mock_batches.pop(next(iter(_mock_batches)))

    images = []
    for item in items:
        width, height = _image_size(item["data"])
        images.append(
            {"image_id": item["image_id"], "width": width, "height": height}
        )

    _mock_batches[batch_id] = {
        "ready_at": time.monotonic() + random.uniform(*config.MOCK_DELAY_SEC),
        "fail": random.random() < config.MOCK_FAIL_RATE,
        "images": images,
        "results": None,
    }


def _mock_get(batch_id: str) -> dict:
    batch = _mock_batches.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Батч не найден")
    if time.monotonic() < batch["ready_at"]:
        return {
            "batch_id": batch_id,
            "status": "processing",
            "ready": False,
            "results": [],
        }
    if batch["fail"]:
        return {
            "batch_id": batch_id,
            "status": "failed",
            "error": "mock: onnx runtime: CUDA OOM",
        }
    if batch["results"] is None:
        batch["results"] = [
            {**img, "boxes": _random_boxes()} for img in batch["images"]
        ]
    return {
        "batch_id": batch_id,
        "status": "done",
        "ready": True,
        "results": batch["results"],
    }


# --- API --------------------------------------------------------------------
@app.get("/api/config")
async def get_config():
    return config.public_config()


@app.get("/api/health")
async def health():
    """Состояние model-service. Ответ всегда 200, детали — в теле."""
    if not config.CHECK_HEALTH:
        return {"enabled": False, "status": "disabled"}
    if config.USE_MOCK:
        return {"enabled": True, "status": "ok", "mode": "mock"}

    httpx = _httpx()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{config.MODEL_SERVICE_URL}/health")
            resp.raise_for_status()
            ok = resp.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "enabled": True,
            "status": "unavailable",
            "mode": "proxy",
            "error": str(exc),
        }
    return {
        "enabled": True,
        "status": "ok" if ok else "unavailable",
        "mode": "proxy",
    }


@app.post("/api/batches")
async def create_batch(files: Annotated[list[UploadFile], File()]):
    """Принимает батч от браузера. batch_id и image_ids генерирует этот сервис."""
    items = await _read_files(files)
    batch_id = str(uuid.uuid4())

    if config.USE_MOCK:
        _mock_create(batch_id, items)
    else:
        httpx = _httpx()
        # Имя файла = image_id, чтобы model-service вернул результаты с теми же id.
        multipart = [
            (
                "files",
                (
                    f"{it['image_id']}{it['ext']}",
                    it["data"],
                    it["content_type"],
                ),
            )
            for it in items
        ]
        try:
            async with httpx.AsyncClient(
                timeout=config.REQUEST_TIMEOUT_SEC
            ) as client:
                resp = await client.post(
                    f"{config.MODEL_SERVICE_URL}/batches",
                    data={"batch_id": batch_id},
                    files=multipart,
                )
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"model-service: {exc}"
            ) from exc

    return {
        "batch_id": batch_id,
        "image_ids": [it["image_id"] for it in items],
        "status": "accepted",
    }


@app.get("/api/batches/{batch_id}")
async def get_batch(batch_id: str):
    if config.USE_MOCK:
        return _mock_get(batch_id)

    httpx = _httpx()
    try:
        async with httpx.AsyncClient(
            timeout=config.REQUEST_TIMEOUT_SEC
        ) as client:
            resp = await client.get(
                f"{config.MODEL_SERVICE_URL}/batches/{batch_id}"
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"model-service: {exc}"
        ) from exc

    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Батч не найден")
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502, detail=f"model-service: HTTP {resp.status_code}"
        )
    return resp.json()


# --- статика (только три файла, чтобы не раздавать app.py и node_modules) ----
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(config.BASE_DIR / "index.html")


@app.get("/app.js", include_in_schema=False)
async def app_js():
    return FileResponse(
        config.BASE_DIR / "app.js", media_type="text/javascript"
    )


@app.get("/style.css", include_in_schema=False)
async def style_css():
    return FileResponse(config.BASE_DIR / "style.css", media_type="text/css")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=True)
