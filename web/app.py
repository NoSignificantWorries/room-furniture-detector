from __future__ import annotations

import asyncio
import random
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import config
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Boxes Web Demo")

BATCHES: dict[str, dict[str, Any]] = {}


def _model_url(path: str) -> str:
    return f"{config.MODEL_SERVICE_URL.rstrip('/')}{path}"


def _mock_boxes() -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    count = random.randint(config.MOCK_MIN_BOXES, config.MOCK_MAX_BOXES)

    for _ in range(count):
        x1, x2 = sorted((random.random(), random.random()))
        y1, y2 = sorted((random.random(), random.random()))

        if x2 - x1 < 0.02:
            x2 = min(1.0, x1 + 0.05)
        if y2 - y1 < 0.02:
            y2 = min(1.0, y1 + 0.05)

        boxes.append(
            {
                "cls": random.randint(0, 5),
                "conf": round(random.uniform(0.30, 0.99), 3),
                "xyxy_norm": [
                    round(x1, 4),
                    round(y1, 4),
                    round(x2, 4),
                    round(y2, 4),
                ],
            }
        )

    return boxes


async def _mock_process(batch_id: str) -> None:
    await asyncio.sleep(config.MOCK_DELAY_SECONDS)

    record = BATCHES.get(batch_id)
    if record is None:
        return

    results = []
    for image_id in record["image_ids"]:
        width = random.choice([800, 1024, 1280, 1920])
        height = random.choice([600, 720, 768, 1080])

        results.append(
            {
                "image_id": image_id,
                "boxes": _mock_boxes(),
                "width": width,
                "height": height,
            }
        )

    record.update(
        {
            "status": "done",
            "ready": True,
            "results": results,
        }
    )


@app.get("/config")
async def get_config() -> dict[str, Any]:
    return {
        "batch_size": config.BATCH_SIZE,
        "poll_interval_ms": config.POLL_INTERVAL_MS,
        "use_mock": config.USE_MOCK,
        "check_health": config.CHECK_HEALTH,
        "mock_delay_seconds": config.MOCK_DELAY_SECONDS,
    }


@app.get("/health")
async def health():
    if not config.CHECK_HEALTH:
        return {
            "status": "disabled",
            "detail": "CHECK_HEALTH=false",
        }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(_model_url("/health"))

        response.raise_for_status()
        data = response.json()

        return {"status": data.get("status", "ok")}

    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "detail": f"model-service unavailable: {exc}",
            },
        )


@app.post("/batches")
async def create_batch(
    batch_id: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile], File()] = ...,
):
    if not files:
        raise HTTPException(status_code=400, detail="files are required")

    for upload in files:
        if upload.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported content type: {upload.content_type}",
            )

    batch_id = batch_id or str(uuid.uuid4())
    image_ids = [str(uuid.uuid4()) for _ in files]

    if config.USE_MOCK:
        BATCHES[batch_id] = {
            "batch_id": batch_id,
            "status": "processing",
            "ready": False,
            "results": [],
            "image_ids": image_ids,
            "error": None,
            "created_at": time.time(),
        }

        asyncio.create_task(_mock_process(batch_id))

        return {
            "batch_id": batch_id,
            "image_ids": image_ids,
            "status": "accepted",
        }

    files_payload = []
    for upload in files:
        content = await upload.read()
        files_payload.append(
            (
                "files",
                (
                    upload.filename,
                    content,
                    upload.content_type or "application/octet-stream",
                ),
            )
        )

    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
            response = await client.post(
                _model_url("/batches"),
                data={"batch_id": batch_id},
                files=files_payload,
            )

        response.raise_for_status()
        payload = response.json()
        payload.setdefault("batch_id", batch_id)
        payload.setdefault("image_ids", image_ids)

        return payload

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"model-service unavailable: {exc}",
        ) from exc


@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str):
    if config.USE_MOCK:
        record = BATCHES.get(batch_id)
        if record is None:
            raise HTTPException(status_code=404, detail="batch not found")

        payload = {
            "batch_id": record["batch_id"],
            "status": record["status"],
            "ready": record["ready"],
            "results": record["results"],
        }

        if record.get("error"):
            payload["error"] = record["error"]

        return payload

    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
            response = await client.get(_model_url(f"/batches/{batch_id}"))

        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="batch not found")

        response.raise_for_status()
        return response.json()

    except HTTPException:
        raise

    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "batch_id": batch_id,
                "status": "failed",
                "ready": False,
                "error": f"model-service unavailable: {exc}",
                "results": [],
            },
        )


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/app.js")
async def app_js():
    return FileResponse(
        BASE_DIR / "app.js",
        media_type="application/javascript",
    )


@app.get("/style.css")
async def style_css():
    return FileResponse(
        BASE_DIR / "style.css",
        media_type="text/css",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
