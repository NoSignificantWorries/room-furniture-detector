import asyncio
import logging
import warnings
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from io import BytesIO
from time import monotonic
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from room_furniture_detector import config
from room_furniture_detector.inference import Detector

logger = logging.getLogger(__name__)

Coordinate = Annotated[
    float,
    Field(ge=0, le=1, allow_inf_nan=False),
]


class Box(BaseModel):
    cls: int = Field(ge=0)
    conf: float = Field(ge=0, le=1, allow_inf_nan=False)
    xyxy_norm: tuple[Coordinate, Coordinate, Coordinate, Coordinate]


class ImageResult(BaseModel):
    image_id: UUID
    boxes: list[Box]
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class BatchAccepted(BaseModel):
    batch_id: UUID
    image_ids: list[UUID]
    status: Literal["accepted"] = "accepted"


class BatchStatus(BaseModel):
    batch_id: UUID
    status: Literal["processing", "done", "failed"]
    ready: bool
    results: list[ImageResult]
    error: str | None = None


class Error(BaseModel):
    detail: str


@dataclass
class BatchRecord:
    image_ids: list[UUID]
    status: Literal["processing", "done", "failed"] = "processing"
    results: list[ImageResult] = field(default_factory=list)
    error: str | None = None
    finished_at: float | None = None


def decode_image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(BytesIO(data)) as source:
                if source.format not in config.API_ALLOWED_FORMATS:
                    raise HTTPException(
                        status_code=400,
                        detail="Unsupported image format.",
                    )

                width, height = source.size

                if width * height > config.API_MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=400,
                        detail="Image dimensions exceed the configured limit.",
                    )

                if getattr(source, "is_animated", False):
                    raise HTTPException(
                        status_code=400,
                        detail="Animated images are not supported.",
                    )

                source.load()
                return source.copy()

    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Image dimensions are too large.",
        ) from exc

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode image.",
        ) from exc


def validate_image(data: bytes) -> None:
    image = decode_image(data)
    image.close()


def process_batch(
    detector: Detector,
    image_ids: list[UUID],
    payloads: list[bytes],
) -> list[ImageResult]:
    results = []

    for image_id, data in zip(image_ids, payloads, strict=True):
        image = decode_image(data)

        try:
            prediction = detector.predict(image)
        finally:
            image.close()

        results.append(
            ImageResult(
                image_id=image_id,
                **prediction,
            )
        )

    return results


def remove_expired_batches(app: FastAPI) -> None:
    now = monotonic()
    records = app.state.batches

    expired = [
        batch_id
        for batch_id, record in records.items()
        if record.finished_at is not None
        and now - record.finished_at >= config.API_RESULT_TTL_SECONDS
    ]

    for batch_id in expired:
        del records[batch_id]


async def batch_worker(app: FastAPI) -> None:
    queue = app.state.queue

    while True:
        batch_id, payloads = await queue.get()
        record = app.state.batches[batch_id]

        try:
            results = await run_in_threadpool(
                process_batch,
                app.state.detector,
                record.image_ids,
                payloads,
            )

            record.results = results
            record.status = "done"

        except Exception:
            logger.exception("Batch inference failed: %s", batch_id)
            record.results = []
            record.status = "failed"
            record.error = "Batch inference failed."

        finally:
            record.finished_at = monotonic()
            payloads.clear()
            queue.task_done()


def validate_settings() -> None:
    limits = {
        "API_MAX_FILES": config.API_MAX_FILES,
        "API_MAX_UPLOAD_BYTES": config.API_MAX_UPLOAD_BYTES,
        "API_MAX_BATCH_BYTES": config.API_MAX_BATCH_BYTES,
        "API_MAX_IMAGE_PIXELS": config.API_MAX_IMAGE_PIXELS,
        "API_MAX_STORED_BATCHES": config.API_MAX_STORED_BATCHES,
        "API_QUEUE_SIZE": config.API_QUEUE_SIZE,
        "API_RESULT_TTL_SECONDS": config.API_RESULT_TTL_SECONDS,
    }

    for name, value in limits.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if not config.API_ALLOWED_CONTENT_TYPES or not config.API_ALLOWED_FORMATS:
        raise ValueError("Allowed image types and formats must not be empty.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_settings()

    app.state.detector = await run_in_threadpool(Detector)
    app.state.batches = {}
    app.state.queue = asyncio.Queue(maxsize=config.API_QUEUE_SIZE)
    app.state.worker = asyncio.create_task(batch_worker(app))

    logger.info(
        "Model service started: backend=%s",
        app.state.detector.backend_name,
    )

    try:
        yield
    finally:
        # Finish accepted work before stopping the worker.
        await app.state.queue.join()
        app.state.worker.cancel()

        with suppress(asyncio.CancelledError):
            await app.state.worker

        app.state.batches.clear()
        del app.state.detector


app = FastAPI(
    title="Model Service API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    operation_id="getHealth",
    responses={503: {"description": "Service unavailable"}},
)
async def health(request: Request) -> dict[str, str]:
    worker = request.app.state.worker

    if worker.done():
        raise HTTPException(
            status_code=503,
            detail="Batch worker is unavailable.",
        )

    return {"status": "ok"}


@app.post(
    "/batches",
    response_model=BatchAccepted,
    operation_id="createBatch",
    responses={
        400: {"model": Error},
        409: {"model": Error},
        503: {"model": Error},
    },
)
async def create_batch(
    request: Request,
    files: Annotated[list[UploadFile] | None, File()] = None,
    batch_id: Annotated[str | None, Form()] = None,
) -> BatchAccepted:
    uploads = files or []

    try:
        if not uploads:
            raise HTTPException(
                status_code=400,
                detail="No files provided.",
            )

        if len(uploads) > config.API_MAX_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum files per batch: {config.API_MAX_FILES}.",
            )

        if batch_id is None:
            resolved_id = uuid4()
        else:
            try:
                resolved_id = UUID(batch_id)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid batch_id UUID.",
                ) from exc

        app = request.app
        remove_expired_batches(app)

        if resolved_id in app.state.batches:
            raise HTTPException(
                status_code=409,
                detail="Batch ID already exists.",
            )

        if app.state.worker.done():
            raise HTTPException(
                status_code=503,
                detail="Batch worker is unavailable.",
            )

        payloads = []
        total_bytes = 0

        for upload in uploads:
            if upload.content_type not in config.API_ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"unsupported content type: {upload.content_type}",
                )

            data = await upload.read(config.API_MAX_UPLOAD_BYTES + 1)

            if not data:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty.",
                )

            if len(data) > config.API_MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="File exceeds the configured size limit.",
                )

            total_bytes += len(data)

            if total_bytes > config.API_MAX_BATCH_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="Batch exceeds the configured size limit.",
                )

            await run_in_threadpool(validate_image, data)
            payloads.append(data)

        # Recheck after awaits: another request may have changed the store.
        remove_expired_batches(app)

        if resolved_id in app.state.batches:
            raise HTTPException(
                status_code=409,
                detail="Batch ID already exists.",
            )

        if (
            app.state.queue.full()
            or len(app.state.batches) >= config.API_MAX_STORED_BATCHES
        ):
            raise HTTPException(
                status_code=503,
                detail="Service capacity reached. Retry later.",
            )

        image_ids = [uuid4() for _ in payloads]

        # No await between inserting the record and enqueueing the job.
        app.state.batches[resolved_id] = BatchRecord(image_ids=image_ids)
        app.state.queue.put_nowait((resolved_id, payloads))

        return BatchAccepted(
            batch_id=resolved_id,
            image_ids=image_ids,
        )

    finally:
        for upload in uploads:
            await upload.close()


@app.get(
    "/batches/{batch_id}",
    response_model=BatchStatus,
    operation_id="getBatch",
    responses={404: {"model": Error}},
)
async def get_batch(
    request: Request,
    batch_id: UUID,
) -> BatchStatus:
    app = request.app
    remove_expired_batches(app)
    record = app.state.batches.get(batch_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Batch not found.",
        )

    return BatchStatus(
        batch_id=batch_id,
        status=record.status,
        ready=record.status != "processing",
        results=record.results,
        error=record.error,
    )
