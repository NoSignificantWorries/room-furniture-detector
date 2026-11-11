from io import BytesIO
from threading import Event
from time import monotonic, sleep
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from room_furniture_detector import api


class FakeDetector:
    def __init__(self):
        self.backend_name = "fake"
        self.calls = 0
        self.fail = False
        self.empty = False
        self.started = Event()
        self.release = Event()
        self.release.set()

    def predict(self, image):
        self.calls += 1
        self.started.set()

        if not self.release.wait(timeout=10):
            raise RuntimeError("Test detector timed out.")

        if self.fail:
            raise RuntimeError("Internal secret model error.")

        width, height = image.size

        return {
            "width": width,
            "height": height,
            "boxes": (
                []
                if self.empty
                else [
                    {
                        "class_name": "sofa",
                        "conf": 0.9,
                        "xyxy_norm": [0.1, 0.2, 0.8, 0.9],
                    }
                ]
            ),
        }


@pytest.fixture
def detector():
    return FakeDetector()


@pytest.fixture
def client(monkeypatch, detector):
    monkeypatch.setattr(api, "Detector", lambda: detector)

    with TestClient(api.app) as test_client:
        try:
            yield test_client
        finally:
            # Prevent shutdown hanging if a test fails while blocked.
            detector.release.set()


def make_image(image_format="PNG", size=(100, 60)):
    buffer = BytesIO()

    with Image.new("RGB", size, "white") as image:
        image.save(buffer, format=image_format)

    return buffer.getvalue()


def create_batch(client, count=1, batch_id=None):
    data = make_image()
    fields = {} if batch_id is None else {"batch_id": batch_id}

    return client.post(
        "/batches",
        data=fields,
        files=[
            ("files", (f"room_{index}.png", data, "image/png"))
            for index in range(count)
        ],
    )


def wait_for_result(client, batch_id):
    deadline = monotonic() + 5

    while monotonic() < deadline:
        response = client.get(f"/batches/{batch_id}")
        assert response.status_code == 200

        result = response.json()
        if result["ready"]:
            return result

        sleep(0.01)

    pytest.fail(f"Batch did not finish: {batch_id}")


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_batch_generated_id_and_results(client, detector):
    response = create_batch(client, count=2)

    assert response.status_code == 200
    accepted = response.json()

    assert set(accepted) == {"batch_id", "image_ids", "status"}
    assert accepted["status"] == "accepted"
    assert UUID(accepted["batch_id"]).version == 4
    assert len(accepted["image_ids"]) == 2
    assert len(set(accepted["image_ids"])) == 2

    for image_id in accepted["image_ids"]:
        assert UUID(image_id).version == 4

    result = wait_for_result(client, accepted["batch_id"])

    assert result["batch_id"] == accepted["batch_id"]
    assert result["status"] == "done"
    assert result["ready"] is True
    assert result["error"] is None
    assert [item["image_id"] for item in result["results"]] == accepted[
        "image_ids"
    ]

    for item in result["results"]:
        assert item["width"] == 100
        assert item["height"] == 60
        assert item["boxes"] == [
            {
                "class_name": "sofa",
                "conf": 0.9,
                "xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            }
        ]

    assert detector.calls == 2


def test_supplied_batch_id(client):
    batch_id = str(uuid4())
    response = create_batch(client, batch_id=batch_id)

    assert response.status_code == 200
    assert response.json()["batch_id"] == batch_id

    result = wait_for_result(client, batch_id)
    assert result["status"] == "done"


def test_processing_state(client, detector):
    detector.release.clear()

    response = create_batch(client)
    assert response.status_code == 200
    batch_id = response.json()["batch_id"]

    try:
        assert detector.started.wait(timeout=5)

        response = client.get(f"/batches/{batch_id}")
        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "processing"
        assert result["ready"] is False
        assert result["results"] == []
        assert result["error"] is None
    finally:
        detector.release.set()

    assert wait_for_result(client, batch_id)["status"] == "done"


def test_unknown_batch(client):
    response = client.get(f"/batches/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Batch not found."}


def test_missing_files(client, detector):
    response = client.post("/batches")

    assert response.status_code == 400
    assert detector.calls == 0


def test_invalid_batch_id(client, detector):
    response = create_batch(client, batch_id="not-a-uuid")

    assert response.status_code == 400
    assert detector.calls == 0


def test_duplicate_batch_id(client):
    batch_id = str(uuid4())

    first = create_batch(client, batch_id=batch_id)
    second = create_batch(client, batch_id=batch_id)

    assert first.status_code == 200
    assert second.status_code == 409
    assert wait_for_result(client, batch_id)["status"] == "done"


@pytest.mark.parametrize(
    ("image_format", "content_type", "extension"),
    [
        ("PNG", "image/png", "png"),
        ("JPEG", "image/jpeg", "jpg"),
        ("WEBP", "image/webp", "webp"),
    ],
)
def test_supported_formats(client, image_format, content_type, extension):
    response = client.post(
        "/batches",
        files={
            "files": (
                f"room.{extension}",
                make_image(image_format),
                content_type,
            )
        },
    )

    assert response.status_code == 200
    result = wait_for_result(client, response.json()["batch_id"])
    assert result["status"] == "done"


@pytest.mark.parametrize(
    ("data", "content_type"),
    [
        (b"", "image/png"),
        (b"not an image", "image/png"),
        (b"some data", "image/gif"),
    ],
)
def test_invalid_upload(client, detector, data, content_type):
    response = client.post(
        "/batches",
        files={"files": ("room.png", data, content_type)},
    )

    assert response.status_code == 400
    assert detector.calls == 0


def test_content_is_checked_not_only_mime(client, detector):
    response = client.post(
        "/batches",
        files={"files": ("room.png", make_image("BMP"), "image/png")},
    )

    assert response.status_code == 400
    assert detector.calls == 0


def test_batch_rejected_if_one_file_invalid(client, detector):
    batch_id = str(uuid4())

    response = client.post(
        "/batches",
        data={"batch_id": batch_id},
        files=[
            ("files", ("valid.png", make_image(), "image/png")),
            ("files", ("invalid.png", b"invalid", "image/png")),
        ],
    )

    assert response.status_code == 400
    assert detector.calls == 0
    assert client.get(f"/batches/{batch_id}").status_code == 404


def test_file_count_limit(client, detector, monkeypatch):
    monkeypatch.setattr(api.config, "API_MAX_FILES", 1)

    response = create_batch(client, count=2)

    assert response.status_code == 400
    assert detector.calls == 0


def test_file_size_limit(client, detector, monkeypatch):
    monkeypatch.setattr(api.config, "API_MAX_UPLOAD_BYTES", 10)

    response = create_batch(client)

    assert response.status_code == 400
    assert detector.calls == 0


def test_batch_size_limit(client, detector, monkeypatch):
    image_size = len(make_image())
    monkeypatch.setattr(
        api.config,
        "API_MAX_BATCH_BYTES",
        image_size * 2 - 1,
    )

    response = create_batch(client, count=2)

    assert response.status_code == 400
    assert detector.calls == 0


def test_pixel_limit(client, detector, monkeypatch):
    monkeypatch.setattr(api.config, "API_MAX_IMAGE_PIXELS", 100)

    response = create_batch(client)

    assert response.status_code == 400
    assert detector.calls == 0


def test_failed_batch(client, detector):
    detector.fail = True

    response = create_batch(client)
    assert response.status_code == 200

    result = wait_for_result(client, response.json()["batch_id"])

    assert result["status"] == "failed"
    assert result["ready"] is True
    assert result["results"] == []
    assert result["error"] == "Batch inference failed."
    assert "secret" not in str(result)


def test_empty_detections_are_success(client, detector):
    detector.empty = True

    response = create_batch(client)
    assert response.status_code == 200

    result = wait_for_result(client, response.json()["batch_id"])

    assert result["status"] == "done"
    assert result["results"][0]["boxes"] == []


def test_repeated_polling_keeps_result(client):
    response = create_batch(client)
    assert response.status_code == 200
    batch_id = response.json()["batch_id"]

    first = wait_for_result(client, batch_id)
    second = client.get(f"/batches/{batch_id}")

    assert second.status_code == 200
    assert second.json() == first


def test_detector_initialized_once(monkeypatch):
    instances = []

    def factory():
        detector = FakeDetector()
        instances.append(detector)
        return detector

    monkeypatch.setattr(api, "Detector", factory)

    with TestClient(api.app) as client:
        for _ in range(2):
            response = create_batch(client)
            assert response.status_code == 200

            result = wait_for_result(
                client,
                response.json()["batch_id"],
            )
            assert result["status"] == "done"

        assert len(instances) == 1
        assert instances[0].calls == 2
