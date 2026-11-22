from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageOps

from room_furniture_detector import config


class OnnxBackend:
    def __init__(self) -> None:
        import onnxruntime as ort

        path = config.MODEL_ONNX_PATH
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        self.session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one model input and one output.")

        self.input = inputs[0]
        self.output = outputs[0]

        expected = [
            1,
            3,
            config.INFERENCE_IMGSZ,
            config.INFERENCE_IMGSZ,
        ]

        if self.input.shape != expected:
            raise ValueError(
                f"Unexpected ONNX input: {self.input.shape}; "
                f"expected {expected}"
            )

        if self.input.type != "tensor(float)":
            raise TypeError("Expected float32 ONNX input.")

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(
            [self.output.name],
            {self.input.name: tensor},
        )[0]


class TensorRTBackend:
    def __init__(self) -> None:
        import tensorrt_lean as trt
        from cuda.bindings import runtime as cudart

        self.trt = trt
        self.cudart = cudart
        self.device_id = config.TENSORRT_DEVICE

        (device_count,) = self._cuda_check(
            cudart.cudaGetDeviceCount(),
            "cudaGetDeviceCount",
        )

        if device_count == 0:
            raise RuntimeError("TensorRT requires an available CUDA GPU.")

        if not 0 <= self.device_id < device_count:
            raise ValueError("Invalid TENSORRT_DEVICE.")

        path = config.MODEL_TENSORRT_PATH
        if not path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")

        self._cuda_check(
            cudart.cudaSetDevice(self.device_id),
            "cudaSetDevice",
        )

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.runtime.engine_host_code_allowed = True
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())

        if self.engine is None:
            raise RuntimeError("Cannot deserialize TensorRT engine.")

        inputs = []
        outputs = []

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one model input and one output.")

        self.input_name = inputs[0]
        self.output_name = outputs[0]

        expected = (
            1,
            3,
            config.INFERENCE_IMGSZ,
            config.INFERENCE_IMGSZ,
        )

        if tuple(self.engine.get_tensor_shape(self.input_name)) != expected:
            raise ValueError("Unexpected TensorRT input shape.")

        for name in (self.input_name, self.output_name):
            if self.engine.get_tensor_dtype(name) != trt.float32:
                raise TypeError(f"Expected float32 tensor: {name}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT context.")

        self.output_shape = tuple(
            self.context.get_tensor_shape(self.output_name)
        )

        if any(dimension <= 0 for dimension in self.output_shape):
            raise ValueError("Only static TensorRT outputs are supported.")

        self.input_shape = expected

    def _cuda_check(self, result: tuple, operation: str) -> tuple:
        error, *values = result

        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed: {error}")

        return tuple(values)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        from contextlib import ExitStack

        cudart = self.cudart

        if tensor.shape != self.input_shape:
            raise ValueError(
                f"Unexpected TensorRT input shape: {tensor.shape}; "
                f"expected {self.input_shape}"
            )

        if tensor.dtype != np.float32:
            raise TypeError("Expected float32 TensorRT input.")

        host_input = np.ascontiguousarray(tensor)
        host_output = np.empty(self.output_shape, dtype=np.float32)

        self._cuda_check(
            cudart.cudaSetDevice(self.device_id),
            "cudaSetDevice",
        )

        with ExitStack() as resources:
            (stream,) = self._cuda_check(
                cudart.cudaStreamCreate(),
                "cudaStreamCreate",
            )
            resources.callback(
                lambda: self._cuda_check(
                    cudart.cudaStreamDestroy(stream),
                    "cudaStreamDestroy",
                )
            )

            (device_input,) = self._cuda_check(
                cudart.cudaMalloc(host_input.nbytes),
                "cudaMalloc(input)",
            )
            resources.callback(
                lambda: self._cuda_check(
                    cudart.cudaFree(device_input),
                    "cudaFree(input)",
                )
            )

            (device_output,) = self._cuda_check(
                cudart.cudaMalloc(host_output.nbytes),
                "cudaMalloc(output)",
            )
            resources.callback(
                lambda: self._cuda_check(
                    cudart.cudaFree(device_output),
                    "cudaFree(output)",
                )
            )

            try:
                if not self.context.set_tensor_address(
                    self.input_name,
                    int(device_input),
                ):
                    raise RuntimeError("Cannot bind TensorRT input.")

                if not self.context.set_tensor_address(
                    self.output_name,
                    int(device_output),
                ):
                    raise RuntimeError("Cannot bind TensorRT output.")

                self._cuda_check(
                    cudart.cudaMemcpyAsync(
                        device_input,
                        host_input.ctypes.data,
                        host_input.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    ),
                    "cudaMemcpyAsync(input)",
                )

                if not self.context.execute_async_v3(stream_handle=int(stream)):
                    raise RuntimeError("TensorRT inference failed.")

                self._cuda_check(
                    cudart.cudaMemcpyAsync(
                        host_output.ctypes.data,
                        device_output,
                        host_output.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    "cudaMemcpyAsync(output)",
                )
            finally:
                self._cuda_check(
                    cudart.cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize",
                )

        return host_output


class TritonBackend:
    def __init__(self) -> None:
        raise NotImplementedError("Triton backend is not implemented yet.")


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep = []

    while order.size:
        index = int(order[0])
        keep.append(index)

        if order.size == 1:
            break

        remaining = order[1:]

        left_top = np.maximum(
            boxes[index, :2],
            boxes[remaining, :2],
        )
        right_bottom = np.minimum(
            boxes[index, 2:],
            boxes[remaining, 2:],
        )

        intersection = np.maximum(right_bottom - left_top, 0).prod(axis=1)

        area = np.maximum(boxes[index, 2:] - boxes[index, :2], 0).prod()

        other_areas = np.maximum(
            boxes[remaining, 2:] - boxes[remaining, :2], 0
        ).prod(axis=1)

        union = area + other_areas - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = remaining[iou <= threshold]

    return keep


class Detector:
    def __init__(self, backend: str | None = None) -> None:
        if not 0 <= config.INFERENCE_CONF <= 1:
            raise ValueError("INFERENCE_CONF must be between 0 and 1.")

        if not 0 <= config.INFERENCE_IOU <= 1:
            raise ValueError("INFERENCE_IOU must be between 0 and 1.")

        if (
            config.INFERENCE_IMGSZ <= 0
            or config.INFERENCE_MAX_DET <= 0
            or config.INFERENCE_MAX_NMS <= 0
        ):
            raise ValueError("Inference sizes and limits must be positive.")

        if not config.MODEL_CLASS_NAMES or any(
            not name for name in config.MODEL_CLASS_NAMES
        ):
            raise ValueError("MODEL_CLASS_NAMES must contain class names.")

        factories = {
            "onnx": OnnxBackend,
            "tensorrt": TensorRTBackend,
            "triton": TritonBackend,
        }

        self.backend_name = backend or config.MODEL_BACKEND

        if self.backend_name not in factories:
            raise ValueError(f"Unsupported backend: {self.backend_name}")

        self.backend = factories[self.backend_name]()
        self.lock = Lock()

    def predict(self, image: Image.Image) -> dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        size = config.INFERENCE_IMGSZ

        ratio = min(size / width, size / height)
        resized_width = max(1, round(width * ratio))
        resized_height = max(1, round(height * ratio))

        left = round((size - resized_width) / 2 - 0.1)
        top = round((size - resized_height) / 2 - 0.1)

        resized = cv2.resize(
            np.asarray(image),
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[
            top : top + resized_height,
            left : left + resized_width,
        ] = resized

        tensor = np.ascontiguousarray(
            canvas.transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        tensor /= 255.0

        with self.lock:
            output = self.backend.run(tensor)

        expected_channels = 4 + len(config.MODEL_CLASS_NAMES)
        if (
            output.ndim != 3
            or output.shape[0] != 1
            or output.shape[1] != expected_channels
        ):
            raise ValueError(f"Unexpected model output shape: {output.shape}")

        if not np.isfinite(output).all():
            raise ValueError("Model output contains NaN or infinity.")

        candidates = output[0].T
        classes = candidates[:, 4:].argmax(axis=1)
        scores = candidates[:, 4:].max(axis=1)
        selected = scores >= config.INFERENCE_CONF

        candidates = candidates[selected]
        classes = classes[selected]
        scores = scores[selected]

        order = np.argsort(-scores, kind="stable")[: config.INFERENCE_MAX_NMS]
        candidates = candidates[order]
        classes = classes[order]
        scores = scores[order]

        boxes = np.empty((len(candidates), 4), dtype=np.float32)
        boxes[:, :2] = candidates[:, :2] - candidates[:, 2:4] / 2
        boxes[:, 2:] = candidates[:, :2] + candidates[:, 2:4] / 2

        valid = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
        boxes, scores, classes = (
            boxes[valid],
            scores[valid],
            classes[valid],
        )

        keep = []

        for class_id in np.unique(classes):
            indexes = np.flatnonzero(classes == class_id)
            local_keep = _nms(
                boxes[indexes],
                scores[indexes],
                config.INFERENCE_IOU,
            )
            keep.extend(indexes[local_keep].tolist())

        keep.sort(key=lambda index: -float(scores[index]))
        keep = keep[: config.INFERENCE_MAX_DET]

        detections = []

        for index in keep:
            box = boxes[index].copy()
            box[[0, 2]] = (box[[0, 2]] - left) / ratio
            box[[1, 3]] = (box[[1, 3]] - top) / ratio

            box[[0, 2]] = np.clip(box[[0, 2]], 0, width) / width
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height) / height

            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            class_id = int(classes[index])

            detections.append(
                {
                    "class_name": config.MODEL_CLASS_NAMES[class_id],
                    "conf": float(np.clip(scores[index], 0, 1)),
                    "xyxy_norm": box.tolist(),
                }
            )

        return {
            "width": width,
            "height": height,
            "boxes": detections,
        }
