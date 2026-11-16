import hashlib
import json
import math
from pathlib import Path

from room_furniture_detector import config


def export_tensorrt() -> Path:
    import tensorrt as trt
    import torch

    source = config.MODEL_ONNX_PATH
    destination = config.MODEL_TENSORRT_PATH
    metadata_path = config.MODEL_TENSORRT_METADATA_PATH

    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")

    if destination.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Export destination already exists: {destination}. "
            "Choose another MODEL_TENSORRT_PATH or move existing files."
        )

    workspace_gb = config.TENSORRT_WORKSPACE_GB
    if not math.isfinite(workspace_gb) or workspace_gb <= 0:
        raise ValueError("TENSORRT_WORKSPACE_GB must be positive and finite.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = config.TENSORRT_DEVICE
    if device < 0 or device >= torch.cuda.device_count():
        raise ValueError(f"Invalid TENSORRT_DEVICE: {device}")

    torch.cuda.set_device(device)
    torch.cuda.init()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    if builder is None:
        raise RuntimeError("Failed to create TensorRT Builder.")

    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(source)):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"Failed to parse ONNX:\n{errors}")

    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(tensor.shape)

        if any(dimension <= 0 for dimension in shape):
            raise ValueError(
                f"Only static ONNX inputs are supported: {tensor.name} {shape}"
            )

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),
    )

    print(f"Building TensorRT engine from: {source}")
    serialized = builder.build_serialized_network(
        network,
        builder_config,
    )

    if serialized is None:
        raise RuntimeError(
            "TensorRT engine build failed. See TensorRT logs above."
        )

    engine_bytes = bytes(serialized)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    if engine is None:
        raise RuntimeError("Failed to deserialize the built engine.")

    tensors = []

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)

        item = {
            "name": name,
            "mode": str(engine.get_tensor_mode(name)),
            "shape": list(engine.get_tensor_shape(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
        }
        tensors.append(item)
        print(item)

    metadata = {
        "source_onnx": source.name,
        "source_onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(engine_bytes).hexdigest(),
        "tensorrt_version": trt.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "workspace_gb": workspace_gb,
        "tensors": tensors,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".engine.part")

    try:
        temporary.write_bytes(engine_bytes)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"TensorRT engine saved to: {destination}")
    print(f"Build metadata saved to: {metadata_path}")
    return destination


def main() -> None:
    export_tensorrt()


if __name__ == "__main__":
    main()
