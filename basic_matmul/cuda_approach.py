import time

from torch import Tensor, cuda, version
from torch import __version__ as torch_version

from matrix_types import Matrix


def require_cuda() -> str:
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. Check that the NVIDIA driver is "
            "installed, running, and visible in this environment."
        )

    cuda_device = "cuda"
    print(f"CUDA device: {cuda.get_device_name(cuda_device)}")
    print(f"PyTorch version: {torch_version}")
    print(f"CUDA runtime used by PyTorch: {version.cuda}")
    return cuda_device


def cuda_multiply(
    a_values: Matrix, b_values: Matrix, sample_size: int
) -> float:
    device = require_cuda()

    a = Tensor(a_values).to(device)
    b = Tensor(b_values).to(device)

    cuda.synchronize()
    start = time.perf_counter()
    c = a @ b
    cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"CUDA/PyTorch elapsed: {elapsed_ms:.2f} ms")
    print(c[:sample_size, :sample_size])
    return elapsed_ms
