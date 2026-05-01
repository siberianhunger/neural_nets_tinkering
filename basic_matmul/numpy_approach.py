import time

from numpy import array, matmul

from matrix_types import Matrix


def numpy_multiply(
    a_values: Matrix, b_values: Matrix, sample_size: int
) -> float:
    a = array(a_values)
    b = array(b_values)

    start = time.perf_counter()
    c = matmul(a, b)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"NumPy elapsed: {elapsed_ms:.2f} ms")
    print(c[:sample_size, :sample_size])
    return elapsed_ms
