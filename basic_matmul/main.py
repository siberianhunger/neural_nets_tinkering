from cuda_approach import cuda_multiply
from matrix_types import Matrix
from numpy_approach import numpy_multiply
from raw_python_approach import raw_python_multiply

MATRIX_SIZE = 1024
SAMPLE_SIZE = 300


def make_matrices(size: int) -> tuple[Matrix, Matrix]:
    a = [
        [((row * 31 + col * 17) % 97) / 97 for col in range(size)]
        for row in range(size)
    ]
    b = [
        [((row * 13 + col * 19) % 89) / 89 for col in range(size)]
        for row in range(size)
    ]
    return a, b


def main() -> None:
    print(f"Multiplying ({MATRIX_SIZE}x{MATRIX_SIZE}) @ ({MATRIX_SIZE}x{MATRIX_SIZE})")
    print("The same input values are used for all three methods.\n")

    a, b = make_matrices(MATRIX_SIZE)

    cuda_elapsed = cuda_multiply(a, b, SAMPLE_SIZE)
    print()
    numpy_elapsed = numpy_multiply(a, b, SAMPLE_SIZE)
    print()
    raw_python_elapsed = raw_python_multiply(a, b, SAMPLE_SIZE)

    print("\nSummary:")
    print(f"CUDA/PyTorch: {cuda_elapsed:.2f} ms")
    print(f"NumPy:        {numpy_elapsed:.2f} ms")
    print(f"Raw Python:  {raw_python_elapsed:.2f} ms")


if __name__ == "__main__":
    main()
