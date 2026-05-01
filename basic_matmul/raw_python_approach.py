import time

from matrix_types import Matrix


def raw_python_multiply(a: Matrix, b: Matrix, sample_size: int) -> float:
    b_columns = [[row[col] for row in b] for col in range(len(b[0]))]

    start = time.perf_counter()
    c = [
        [
            sum(left * right for left, right in zip(a_row, b_col, strict=True))
            for b_col in b_columns
        ]
        for a_row in a
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Raw Python elapsed: {elapsed_ms:.2f} ms")
    for row in c[:sample_size]:
        print(row[:sample_size])
    return elapsed_ms
