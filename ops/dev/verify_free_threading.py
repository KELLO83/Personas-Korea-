from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
import sysconfig
import time


def _sum_squares(start: int, stop: int) -> int:
    total = 0
    for value in range(start, stop):
        total += value * value
    return total


def _run_sequential(total_items: int, chunks: int) -> tuple[int, float]:
    chunk_size = math.ceil(total_items / chunks)
    started = time.perf_counter()
    total = 0
    for chunk_index in range(chunks):
        start = chunk_index * chunk_size
        stop = min(start + chunk_size, total_items)
        total += _sum_squares(start, stop)
    return total, time.perf_counter() - started


def _run_threaded(total_items: int, workers: int) -> tuple[int, float]:
    chunk_size = math.ceil(total_items / workers)
    ranges = [
        (worker_index * chunk_size, min((worker_index + 1) * chunk_size, total_items))
        for worker_index in range(workers)
    ]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        total = sum(executor.map(lambda bounds: _sum_squares(*bounds), ranges))
    return total, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=80_000_000)
    parser.add_argument("--workers", type=int, default=min(max((os.cpu_count() or 4) - 4, 1), 18))
    args = parser.parse_args()

    print(f"python: {sys.version}")
    print(f"executable: {sys.executable}")
    print(f"gil_enabled: {sys._is_gil_enabled()}")
    print(f"Py_GIL_DISABLED: {sysconfig.get_config_var('Py_GIL_DISABLED')}")
    print(f"SOABI: {sysconfig.get_config_var('SOABI')}")
    print(f"os_cpu_count: {os.cpu_count()}")
    print(f"workers: {args.workers}")
    print(f"items: {args.items}")

    sequential_total, sequential_seconds = _run_sequential(args.items, args.workers)
    threaded_total, threaded_seconds = _run_threaded(args.items, args.workers)

    if sequential_total != threaded_total:
        print("result_match: False")
        return 1

    speedup = sequential_seconds / threaded_seconds if threaded_seconds else float("inf")
    print(f"sequential_seconds: {sequential_seconds:.3f}")
    print(f"threaded_seconds: {threaded_seconds:.3f}")
    print(f"speedup: {speedup:.2f}x")
    print("result_match: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
