from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class PackageCheck:
    import_name: str
    dist_name: str | None = None
    required: bool = True


PACKAGE_CHECKS = [
    PackageCheck("pydantic"),
    PackageCheck("pandas"),
    PackageCheck("polars"),
    PackageCheck("pyarrow"),
    PackageCheck("datasets"),
    PackageCheck("neo4j"),
    PackageCheck("graphdatascience"),
    PackageCheck("sentence_transformers", "sentence-transformers"),
    PackageCheck("torch"),
    PackageCheck("langchain"),
    PackageCheck("langchain_community", "langchain-community"),
    PackageCheck("langchain_neo4j", "langchain-neo4j"),
    PackageCheck("langchain_openai", "langchain-openai"),
    PackageCheck("langgraph"),
    PackageCheck("fastapi"),
    PackageCheck("uvicorn"),
    PackageCheck("tqdm"),
    PackageCheck("httpx"),
    PackageCheck("pytest"),
    PackageCheck("yaml", "pyyaml"),
    PackageCheck("torchinfo"),
    PackageCheck("lightgbm"),
    PackageCheck("sklearn", "scikit-learn"),
    PackageCheck("matplotlib"),
    PackageCheck("catboost"),
]


def _dist_version(check: PackageCheck) -> str:
    dist_name = check.dist_name or check.import_name
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _print_runtime() -> None:
    print(f"python: {sys.version}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is None:
        print("gil_enabled: unavailable")
    else:
        print(f"gil_enabled: {is_gil_enabled()}")


def _check_imports() -> list[str]:
    failures: list[str] = []
    for check in PACKAGE_CHECKS:
        try:
            importlib.import_module(check.import_name)
        except Exception as exc:
            status = "required" if check.required else "optional"
            failures.append(f"{check.import_name} ({status}): {type(exc).__name__}: {exc}")
            print(f"[FAIL] {check.import_name}: {type(exc).__name__}: {exc}")
            continue
        print(f"[OK] {check.import_name}: {_dist_version(check)}")
    return failures


def _check_torch_cuda() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"torch_cuda: unavailable ({type(exc).__name__}: {exc})")
        return

    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")
    print(f"torch_cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.ones((512, 512), device=device)
        result = torch.mm(tensor, tensor).sum().item()
        print(f"torch_cuda_device: {torch.cuda.get_device_name(0)}")
        print(f"torch_cuda_smoke_sum: {result:.1f}")
        print(f"torch_cuda_peak_mb: {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")


def main() -> int:
    _print_runtime()
    print()
    failures = _check_imports()
    print()
    _check_torch_cuda()

    if failures:
        print()
        print("environment_status: FAIL")
        return 1

    print()
    print("environment_status: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
