import os
import subprocess
import sys
from pathlib import Path

# Default config for CI / editorial smoke test (fast on CPU).
CONFIG = "configs/quick_test.yaml"


def _thread_limited_env() -> dict:
    """Cap BLAS / OpenMP threads so quick-test stays responsive on shared CPUs."""
    base = {**os.environ, "PYTHONUTF8": "1"}
    caps = {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "TORCH_NUM_THREADS": "2",
        "PGSSM_QUICK_SYNTHETIC_N": "420",
    }
    for k, v in caps.items():
        base[k] = v
    return base


def main():
    root = Path(__file__).resolve().parent.parent
    print("Running PG-SSM quick test (lightweight config)...", flush=True)
    env = _thread_limited_env()
    subprocess.run(
        [sys.executable, str(root / "scripts" / "generate_synthetic_demo.py")],
        cwd=str(root),
        check=True,
        env=env,
    )
    subprocess.run(
        [sys.executable, str(root / "src" / "train.py"), "--config", CONFIG],
        cwd=str(root),
        check=True,
        env=env,
    )
    subprocess.run(
        [sys.executable, str(root / "src" / "evaluate.py"), "--config", CONFIG],
        cwd=str(root),
        check=True,
        env=env,
    )
    print("Quick test completed successfully.", flush=True)


if __name__ == "__main__":
    main()
