from pathlib import Path


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results"
