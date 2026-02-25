from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

ROLE_TARGETS = {
    "Paul": SRC_DIR / "inference.py",
    "Anca": SRC_DIR / "training.py",
    "Mihaela": SRC_DIR / "download.py",
    "Claudia": SRC_DIR / "parse_kitti_label.py",
}

ROLE_TEST_FILES = {
    "Paul": PROJECT_ROOT / "tests" / "inference_tests.py",
    "Anca": PROJECT_ROOT / "tests" / "training_tests.py",
    "Mihaela": PROJECT_ROOT / "tests" / "download_tests.py",
    "Claudia": PROJECT_ROOT / "tests" / "parse_tests.py",
}


def load_module_from_path(module_path: Path):
    assert module_path.exists(), f"Missing expected role file: {module_path}"

    module_name = f"contract_{module_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None, f"Unable to load module spec for {module_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def src_dir() -> Path:
    return SRC_DIR


@pytest.fixture
def role_targets():
    return ROLE_TARGETS


@pytest.fixture
def role_test_files():
    return ROLE_TEST_FILES


@pytest.fixture
def sample_kitti_tree(tmp_path: Path):
    """Create a minimal KITTI-like training tree for traversal tests."""
    image_dir = tmp_path / "training" / "image_2"
    label_dir = tmp_path / "training" / "label_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = ["000000", "000001", "000002"]
    for sample_id in sample_ids:
        (image_dir / f"{sample_id}.png").write_bytes(b"fake-image")
        (label_dir / f"{sample_id}.txt").write_text("Car 0 0 0 1 2 3 4 0 0 0 0 0 0 0\n", encoding="utf-8")

    # An unmatched image should be ignored by traversal scripts that require pairs.
    (image_dir / "999999.png").write_bytes(b"orphan-image")

    return {
        "root": tmp_path,
        "image_dir": image_dir,
        "label_dir": label_dir,
        "sample_ids": sample_ids,
    }
