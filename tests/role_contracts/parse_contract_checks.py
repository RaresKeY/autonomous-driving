from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import load_module_from_path


TRAVERSAL_FN_CANDIDATES = (
    "iter_kitti_samples",
    "iter_kitti_pairs",
    "iter_dataset",
    "scan_kitti_dataset",
    "list_kitti_samples",
    "collect_kitti_samples",
)


def _load_parse_module(src_dir):
    return load_module_from_path(src_dir / "parse_kitti_label.py")


def _pick_traversal_fn(module):
    for name in TRAVERSAL_FN_CANDIDATES:
        fn = getattr(module, name, None)
        if callable(fn):
            return name, fn
    return None, None


def _invoke_traversal(fn, kitti_root: Path, limit: int | None = None):
    sig = inspect.signature(fn)
    kwargs = {}
    partial_supported = False
    sample_ids = ["000000", "000001"]
    for name, param in sig.parameters.items():
        if name in {"kitti_dir", "dataset_dir", "dataset_root", "root", "base_dir", "path"}:
            kwargs[name] = kitti_root
        elif name in {"image_dir", "images_dir", "img_dir"}:
            kwargs[name] = kitti_root / "training" / "image_2"
        elif name in {"label_dir", "labels_dir", "labels_path", "ann_dir"}:
            kwargs[name] = kitti_root / "training" / "label_2"
        elif name in {"limit", "max_samples", "max_items", "n"} and limit is not None:
            kwargs[name] = limit
            partial_supported = True
        elif name in {"sample_ids", "ids", "selected_ids", "subset_ids"} and limit is not None:
            kwargs[name] = sample_ids[:limit]
            partial_supported = True
        elif name in {"subset"} and limit is not None:
            kwargs[name] = limit
            partial_supported = True
        elif name in {"partial"} and limit is not None:
            kwargs[name] = True
            partial_supported = True
        elif param.default is inspect._empty:
            raise AssertionError(f"Unsupported required parameter `{name}` in traversal function")
    result = fn(**kwargs)
    return list(result) if not isinstance(result, list) else result, partial_supported


def _normalize_ids(items):
    ids = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            ids.append(str(item.get("sample_id") or item.get("id") or item.get("image_id")))
        elif isinstance(item, (tuple, list)) and item:
            ids.append(item[0] if isinstance(item[0], str) else Path(item[0]).stem)
        else:
            raise AssertionError(f"Unsupported traversal item: {item!r}")
    return ids


def test_parse_module_exposes_traversal_contract(src_dir):
    module = _load_parse_module(src_dir)
    assert callable(getattr(module, "parse_args", None))
    assert callable(getattr(module, "main", None))
    _, fn = _pick_traversal_fn(module)
    assert fn is not None, "`src/parse_kitti_label.py` must expose traversal function"


def test_parse_traversal_can_scan_complete_dataset(src_dir, sample_kitti_tree):
    module = _load_parse_module(src_dir)
    _, fn = _pick_traversal_fn(module)
    items, _ = _invoke_traversal(fn, sample_kitti_tree["root"])
    ids = _normalize_ids(items)
    assert ids[:3] == sample_kitti_tree["sample_ids"]
    assert "999999" not in ids


def test_parse_traversal_supports_partial_mode(src_dir, sample_kitti_tree):
    module = _load_parse_module(src_dir)
    _, fn = _pick_traversal_fn(module)
    items, partial_supported = _invoke_traversal(fn, sample_kitti_tree["root"], limit=2)
    assert partial_supported
    assert len(_normalize_ids(items)) <= 2


def test_parse_main_smoke_with_mocked_traversal(monkeypatch, src_dir, tmp_path):
    module = _load_parse_module(src_dir)
    traversal_name, _ = _pick_traversal_fn(module)
    fake_args = SimpleNamespace(dataset_dir=tmp_path, kitti_dir=tmp_path, limit=2, subset=2)
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)
    monkeypatch.setattr(module, traversal_name, lambda *a, **k: [("000000", tmp_path / "a.png", tmp_path / "a.txt")])
    assert module.main() in (0, None)


def test_parse_main_returns_nonzero_on_traversal_failure(monkeypatch, src_dir, tmp_path):
    module = _load_parse_module(src_dir)
    traversal_name, _ = _pick_traversal_fn(module)
    fake_args = SimpleNamespace(dataset_dir=tmp_path, kitti_dir=tmp_path, limit=2, subset=2)
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)
    monkeypatch.setattr(module, traversal_name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert module.main() not in (0, None)
