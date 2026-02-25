from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import load_module_from_path


TRAVERSAL_FN_CANDIDATES = (
    "iter_kitti_samples",
    "iter_kitti_pairs",
    "iter_dataset",
    "scan_kitti_dataset",
    "list_kitti_samples",
    "collect_kitti_samples",
)


def _load_download_module(src_dir):
    return load_module_from_path(src_dir / "download.py")


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
            raise AssertionError(
                f"Traversal function has unsupported required parameter `{name}`; "
                "use common path/limit names so tests can call it."
            )

    result = fn(**kwargs)
    items = list(result) if not isinstance(result, list) else result
    return items, partial_supported


def _normalize_ids(items):
    ids = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
            continue
        if isinstance(item, dict):
            for key in ("sample_id", "id", "image_id", "name"):
                if key in item:
                    ids.append(str(item[key]))
                    break
            else:
                raise AssertionError(f"Cannot extract sample id from dict item: {item}")
            continue
        if isinstance(item, (tuple, list)) and item:
            head = item[0]
            if isinstance(head, str):
                ids.append(head)
            elif isinstance(head, Path):
                ids.append(head.stem)
            else:
                raise AssertionError(f"Unsupported tuple/list item shape for traversal output: {item!r}")
            continue
        raise AssertionError(f"Unsupported traversal item type: {type(item)!r}")
    return ids


def test_download_module_exposes_traversal_contract(src_dir):
    module = _load_download_module(src_dir)

    assert callable(getattr(module, "parse_args", None)), "`src/download.py` must define parse_args()"
    assert callable(getattr(module, "main", None)), "`src/download.py` must define main()"

    name, fn = _pick_traversal_fn(module)
    assert fn is not None, (
        "`src/download.py` must define a KITTI traversal function, e.g. "
        "`iter_kitti_samples()` or `iter_kitti_pairs()`"
    )
    assert name


def test_download_traversal_can_scan_complete_dataset(src_dir, sample_kitti_tree):
    module = _load_download_module(src_dir)
    _, fn = _pick_traversal_fn(module)
    assert fn is not None, "Missing traversal function"

    items, _ = _invoke_traversal(fn, sample_kitti_tree["root"])
    ids = _normalize_ids(items)

    assert ids, "Traversal should return at least one sample"
    assert ids[:3] == sample_kitti_tree["sample_ids"], "Traversal should be deterministic and sorted by sample id"
    assert "999999" not in ids, "Traversal should ignore orphan images with missing labels"


def test_download_traversal_supports_partial_mode(src_dir, sample_kitti_tree):
    module = _load_download_module(src_dir)
    _, fn = _pick_traversal_fn(module)
    assert fn is not None, "Missing traversal function"

    items, partial_supported = _invoke_traversal(fn, sample_kitti_tree["root"], limit=2)
    assert partial_supported, "Traversal must support a partial/subset mode (limit or selected ids)"

    ids = _normalize_ids(items)
    assert len(ids) <= 2, "Partial traversal should not return more samples than requested"


def test_download_main_smoke_with_mocked_traversal(monkeypatch, src_dir, tmp_path):
    module = _load_download_module(src_dir)
    traversal_name, traversal_fn = _pick_traversal_fn(module)
    assert traversal_fn is not None, "Missing traversal function"

    fake_args = SimpleNamespace(
        kitti_dir=tmp_path,
        dataset_dir=tmp_path,
        limit=2,
        subset=2,
        partial=True,
        output=None,
    )
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)
    monkeypatch.setattr(
        module,
        traversal_name,
        lambda *args, **kwargs: [("000000", tmp_path / "a.png", tmp_path / "a.txt")],
        raising=True,
    )

    result = module.main()
    assert result in (0, None), "main() should return 0/None on success"
