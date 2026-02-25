from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import load_module_from_path


def _load_download_module(src_dir):
    return load_module_from_path(src_dir / "download.py")


def test_download_module_exposes_download_contract(src_dir):
    module = _load_download_module(src_dir)
    assert hasattr(module, "COMPONENTS")
    assert callable(getattr(module, "parse_args", None))
    assert callable(getattr(module, "main", None))
    assert "images" in module.COMPONENTS and "labels" in module.COMPONENTS


def test_download_args_support_output_dir_and_partial_components(monkeypatch, src_dir, tmp_path):
    module = _load_download_module(src_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["download.py", "--output-dir", str(tmp_path), "--components", "images", "--no-extract"],
    )
    args = module.parse_args()
    assert list(args.components) == ["images"]
    assert args.no_extract is True


def test_download_main_downloads_only_requested_components(monkeypatch, src_dir, tmp_path):
    module = _load_download_module(src_dir)
    fake_args = SimpleNamespace(
        output_dir=tmp_path,
        components=["images"],
        no_extract=False,
        force_download=False,
        force_extract=False,
        delete_archives=False,
        timeout=5,
    )
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)
    downloaded = []
    extracted = []
    verified = []
    monkeypatch.setattr(module, "download_file", lambda url, dest, timeout=60, force=False: downloaded.append((url, Path(dest))))
    monkeypatch.setattr(module, "extract_zip", lambda archive, output_dir, force_extract=False: extracted.append((Path(archive), Path(output_dir))))
    monkeypatch.setattr(module, "verify_expected_dirs", lambda output_dir, selected: verified.append((Path(output_dir), list(selected))), raising=False)
    assert module.main() == 0
    assert len(downloaded) == 1 and len(extracted) == 1 and verified


def test_download_main_skips_extract_when_no_extract_enabled(monkeypatch, src_dir, tmp_path):
    module = _load_download_module(src_dir)
    fake_args = SimpleNamespace(
        output_dir=tmp_path,
        components=["images", "labels"],
        no_extract=True,
        force_download=False,
        force_extract=False,
        delete_archives=False,
        timeout=5,
    )
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)
    downloads = []
    extracts = []
    monkeypatch.setattr(module, "download_file", lambda *a, **k: downloads.append((a, k)))
    monkeypatch.setattr(module, "extract_zip", lambda *a, **k: extracts.append((a, k)))
    assert module.main() == 0
    assert len(downloads) == 2 and not extracts


def test_download_main_returns_nonzero_on_download_failure(monkeypatch, src_dir, tmp_path):
    module = _load_download_module(src_dir)
    fake_args = SimpleNamespace(
        output_dir=tmp_path,
        components=["images"],
        no_extract=True,
        force_download=False,
        force_extract=False,
        delete_archives=False,
        timeout=5,
    )
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)
    monkeypatch.setattr(module, "download_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network failed")))
    assert module.main() not in (0, None)


def test_download_main_returns_nonzero_on_verify_failure(monkeypatch, src_dir, tmp_path):
    module = _load_download_module(src_dir)
    fake_args = SimpleNamespace(
        output_dir=tmp_path,
        components=["images"],
        no_extract=False,
        force_download=False,
        force_extract=False,
        delete_archives=False,
        timeout=5,
    )
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)
    monkeypatch.setattr(module, "download_file", lambda *a, **k: None)
    monkeypatch.setattr(module, "extract_zip", lambda *a, **k: None)
    monkeypatch.setattr(module, "verify_expected_dirs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("verify failed")), raising=False)
    assert module.main() not in (0, None)
