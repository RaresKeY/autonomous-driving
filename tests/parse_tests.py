from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import load_module_from_path


def _load_parse_module(src_dir):
    return load_module_from_path(src_dir / "parse_kitti_label.py")


def test_parse_module_exposes_download_role_contract(src_dir):
    module = _load_parse_module(src_dir)

    assert hasattr(module, "COMPONENTS"), "`src/parse_kitti_label.py` must define COMPONENTS for selectable downloads"
    assert callable(getattr(module, "parse_args", None)), "`src/parse_kitti_label.py` must define parse_args()"
    assert callable(getattr(module, "main", None)), "`src/parse_kitti_label.py` must define main()"

    components = getattr(module, "COMPONENTS")
    assert "images" in components and "labels" in components, "COMPONENTS must support partial downloads"


def test_parse_args_supports_output_dir_and_partial_components(monkeypatch, src_dir, tmp_path):
    module = _load_parse_module(src_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_kitti_label.py",
            "--output-dir",
            str(tmp_path / "custom_kitti"),
            "--components",
            "images",
            "--no-extract",
        ],
    )
    args = module.parse_args()

    assert hasattr(args, "output_dir"), "parse_args() must expose --output-dir"
    assert hasattr(args, "components"), "parse_args() must expose --components"
    assert list(args.components) == ["images"]
    assert getattr(args, "no_extract", False) is True


def test_main_downloads_only_requested_components(monkeypatch, src_dir, tmp_path):
    module = _load_parse_module(src_dir)

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

    monkeypatch.setattr(
        module,
        "download_file",
        lambda url, dest, timeout=60, force=False: downloaded.append((url, Path(dest), timeout, force)),
        raising=True,
    )
    monkeypatch.setattr(
        module,
        "extract_zip",
        lambda archive, output_dir, force_extract=False: extracted.append((Path(archive), Path(output_dir), force_extract)),
        raising=True,
    )
    monkeypatch.setattr(
        module,
        "verify_expected_dirs",
        lambda output_dir, selected: verified.append((Path(output_dir), list(selected))),
        raising=False,
    )

    result = module.main()

    assert result == 0, "main() should return 0 on success"
    assert len(downloaded) == 1, "Partial component selection should download only one archive"
    assert len(extracted) == 1, "Selected archive should be extracted when no_extract=False"
    assert downloaded[0][1].parent == tmp_path
    assert verified, "main() should verify extracted component directories"


def test_main_skips_extract_when_no_extract_enabled(monkeypatch, src_dir, tmp_path):
    module = _load_parse_module(src_dir)

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
    monkeypatch.setattr(
        module,
        "download_file",
        lambda *args, **kwargs: downloads.append((args, kwargs)),
        raising=True,
    )
    monkeypatch.setattr(
        module,
        "extract_zip",
        lambda *args, **kwargs: extracts.append((args, kwargs)),
        raising=True,
    )

    result = module.main()
    assert result == 0
    assert len(downloads) == 2, "Full mode should download both images and labels"
    assert not extracts, "--no-extract should skip extraction"


def test_main_returns_nonzero_on_download_failure(monkeypatch, src_dir, tmp_path):
    module = _load_parse_module(src_dir)

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

    def _boom(*args, **kwargs):
        raise RuntimeError("network failed")

    monkeypatch.setattr(module, "download_file", _boom, raising=True)

    result = module.main()
    assert result not in (0, None), "main() must return non-zero when a download step fails"

