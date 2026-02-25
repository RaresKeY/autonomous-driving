from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace

import pytest

from tests.conftest import load_module_from_path


def _install_training_stubs(monkeypatch):
    cv2_mod = types.ModuleType("cv2")
    cv2_mod.imread = lambda *args, **kwargs: None
    cv2_mod.cvtColor = lambda image, code: image
    cv2_mod.resize = lambda image, size: image
    cv2_mod.COLOR_BGR2RGB = 1
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)

    ultralytics_mod = types.ModuleType("ultralytics")
    ultralytics_mod.YOLO = object
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_mod)

    tf_mod = types.ModuleType("tensorflow")
    keras_mod = types.ModuleType("tensorflow.keras")
    keras_mod.Input = lambda *args, **kwargs: object()
    keras_mod.Model = object
    keras_mod.optimizers = types.SimpleNamespace(Adam=lambda *args, **kwargs: object())
    keras_mod.utils = types.SimpleNamespace(Sequence=object)
    keras_mod.layers = types.SimpleNamespace(
        GlobalAveragePooling2D=lambda *args, **kwargs: (lambda x=None: x),
        Dense=lambda *args, **kwargs: (lambda x=None: x),
        Dropout=lambda *args, **kwargs: (lambda x=None: x),
    )
    keras_mod.applications = types.SimpleNamespace(MobileNetV2=lambda *args, **kwargs: object())
    keras_mod.callbacks = types.SimpleNamespace(
        EarlyStopping=lambda *args, **kwargs: object(),
        ModelCheckpoint=lambda *args, **kwargs: object(),
        ReduceLROnPlateau=lambda *args, **kwargs: object(),
    )
    tf_mod.keras = keras_mod

    monkeypatch.setitem(sys.modules, "tensorflow", tf_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.layers", keras_mod.layers)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.applications", keras_mod.applications)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.callbacks", keras_mod.callbacks)

    # Some implementations may import `keras` directly.
    keras_top = types.ModuleType("keras")
    keras_top.utils = keras_mod.utils
    monkeypatch.setitem(sys.modules, "keras", keras_top)


def _load_training_module(monkeypatch, src_dir):
    _install_training_stubs(monkeypatch)
    return load_module_from_path(src_dir / "training.py")


def _pick_callable(module, candidates):
    for name in candidates:
        value = getattr(module, name, None)
        if callable(value):
            return name, value
    return None, None


def test_training_module_exposes_cli_and_training_entrypoints(monkeypatch, src_dir):
    module = _load_training_module(monkeypatch, src_dir)

    assert callable(getattr(module, "parse_args", None)), "`src/training.py` must define parse_args()"
    assert callable(getattr(module, "main", None)), "`src/training.py` must define main()"

    train_name, train_fn = _pick_callable(module, ["train_model", "train", "run_training"])
    assert train_fn is not None, "Define one training entrypoint: train_model(), train(), or run_training()"

    build_name, build_fn = _pick_callable(module, ["build_detection_model", "build_model", "create_model"])
    assert build_fn is not None, "Define one model-builder function (e.g. build_model())"
    assert train_name and build_name  # keeps lints quiet and documents intent


def test_training_main_calls_training_function_with_mock_args(monkeypatch, src_dir, tmp_path):
    module = _load_training_module(monkeypatch, src_dir)
    train_name, train_fn = _pick_callable(module, ["train_model", "train", "run_training"])
    assert train_fn is not None, "Missing training entrypoint"

    fake_args = SimpleNamespace(
        kitti_dir=tmp_path,
        dataset_dir=tmp_path,
        output_model=tmp_path / "model.keras",
        model_out=tmp_path / "model.keras",
        epochs=1,
        batch_size=2,
        image_size=224,
        img_size=224,
        dry_run=True,
        no_train=False,
        subset=4,
    )
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)

    calls = {}

    def fake_train(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {"best_model": "av_perception_best.keras", "final_model": "av_perception_final.keras"}

    monkeypatch.setattr(module, train_name, fake_train, raising=True)
    result = module.main()

    assert calls, "main() must invoke the training entrypoint"
    assert result in (0, None), "CLI-style main() should return 0/None on success"


def test_training_main_returns_nonzero_on_failure(monkeypatch, src_dir, tmp_path):
    module = _load_training_module(monkeypatch, src_dir)
    train_name, train_fn = _pick_callable(module, ["train_model", "train", "run_training"])
    assert train_fn is not None, "Missing training entrypoint"

    fake_args = SimpleNamespace(
        kitti_dir=tmp_path,
        dataset_dir=tmp_path,
        output_model=tmp_path / "model.keras",
        epochs=1,
        batch_size=2,
        dry_run=False,
    )
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)

    def fake_train(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, train_name, fake_train, raising=True)

    try:
        result = module.main()
    except RuntimeError:
        pytest.fail("main() should handle training exceptions and return a non-zero code")

    assert result not in (0, None), "main() should return non-zero on failure"


def test_training_entrypoint_accepts_config_like_inputs(monkeypatch, src_dir, tmp_path):
    module = _load_training_module(monkeypatch, src_dir)
    train_name, train_fn = _pick_callable(module, ["train_model", "train", "run_training"])
    assert train_fn is not None, "Missing training entrypoint"

    sig = inspect.signature(train_fn)
    params = sig.parameters
    assert params, f"{train_name}() should accept configuration inputs (dataset path, epochs, etc.)"

    recognized = set(params)
    expected_any = {
        "kitti_dir",
        "dataset_dir",
        "train_data",
        "config",
        "epochs",
        "output_model",
    }
    assert recognized & expected_any, (
        f"{train_name}() signature should expose config-like params; found {sorted(recognized)}"
    )


def test_training_main_accepts_artifact_summary_result(monkeypatch, src_dir, tmp_path):
    module = _load_training_module(monkeypatch, src_dir)
    train_name, train_fn = _pick_callable(module, ["train_model", "train", "run_training"])
    assert train_fn is not None, "Missing training entrypoint"

    fake_args = SimpleNamespace(
        kitti_dir=tmp_path,
        dataset_dir=tmp_path,
        output_model=tmp_path / "model.keras",
        epochs=1,
        batch_size=2,
        dry_run=True,
    )
    monkeypatch.setattr(module, "parse_args", lambda: fake_args, raising=False)

    def fake_train(*args, **kwargs):
        return {
            "best_model": str(tmp_path / "av_perception_best.keras"),
            "final_model": str(tmp_path / "av_perception_final.keras"),
            "metrics": {"val_class_accuracy": 0.8},
        }

    monkeypatch.setattr(module, train_name, fake_train, raising=True)
    result = module.main()

    assert result in (0, None) or isinstance(result, dict), (
        "main() should return CLI success code (0/None) or a summary dict."
    )
    if isinstance(result, dict):
        assert any(key in result for key in ("best_model", "final_model", "artifacts")), (
            "If main() returns a summary dict, include artifact paths/metadata."
        )
