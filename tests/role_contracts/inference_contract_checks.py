from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest

from tests.conftest import load_module_from_path


def _make_tensorflow_stub():
    tf_mod = types.ModuleType("tensorflow")
    keras_mod = types.ModuleType("tensorflow.keras")
    models_ns = types.SimpleNamespace(load_model=lambda *args, **kwargs: None)
    keras_mod.models = models_ns
    tf_mod.keras = keras_mod
    return tf_mod, keras_mod


def _make_cv2_stub():
    cv2_mod = types.ModuleType("cv2")
    cv2_mod.CAP_PROP_FPS = 5
    cv2_mod.CAP_PROP_FRAME_WIDTH = 3
    cv2_mod.CAP_PROP_FRAME_HEIGHT = 4
    cv2_mod.CAP_PROP_FRAME_COUNT = 7
    cv2_mod.COLOR_BGR2RGB = 1
    cv2_mod.FONT_HERSHEY_SIMPLEX = 0
    cv2_mod.resize = lambda image, size: image
    cv2_mod.cvtColor = lambda image, code: image
    cv2_mod.rectangle = lambda *args, **kwargs: None
    cv2_mod.putText = lambda *args, **kwargs: None
    cv2_mod.VideoWriter_fourcc = lambda *args: 0
    cv2_mod.imshow = lambda *args, **kwargs: None
    cv2_mod.waitKey = lambda *args, **kwargs: ord("q")
    cv2_mod.destroyAllWindows = lambda: None
    return cv2_mod


def _load_inference_module(monkeypatch, src_dir: Path):
    tf_mod, keras_mod = _make_tensorflow_stub()
    cv2_mod = _make_cv2_stub()
    ultralytics_mod = types.ModuleType("ultralytics")
    ultralytics_mod.YOLO = object

    monkeypatch.setitem(sys.modules, "tensorflow", tf_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras_mod)
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_mod)

    return load_module_from_path(src_dir / "inference.py")


def _call_detect_objects(fn, model, image):
    sig = inspect.signature(fn)
    kwargs = {}
    for threshold_name in ("conf_threshold", "confidence_threshold", "threshold"):
        if threshold_name in sig.parameters:
            kwargs[threshold_name] = 0.5
            break

    try:
        return fn(model, image, **kwargs)
    except TypeError:
        try:
            return fn(image, model, **kwargs)
        except TypeError:
            return fn(image, **kwargs)


def _call_process_video(fn, model):
    sig = inspect.signature(fn)
    kwargs = {}
    for name in sig.parameters:
        if name in {"video_path", "input_path", "source", "source_path"}:
            kwargs[name] = "input.mp4"
        elif name in {"output_path", "save_path", "destination", "dest_path"}:
            kwargs[name] = "output.mp4"
        elif name in {"model", "detector", "detector_model"}:
            kwargs[name] = model
        elif name in {"conf_threshold", "confidence_threshold", "threshold"}:
            kwargs[name] = 0.5
    return fn(**kwargs)


def test_inference_module_exposes_core_contract(monkeypatch, src_dir):
    module = _load_inference_module(monkeypatch, src_dir)

    for name in ("detect_objects", "draw_detections", "process_video"):
        assert hasattr(module, name), f"`src/inference.py` must define `{name}`"
        assert callable(getattr(module, name)), f"`{name}` must be callable"


def test_detect_objects_filters_and_denormalizes_bbox(monkeypatch, src_dir):
    np = pytest.importorskip("numpy")
    module = _load_inference_module(monkeypatch, src_dir)

    # Normalize label mapping so contract checks stay deterministic.
    monkeypatch.setattr(module, "ID_TO_CLASS", {0: "Car", 1: "Pedestrian", 2: "Cyclist"}, raising=False)
    monkeypatch.setattr(module, "CLASSES", ["Car", "Pedestrian", "Cyclist"], raising=False)

    class FakeModel:
        def predict(self, batch, verbose=0):
            return {
                "class": np.array([[0.1, 0.8, 0.1]], dtype=np.float32),
                "bbox": np.array([[0.1, 0.2, 0.5, 0.6]], dtype=np.float32),
            }

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = _call_detect_objects(module.detect_objects, FakeModel(), image)

    assert isinstance(detections, list), "detect_objects() must return a list"
    assert detections, "Expected one detection above threshold"

    class_name, confidence, bbox = detections[0]
    assert class_name in {"Car", "Pedestrian", "Cyclist"}
    assert float(confidence) >= 0.5
    assert isinstance(bbox, (tuple, list)) and len(bbox) == 4

    x1, y1, x2, y2 = map(int, bbox)
    assert 0 <= x1 <= 200 and 0 <= x2 <= 200
    assert 0 <= y1 <= 100 and 0 <= y2 <= 100
    assert x2 >= x1 and y2 >= y1


def test_process_video_smoke_with_mock_cv2(monkeypatch, src_dir):
    np = pytest.importorskip("numpy")
    module = _load_inference_module(monkeypatch, src_dir)

    frames = [np.zeros((16, 32, 3), dtype=np.uint8), np.ones((16, 32, 3), dtype=np.uint8)]
    state = {"capture_released": False, "writer_released": False, "written": 0}

    class FakeCapture:
        def __init__(self, path):
            self._idx = 0
            self.path = path

        def isOpened(self):
            return True

        def read(self):
            if self._idx >= len(frames):
                return False, None
            frame = frames[self._idx]
            self._idx += 1
            return True, frame.copy()

        def get(self, prop):
            if prop == module.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == module.cv2.CAP_PROP_FRAME_WIDTH:
                return 32
            if prop == module.cv2.CAP_PROP_FRAME_HEIGHT:
                return 16
            if prop == module.cv2.CAP_PROP_FRAME_COUNT:
                return len(frames)
            return 0

        def release(self):
            state["capture_released"] = True

    class FakeWriter:
        def __init__(self, *args, **kwargs):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame)
            state["written"] += 1

        def release(self):
            state["writer_released"] = True

    monkeypatch.setattr(module.cv2, "VideoCapture", FakeCapture, raising=False)
    monkeypatch.setattr(module.cv2, "VideoWriter", FakeWriter, raising=False)
    monkeypatch.setattr(module.cv2, "VideoWriter_fourcc", lambda *args: 0, raising=False)
    monkeypatch.setattr(module, "detect_objects", lambda *args, **kwargs: [("Car", 0.9, (1, 1, 10, 10))], raising=False)
    monkeypatch.setattr(module, "draw_detections", lambda frame, detections, *a, **kw: frame, raising=False)
    monkeypatch.setattr(module, "model", object(), raising=False)

    if hasattr(module, "time") and hasattr(module.time, "time"):
        tick = {"v": 0.0}

        def _fake_time():
            tick["v"] += 0.01
            return tick["v"]

        monkeypatch.setattr(module.time, "time", _fake_time, raising=False)

    result = _call_process_video(module.process_video, object())

    assert state["capture_released"], "process_video() must release VideoCapture"
    assert state["writer_released"], "process_video() must release VideoWriter"
    assert state["written"] == len(frames), "process_video() must write processed frames"
    assert isinstance(result, dict), (
        "process_video() should return a runtime summary dict so real-time behavior "
        "(FPS and processed frames) can be validated by tests/integration."
    )
    assert any(key in result for key in ("fps", "avg_fps", "effective_fps")), (
        "Runtime summary must include an FPS field (e.g. `fps` or `avg_fps`)."
    )
    assert any(key in result for key in ("frames_processed", "processed_frames", "num_frames")), (
        "Runtime summary must include processed frame count."
    )
    fps_key = next((key for key in ("fps", "avg_fps", "effective_fps") if key in result), None)
    if fps_key is not None:
        assert float(result[fps_key]) >= 0, "Reported FPS must be numeric and non-negative"
