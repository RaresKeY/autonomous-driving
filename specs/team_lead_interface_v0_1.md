# Team Lead Interface v0.1 (Kickoff Contract)

## Purpose

This document defines the concrete interface contracts for the 2-hour kickoff timebox so all roles can start in parallel with stable inputs/outputs and mock-compatible schemas.

Related:
- `specs/team_lead_contracts.md`
- `roles.md`

## Scope (v0.1)

`v0.1` is a kickoff contract for planning and stub integration. It standardizes:

- dataset manifest structure
- split file format
- parser/generator output schema
- model training artifact metadata
- inference model I/O
- inference result format

This version does not require final production implementations.

## Versioning Rules

- Current version: `v0.1`
- Interface changes during kickoff must be recorded as `v0.x`
- Consumers must acknowledge changes before the producer merges
- Files using this contract should include `interface_version: "v0.1"` where applicable

## Canonical Class Set (v0.1)

- `Car`
- `Pedestrian`
- `Cyclist`

Class index mapping (frozen for v0.1):

```json
{
  "Car": 0,
  "Pedestrian": 1,
  "Cyclist": 2
}
```

## File Ownership (Kickoff Targets)

- Dataset collection (`Claudia`):
  - `artifacts/data/dataset_manifest.json`
  - `artifacts/data/splits/train.txt`
  - `artifacts/data/splits/val.txt`
- Data parsing / preprocessing (`Mihaela`):
  - `src/data/kitti_parser.py`
  - `src/data/kitti_generator.py`
  - `artifacts/data/qa_summary.json`
- Model training (`Anca`):
  - `src/train/train_detector.py`
  - `artifacts/models/model_card.json`
- Inference / demo (`Paul`):
  - `src/infer/detect.py`
  - `src/infer/run_video.py`
  - `artifacts/demo/inference_report.json`

Note:
- These are ownership targets for coordination. Files may be stubs during the 2-hour kickoff.

## Interface 1: Dataset Manifest

Producer: Dataset Collection
Consumers: Data Parsing, Model Training, Team Lead

Path (v0.1 target):
- `artifacts/data/dataset_manifest.json`

### Required JSON Schema (v0.1)

```json
{
  "interface_version": "v0.1",
  "dataset_name": "KITTI Object Detection",
  "dataset_root": "~/datasets/kitti",
  "sources": [
    {
      "name": "KITTI Object Detection",
      "type": "download",
      "license_note": "Non-commercial research use; verify official terms",
      "url": "https://www.cvlibs.net/datasets/kitti/"
    }
  ],
  "paths": {
    "train_images": "training/image_2",
    "train_labels": "training/label_2"
  },
  "counts": {
    "train_images": 7481,
    "train_labels": 7481
  },
  "class_scope": ["Car", "Pedestrian", "Cyclist"],
  "integrity": {
    "missing_images": 0,
    "missing_labels": 0,
    "corrupted_files": []
  },
  "generated_at": "2026-02-24T00:00:00Z",
  "owner": "Claudia"
}
```

### Acceptance Checks

- `interface_version == "v0.1"`
- `counts.train_images` and `counts.train_labels` are present
- `class_scope` matches canonical class set
- `paths.train_images` and `paths.train_labels` are present

## Interface 2: Split Files

Producer: Dataset Collection
Consumers: Data Parsing, Model Training

Paths (v0.1 target):
- `artifacts/data/splits/train.txt`
- `artifacts/data/splits/val.txt`
- Optional: `artifacts/data/splits/test.txt`

### Format (Plain Text)

- One sample ID per line
- No file extension
- Zero-padded KITTI ID format (example `000123`)
- Trailing newline allowed

Example `train.txt`:

```text
000000
000001
000002
```

### Acceptance Checks

- IDs are 6-digit numeric strings
- No duplicates within a split
- No overlap between `train.txt` and `val.txt`

## Interface 3: Parsed Object Schema (Per Annotation)

Producer: Data Parsing
Consumers: Data Generator, Training, QA Tools

### Python Dict Shape (v0.1)

```python
parsed_object = {
    "class_name": "Car",
    "class_id": 0,
    "bbox_xyxy": [50.0, 40.0, 180.0, 160.0],  # pixel coords in original image
    "truncated": 0.0,
    "occluded": 0,
    "source_label": "Car"
}
```

### Contract Rules

- `class_name` must be one of canonical classes
- `class_id` must match canonical mapping
- `bbox_xyxy` order is `[x1, y1, x2, y2]`
- `x2 > x1` and `y2 > y1`
- Coordinates are in original image pixel space before resizing

## Interface 4: Training Generator Batch Output

Producer: Data Parsing / Preprocessing
Consumer: Model Training

Primary target:
- `src/data/kitti_generator.py`

### v0.1 Batch Contract (Keras-friendly)

Return value for one batch:

```python
X_batch, y_batch = (
    np.ndarray,  # shape: (B, 224, 224, 3), dtype float32, range [0, 1]
    {
        "class": np.ndarray,  # shape: (B,), dtype int32/int64, values in {0,1,2}
        "bbox": np.ndarray,   # shape: (B, 4), dtype float32, normalized xyxy in [0, 1]
    }
)
```

### Empty-Object Behavior (v0.1)

- If no valid object remains after filtering:
  - assign class `0` (`Car`) as placeholder, and
  - bbox `[0.0, 0.0, 0.0, 0.0]`
- A follow-up version may replace this with an explicit background class

### Determinism

- Generator must accept or expose a seed option for reproducible ordering/shuffling in future versions
- For `v0.1`, deterministic behavior may be documented even if not fully implemented yet

### Mock Batch Example

```python
import numpy as np

X_batch = np.zeros((2, 224, 224, 3), dtype=np.float32)
y_batch = {
    "class": np.array([0, 1], dtype=np.int32),
    "bbox": np.array([
        [0.10, 0.20, 0.60, 0.80],
        [0.30, 0.25, 0.50, 0.70],
    ], dtype=np.float32),
}
```

## Interface 5: Model Artifact Metadata (Training -> Inference)

Producer: Model Training
Consumers: Inference, Team Lead

Path (v0.1 target):
- `artifacts/models/model_card.json`

### Required JSON Schema (v0.1)

```json
{
  "interface_version": "v0.1",
  "model_name": "av_perception_detector",
  "artifact_paths": {
    "best": "artifacts/models/av_perception_best.keras",
    "final": "artifacts/models/av_perception_final.keras"
  },
  "input": {
    "shape": [224, 224, 3],
    "dtype": "float32",
    "range": [0.0, 1.0]
  },
  "outputs": {
    "class": {
      "shape": ["B", 3],
      "dtype": "float32",
      "semantics": "softmax probabilities for canonical classes"
    },
    "bbox": {
      "shape": ["B", 4],
      "dtype": "float32",
      "format": "xyxy_normalized"
    }
  },
  "class_map": {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2
  },
  "training_run": {
    "config_name": "baseline_mobilenetv2",
    "date": "2026-02-24",
    "notes": "stub or initial training artifact for interface validation"
  },
  "owner": "Anca"
}
```

### Acceptance Checks

- `class_map` matches canonical class mapping exactly
- input shape is `[224, 224, 3]` for v0.1
- both `best` and `final` keys exist (paths may be placeholders in kickoff)

## Interface 6: Inference Model I/O Contract

Producer: Model Training (output semantics), Inference (consumer implementation)
Consumer: Inference / Demo

### Model Input (Single Image)

- Input image to inference helper can be original-size `numpy` array (`H x W x 3`)
- Inference helper is responsible for:
  - resize to `224x224`
  - normalize to `[0, 1]`
  - batch dimension add (`1 x 224 x 224 x 3`)

### Model Output (Single Prediction)

Expected outputs from `model.predict(...)`:

```python
class_pred, bbox_pred = model.predict(batch)  # or dict outputs by name
```

Supported v0.1 forms:

- Dict form:
  - `{"class": class_probs, "bbox": bbox_values}`
- Tuple/list form:
  - `[class_probs, bbox_values]`

Normalization / semantics:

- `class_probs`: shape `(1, 3)`
- `bbox_values`: shape `(1, 4)` normalized `xyxy` in `[0, 1]`

## Interface 7: Inference Result Format (`detect_objects`)

Producer: Inference / Demo
Consumers: Video/Webcam pipeline, QA scripts, Team Lead

Primary target:
- `src/infer/detect.py`

### Function Signature (v0.1)

```python
def detect_objects(image, model, conf_threshold=0.5):
    ...
```

### Return Format (v0.1)

List of tuples:

```python
[
    ("Car", 0.92, (120, 80, 300, 220)),
    ("Pedestrian", 0.76, (40, 60, 90, 180)),
]
```

Tuple schema:

- `class_name`: `str`
- `confidence`: `float` (0.0 to 1.0)
- `bbox_xyxy`: `tuple[int, int, int, int]` in original image pixels

### Rules

- Return `[]` if top confidence is below `conf_threshold`
- `bbox_xyxy` must be clipped to image bounds
- `class_name` must come from canonical class set

## Interface 8: Inference Report (Demo Output Metadata)

Producer: Inference / Demo
Consumers: Team Lead

Path (v0.1 target):
- `artifacts/demo/inference_report.json`

### Minimal JSON Schema

```json
{
  "interface_version": "v0.1",
  "owner": "Paul",
  "model_artifact_used": "artifacts/models/av_perception_final.keras",
  "source_type": "video",
  "source_path": "path/to/input.mp4",
  "output_path": "artifacts/demo/output_detected.mp4",
  "conf_threshold": 0.5,
  "processed_frames": 0,
  "avg_fps": 0.0,
  "notes": "stub run allowed during kickoff"
}
```

## Acceptance Checklist (Team Lead Fast Verification)

Use this checklist during the 2-hour kickoff wrap-up.

- Dataset manifest file exists and declares `interface_version: v0.1`
- Split files exist and contain zero-padded IDs
- Generator mock or real batch matches `X_batch` / `y_batch` contract
- Model card JSON exists with canonical class map and model I/O info
- Inference function contract is documented or stubbed with correct signature
- Inference return format is implemented or mocked as list of tuples
- Any deviations are logged with owner + next action

## Known v0.1 Simplifications

- Single-object training target per image (largest bbox) follows current tutorial design
- Empty-object handling uses placeholder class `0` + zero bbox
- Output schemas prioritize integration speed over long-term extensibility

## Next Planned Changes (Candidate v0.2)

- Add explicit background class support
- Add multi-object target format for training/evaluation
- Add structured dataclasses / typed schemas
- Add metrics report schema (precision/recall, IoU summaries)
