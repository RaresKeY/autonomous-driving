# Project Overview (Ground Truth)

## Canonicalization Status

On 2026-02-25, the project's tutorial/planning docs were consolidated into `specs/`. The former root files (`define.md`, `dataset.md`, `roles.md`, `building_realtime.md`) were removed after migration. This file and related specs are now the canonical ground truth for that context.

## Purpose

This repository currently documents a tutorial-style autonomous driving perception workflow focused on real-time object detection with transfer learning, using KITTI data and a MobileNetV2-based detector pipeline.

## Autonomous Driving Perception Areas Mentioned

The docs enumerate four perception tasks:

1. Lane detection and segmentation (semantic segmentation / pixel-wise road-lane masks)
2. Object detection (bounding boxes + class labels)
3. Depth estimation (monocular depth estimation)
4. Motion prediction (temporal CNNs or RNNs)

## Current Focus in This Repo

The current documented implementation path centers on object detection, with downstream uses listed as future/adjacent capabilities (tracking, distance estimation, prediction, path planning).

## Deliverable Goal (Stated Task)

The migrated final-task note (now recorded in `specs/realtime_inference_and_demo.md`) specifies a final goal of overlaying detections for pedestrians/cars/cyclists on a YouTube video.

## Current Execution Scope (Timeboxed)

The active project execution scope is timeboxed to `2 hours`. Within this timebox, the practical goal is to establish project structure and integration readiness (roles, contracts, dependencies, and initial interfaces) rather than complete full model training or final video demo delivery.

Included in 2-hour scope:

- Team role definition and ownership
- Team Lead interface contracts / handoff rules
- Dependency determination and `requirements.txt`
- Initial implementation planning for parallel workstreams

Deferred beyond 2-hour scope:

- Full KITTI dataset download/validation at scale
- End-to-end training run completion
- Final YouTube video detection demo artifact

## Repository State (Observed)

- Root content now includes:
  - `specs/` (canonical project/docs ground truth)
  - `src/` (role-owned implementation modules)
  - `tests/` (role-contract pytest suite, modularized under `tests/role_contracts/`)
  - `requirements.txt`
  - `discord_team_tasks.md` (team task message source used for assignment alignment)
- A top-level `requirements.txt` captures documented Python dependencies for dataset prep, training, inference, testing, and demo-video download support (`tensorflow`, `ultralytics`, `numpy`, `opencv-python`, `matplotlib`, `Pillow`, `yt-dlp`, `pytest`).
- Role-owned pipeline modules now exist in `src/`:
  - `src/inference.py`
  - `src/training.py`
  - `src/parse_kitti_label.py`
  - `src/download.py`
- Current `src/` files exist as empty placeholders by user preference; role implementations and executable contract behavior are intentionally deferred.
- The migrated tutorial pipeline remains documented in specs as the architectural baseline, while `src/` currently serves as file-level ownership placeholders only.

## Gaps / Open Questions

- The docs mention traffic signs in the high-level goal, but the tutorial implementation narrows training classes to `Car`, `Pedestrian`, and `Cyclist`.
- The current role files are empty placeholders, so role-contract tests fail until callable surfaces are implemented.
- `src/training.py` does not yet implement a real YOLO + MobileNetV2 custom-backbone training run.
- `src/parse_kitti_label.py` and `src/download.py` are aligned to filename semantics at the assignment level, but behavior is not implemented in the empty-file state.
- The original tutorial workflow (dataset prep + training + realtime demo) still exceeds the initial `2-hour` planning/setup timebox and must be staged across later sessions.

## Verification Snapshot (2026-02-25)

- After reverting `src/` role modules to empty files (user preference), the role-contract test suite no longer passes.
- Latest observed pytest result in this state: `19 failed, 4 passed` (`tests/`).
