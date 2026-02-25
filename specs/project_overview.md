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

- Root content is now centered on `specs/` and `requirements.txt`; the `final_task.md` note has been migrated into `specs/realtime_inference_and_demo.md`.
- A top-level `requirements.txt` is present and captures the documented Python dependencies for dataset prep, training, inference, and testing (`tensorflow`, `numpy`, `opencv-python`, `matplotlib`, `Pillow`, `pytest`).
- The model and pipeline logic is currently documented in specs (migrated from markdown-embedded tutorial code), not implemented as standalone Python modules in the repo root.

## Gaps / Open Questions

- No standalone runnable training or inference script is present in the repo root at the time of this spec update (only markdown-based code snippets were observed).
- The docs mention traffic signs in the high-level goal, but the tutorial implementation narrows training classes to `Car`, `Pedestrian`, and `Cyclist`.
- The original tutorial workflow (dataset prep + training + realtime demo) exceeds the current `2-hour` planning/setup timebox and must be staged across later sessions.
