# Project Overview (Ground Truth)

## Purpose

This repository currently documents a tutorial-style autonomous driving perception workflow focused on real-time object detection with transfer learning, using KITTI data and a MobileNetV2-based detector pipeline.

Evidence:
- `define.md:26` states the focus is object detection with transfer learning.
- `define.md:28` describes a real-time detector for vehicles, pedestrians, and traffic signs from camera feeds.
- `dataset.md:371` to `dataset.md:376` outlines the next steps from dataset understanding to training and real-time deployment.

## Autonomous Driving Perception Areas Mentioned

The docs enumerate four perception tasks:

1. Lane detection and segmentation (semantic segmentation / pixel-wise road-lane masks)
2. Object detection (bounding boxes + class labels)
3. Depth estimation (monocular depth estimation)
4. Motion prediction (temporal CNNs or RNNs)

Evidence:
- `define.md:1`
- `define.md:7`
- `define.md:13`
- `define.md:18`

## Current Focus in This Repo

The current documented implementation path centers on object detection, with downstream uses listed as future/adjacent capabilities (tracking, distance estimation, prediction, path planning).

Evidence:
- `define.md:26`
- `define.md:30`

## Deliverable Goal (Stated Task)

The root task note specifies a final goal of overlaying detections for pedestrians/cars/cyclists on a YouTube video.

Evidence:
- `final_task.md:1`
- `final_task.md:2`
- `final_task.md:4`

## Repository State (Observed)

- Root content is documentation-heavy (`define.md`, `dataset.md`, `building_realtime.md`, `final_task.md`) plus `specs/` scaffolding.
- The model and pipeline logic currently exists as code embedded inside markdown documents, not as standalone Python modules in the repo root.

Evidence:
- `dataset.md:122` (Python imports begin in docs)
- `building_realtime.md:5` (Python imports begin in docs)

## Gaps / Open Questions

- No standalone runnable training or inference script is present in the repo root at the time of this spec update (only markdown-based code snippets were observed).
- The docs mention traffic signs in the high-level goal, but the tutorial implementation narrows training classes to `Car`, `Pedestrian`, and `Cyclist`.

Evidence:
- `define.md:28`
- `dataset.md:116`
- `dataset.md:140`

