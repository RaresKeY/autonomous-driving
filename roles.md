# Autonomous Driving Project Roles (5 People)

## 1. Team Lead / System Integrator [Rares]

- Own architecture, interfaces, milestones, and merge decisions.
- Define contracts between components (data schema, model I/O, inference API).
- Maintain integration branch, test checklist, and weekly demo.
- Deliverables:
- README workflow
- Interface specs
- Integration plan
- Acceptance criteria

## 2. Dataset Collection & Curation Engineer [Claudia]

- Own dataset sourcing/downloading (KITTI + optional extra dashcam clips), storage layout, and versioning.
- Verify counts, file integrity, train/val/test splits, licenses, and metadata.
- Curate class coverage (`Car`, `Pedestrian`, `Cyclist`) and edge-case samples.
- Deliverables:
- Dataset manifest
- Split files
- Collection scripts
- Data inventory report

## 3. Data Parsing / Preprocessing / Generator Engineer [Mihaela]

- Own label parsing, filtering rules, normalization, resizing, augmentation, and dataloader/generator code.
- Build visualization and dataset QA tools (bbox overlays, class distribution stats).
- Provide a stable training-ready batch interface independent of model code.
- Deliverables:
- `parse_kitti_label` implementation
- Dataloader/generator module
- Preprocessing config
- QA plots / sample overlays

## 4. Model Training Engineer [Anca]

- Own model architecture, losses, training loop, checkpoints, experiment tracking, and fine-tuning.
- Start with a mock/stub dataloader contract so work can begin before full data tooling is finished.
- Produce trained artifacts and reproducible training configs.
- Deliverables:
- Training script/module
- Best/final model checkpoints
- Metrics summary
- Experiment logs

## 5. Inference / Demo / Evaluation Engineer [Paul]

- Own inference API, video/webcam pipeline, overlay rendering, FPS measurement, and demo outputs.
- Build against a model interface contract (can start with dummy model predictions).
- Add evaluation scripts for confidence threshold tuning and qualitative demo review.
- Deliverables:
- Inference module
- Video processing script
- Demo output samples
- Performance report

## Parallelization Rules

- Freeze interfaces on day 1:
- Data sample schema (image tensor + label format)
- Model output format (`class`, `bbox`)
- Inference function signature
- Use mocks immediately:
- Training uses dummy generator first
- Inference uses dummy model first
- Integrate on a fixed cadence (for example every 2-3 days), not continuously.

## Recommended Dependency Order

1. Team lead defines contracts.
2. Dataset collection, data tooling, training, and inference start in parallel.
3. Midpoint integration (real data into training, real model into inference).
4. Final QA and demo pass.
