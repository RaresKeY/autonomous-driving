# Object Detection Model Training Pipeline (Ground Truth)

## Canonicalization Status

This spec is the canonical home for the migrated training/tutorial notes previously kept in a root markdown file. The legacy file was removed on 2026-02-25 after consolidation into `specs/`.

## Architecture Strategy

The documented approach is transfer learning with:

- MobileNetV2 backbone pre-trained on ImageNet
- Custom detection head trained on KITTI

## Current Repo Implementation Status (2026-02-25)

- `src/training.py` exists and provides a contract-compliant CLI/training entrypoint surface for integration and tests.
- The current implementation is intentionally lightweight and returns artifact metadata (contract-first behavior) instead of running full training.
- This spec remains the tutorial baseline and records additional constraints for the future "YOLO + MobileNetV2" implementation target.

## External Architecture Constraints (Validated Online, 2026-02-25)

### MobileNetV2 (paper-level)

MobileNetV2 introduces inverted residuals and linear bottlenecks, and the paper explicitly applies the family to detection/segmentation tasks.

Implication:

- MobileNetV2 is a valid lightweight backbone candidate for AV perception, but the original MobileNetV2 object-detection examples are not YOLO by default.

### Ultralytics YOLO training constraints

Ultralytics train docs support:

- training from pretrained `.pt` weights (recommended)
- building from `.yaml` architecture config
- building from `.yaml` and transferring matching pretrained weights
- transfer-learning controls such as layer freezing
- dataset fraction/subsetting controls during training runs

Implication for Anca's "YOLO + MobileNetV2" task:

- This is a custom architecture integration task, not a simple model-name switch.
- The implementation should document one explicit path:
  - custom YAML backbone path (including channel/stride compatibility), or
  - custom source module integration in Ultralytics, or
  - phased delivery: baseline YOLO first, MobileNetV2-backed YOLO second

### Ultralytics custom YAML/backbone constraints

Ultralytics YAML/model docs highlight:

- custom module integration may require source-code changes/import wiring
- channel mismatch is a common failure mode when customizing backbones/heads
- `model.info()` and layer inspection are recommended validation/debugging steps
- loading pretrained weights into a custom YAML only transfers matching layers

Required documentation for a real custom YOLO+MobileNetV2 experiment in this repo:

- model YAML path / variant
- class count alignment (`nc=3` for `Car`, `Pedestrian`, `Cyclist`)
- pretrained weights source and expected partial weight transfer behavior
- freeze/unfreeze strategy (if used)
- validation sanity check (`model.info()` / layer summary) before long training runs

## Core Model Configuration

The tutorial defines:

- input size `224x224`
- `BATCH_SIZE = 16`
- `NUM_CLASSES = len(CLASSES)` (shared with dataset doc class list)

## Model Outputs

The detector is a simplified multi-output model (tutorial simplification, not YOLO/SSD/Faster R-CNN):

- `class` output: softmax classification over `NUM_CLASSES`
- `bbox` output: 4-value sigmoid regression `[x1, y1, x2, y2]` normalized to `[0, 1]`

## Training Losses And Metrics

The model is compiled as a multi-task problem:

- optimizer: Adam (`lr=0.001`) for stage 1
- classification loss: `sparse_categorical_crossentropy`
- bounding box loss: `mean_squared_error`
- loss weights: class `1.0`, bbox `5.0`
- metrics: class accuracy, bbox MAE

## KITTI Training Data Generator Behavior

The documented `KITTIDataGenerator`:

- loads images/labels on demand (`keras.utils.Sequence`)
- resizes images to model input size and normalizes pixel values to `[0,1]`
- parses KITTI labels using `parse_kitti_label(...)`
- chooses a single target object per image for training (largest bbox if multiple)
- normalizes bbox coordinates to image-relative values
- uses zero bbox and default class `0` when no objects are present

## Train/Validation Split

The docs split image IDs from `TRAIN_IMG_DIR` into an `80/20` train/validation split after sorting IDs.

## Training Stages

### Stage 1

- Frozen MobileNetV2 backbone
- `model.fit(...)` for `30` epochs with train/validation generators

### Stage 2 (Optional Fine-Tuning)

- Unfreeze last `30` layers of MobileNetV2
- Recompile with lower learning rate (`1e-5`)
- Run additional `10` epochs

## Training Callbacks And Checkpoints

The training pipeline uses:

- `EarlyStopping` on `val_loss` (patience `10`, restore best weights)
- `ModelCheckpoint` writing `av_perception_best.keras`, monitored by `val_class_accuracy`
- `ReduceLROnPlateau` on `val_loss`

## Model Artifacts

- Best checkpoint: `av_perception_best.keras`
- Final saved model: `av_perception_final.keras`

Current repo note:

- `src/training.py` currently returns these artifact names as placeholders for integration/test consistency; the files are not produced by a real training loop yet.

## Performance Expectations (Training)

The docs state estimated training time:

- CPU: ~2-4 hours
- GPU: ~20-40 minutes

## Architecture Divergence Note (Tutorial vs Current Team Ask)

- The migrated tutorial baseline is a TensorFlow/Keras MobileNetV2 + custom detection head.
- The current team assignment for `src/training.py` is a YOLO-based pipeline using MobileNetV2 concepts/backbone customization.
- Treat tutorial losses/metrics/timings in this spec as baseline guidance, not as finalized expectations for the future Ultralytics-based implementation.
