# Object Detection Model Training Pipeline (Ground Truth)

## Canonicalization Status

This spec is the canonical home for the migrated training/tutorial notes previously kept in a root markdown file. The legacy file was removed on 2026-02-25 after consolidation into `specs/`.

## Architecture Strategy

The documented approach is transfer learning with:

- MobileNetV2 backbone pre-trained on ImageNet
- Custom detection head trained on KITTI

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

## Performance Expectations (Training)

The docs state estimated training time:

- CPU: ~2-4 hours
- GPU: ~20-40 minutes
