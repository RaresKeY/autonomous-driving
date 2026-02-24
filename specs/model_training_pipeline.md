# Object Detection Model Training Pipeline (Ground Truth)

## Architecture Strategy

The documented approach is transfer learning with:

- MobileNetV2 backbone pre-trained on ImageNet
- Custom detection head trained on KITTI

Evidence:
- `building_realtime.md:1`
- `building_realtime.md:3`
- `building_realtime.md:32` to `building_realtime.md:40`

## Core Model Configuration

The tutorial defines:

- input size `224x224`
- `BATCH_SIZE = 16`
- `NUM_CLASSES = len(CLASSES)` (shared with dataset doc class list)

Evidence:
- `building_realtime.md:23` to `building_realtime.md:26`

## Model Outputs

The detector is a simplified multi-output model (tutorial simplification, not YOLO/SSD/Faster R-CNN):

- `class` output: softmax classification over `NUM_CLASSES`
- `bbox` output: 4-value sigmoid regression `[x1, y1, x2, y2]` normalized to `[0, 1]`

Evidence:
- `building_realtime.md:50` to `building_realtime.md:52`
- `building_realtime.md:76` to `building_realtime.md:89`
- `building_realtime.md:94` to `building_realtime.md:99`

## Training Losses And Metrics

The model is compiled as a multi-task problem:

- optimizer: Adam (`lr=0.001`) for stage 1
- classification loss: `sparse_categorical_crossentropy`
- bounding box loss: `mean_squared_error`
- loss weights: class `1.0`, bbox `5.0`
- metrics: class accuracy, bbox MAE

Evidence:
- `building_realtime.md:117` to `building_realtime.md:131`

## KITTI Training Data Generator Behavior

The documented `KITTIDataGenerator`:

- loads images/labels on demand (`keras.utils.Sequence`)
- resizes images to model input size and normalizes pixel values to `[0,1]`
- parses KITTI labels using `parse_kitti_label(...)`
- chooses a single target object per image for training (largest bbox if multiple)
- normalizes bbox coordinates to image-relative values
- uses zero bbox and default class `0` when no objects are present

Evidence:
- `building_realtime.md:142` to `building_realtime.md:146`
- `building_realtime.md:190` to `building_realtime.md:193`
- `building_realtime.md:194` to `building_realtime.md:197`
- `building_realtime.md:198` to `building_realtime.md:216`

## Train/Validation Split

The docs split image IDs from `TRAIN_IMG_DIR` into an `80/20` train/validation split after sorting IDs.

Evidence:
- `building_realtime.md:232` to `building_realtime.md:239`

## Training Stages

### Stage 1

- Frozen MobileNetV2 backbone
- `model.fit(...)` for `30` epochs with train/validation generators

Evidence:
- `building_realtime.md:39` to `building_realtime.md:40`
- `building_realtime.md:309` to `building_realtime.md:316`

### Stage 2 (Optional Fine-Tuning)

- Unfreeze last `30` layers of MobileNetV2
- Recompile with lower learning rate (`1e-5`)
- Run additional `10` epochs

Evidence:
- `building_realtime.md:323` to `building_realtime.md:325`
- `building_realtime.md:330` to `building_realtime.md:343`
- `building_realtime.md:357` to `building_realtime.md:364`

## Training Callbacks And Checkpoints

The training pipeline uses:

- `EarlyStopping` on `val_loss` (patience `10`, restore best weights)
- `ModelCheckpoint` writing `av_perception_best.keras`, monitored by `val_class_accuracy`
- `ReduceLROnPlateau` on `val_loss`

Evidence:
- `building_realtime.md:273` to `building_realtime.md:294`

## Model Artifacts

- Best checkpoint: `av_perception_best.keras`
- Final saved model: `av_perception_final.keras`

Evidence:
- `building_realtime.md:281`
- `building_realtime.md:374` to `building_realtime.md:375`

## Performance Expectations (Training)

The docs state estimated training time:

- CPU: ~2-4 hours
- GPU: ~20-40 minutes

Evidence:
- `building_realtime.md:303` to `building_realtime.md:306`

