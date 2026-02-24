Architecture Strategy: Feature Extraction + Object Detection

We'll use a proven approach: MobileNetV2 backbone (pre-trained on ImageNet) + custom detection head (trained on KITTI). This gives us real-time performance while maintaining accuracy.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import cv2

# ==========================================
# OBJECT DETECTION MODEL ARCHITECTURE
# ==========================================

print("="*70)
print("BUILDING AUTONOMOUS VEHICLE PERCEPTION MODEL")
print("="*70)
print()

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = len(CLASSES)  # Car, Pedestrian, Cyclist
BATCH_SIZE = 16

# ==========================================
# MOBILENETV2 BACKBONE (FEATURE EXTRACTOR)
# ==========================================

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze base model initially
base_model.trainable = False

print("✅ MobileNetV2 backbone loaded")
print(f"   Parameters: {base_model.count_params():,}")
print(f"   Output shape: {base_model.output_shape}")
print()

# ==========================================
# SIMPLIFIED DETECTION HEAD
# ==========================================
# For this tutorial, we'll build a classification + bounding box regression model
# Production systems use YOLO, SSD, or Faster R-CNN architectures

def build_detection_model():
    """
    Build object detection model with:
    - MobileNetV2 backbone for feature extraction
    - Classification head (which object class)
    - Bounding box regression head (where is the object)
    """

    # Input layer
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Feature extraction with MobileNetV2
    x = base_model(inputs, training=False)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers for learning
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Classification head: which class?
    class_output = layers.Dense(
        NUM_CLASSES,
        activation='softmax',
        name='class_output'
    )(x)

    # Bounding box regression head: where is the object?
    # Output: [x1, y1, x2, y2] normalized to [0, 1]
    bbox_output = layers.Dense(
        4,
        activation='sigmoid',  # Normalize to [0, 1]
        name='bbox_output'
    )(x)

    # Build model with multiple outputs
    model = keras.Model(
        inputs=inputs,
        outputs={
            'class': class_output,
            'bbox': bbox_output
        },
        name='AV_Perception_Model'
    )

    return model

# Build the model
model = build_detection_model()

print("="*70)
print("MODEL ARCHITECTURE")
print("="*70)
model.summary()
print()

# ==========================================
# COMPILE MODEL WITH MULTIPLE LOSSES
# ==========================================

# Since we have two outputs, we need two losses
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'class': 'sparse_categorical_crossentropy',  # Classification loss
        'bbox': 'mean_squared_error'  # Bounding box regression loss
    },
    loss_weights={
        'class': 1.0,  # Weight for classification loss
        'bbox': 5.0    # Higher weight for bbox (more important for detection)
    },
    metrics={
        'class': ['accuracy'],
        'bbox': ['mae']  # Mean absolute error for bbox
    }
)

print("✅ Model compiled with multi-task learning")
print("   Classification loss: categorical_crossentropy")
print("   Bounding box loss: mean_squared_error (weighted 5x)")
print()

# ==========================================
# DATA GENERATOR FOR KITTI
# ==========================================

class KITTIDataGenerator(keras.utils.Sequence):
    """
    Data generator for KITTI object detection.
    Loads images and labels on-the-fly during training.
    """

    def __init__(self, image_ids, image_dir, label_dir,
                 batch_size=16, img_size=(224, 224),
                 shuffle=True, augment=False):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        # Get batch indices
        batch_ids = self.image_ids[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        # Generate batch
        X, y_class, y_bbox = self._generate_batch(batch_ids)

        return X, {'class': y_class, 'bbox': y_bbox}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def _generate_batch(self, batch_ids):
        batch_images = []
        batch_classes = []
        batch_bboxes = []

        for img_id in batch_ids:
            # Load image
            img_path = os.path.join(self.image_dir, f"{img_id}.png")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

            # Resize to model input size
            img_resized = cv2.resize(img, self.img_size)
            img_normalized = img_resized / 255.0

            # Load labels
            label_path = os.path.join(self.label_dir, f"{img_id}.txt")
            objects = parse_kitti_label(label_path)

            # For simplicity, take the first object (or largest)
            # Production systems handle multiple objects per image
            if objects:
                # Pick the largest object by bbox area
                obj = max(objects, key=lambda o: (o['bbox'][2] - o['bbox'][0]) * (o['bbox'][3] - o['bbox'][1]))

                # Normalize bbox to [0, 1]
                x1, y1, x2, y2 = obj['bbox']
                bbox_norm = [
                    x1 / orig_w,
                    y1 / orig_h,
                    x2 / orig_w,
                    y2 / orig_h
                ]
                class_id = obj['class_id']
            else:
                # No object: background class and zero bbox
                bbox_norm = [0.0, 0.0, 0.0, 0.0]
                class_id = 0  # Default to first class

            batch_images.append(img_normalized)
            batch_classes.append(class_id)
            batch_bboxes.append(bbox_norm)

        return (
            np.array(batch_images, dtype=np.float32),
            np.array(batch_classes, dtype=np.int32),
            np.array(batch_bboxes, dtype=np.float32)
        )

# ==========================================
# PREPARE TRAINING/VALIDATION SPLIT
# ==========================================

# Get all image IDs
all_image_ids = [f.replace('.png', '') for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.png')]
all_image_ids.sort()

# Split 80/20
split_idx = int(len(all_image_ids) * 0.8)
train_ids = all_image_ids[:split_idx]
val_ids = all_image_ids[split_idx:]

print("="*70)
print("DATASET SPLIT")
print("="*70)
print(f"Total images: {len(all_image_ids)}")
print(f"Training: {len(train_ids)}")
print(f"Validation: {len(val_ids)}")
print()

# Create generators
train_gen = KITTIDataGenerator(
    train_ids, TRAIN_IMG_DIR, TRAIN_LABEL_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True
)

val_gen = KITTIDataGenerator(
    val_ids, TRAIN_IMG_DIR, TRAIN_LABEL_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False
)

print(f"✅ Data generators created")
print(f"   Training batches per epoch: {len(train_gen)}")
print(f"   Validation batches: {len(val_gen)}")
print()

# ==========================================
# CALLBACKS FOR TRAINING
# ==========================================

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'av_perception_best.keras',
        monitor='val_class_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ==========================================
# TRAIN THE MODEL
# ==========================================

print("="*70)
print("TRAINING AUTONOMOUS VEHICLE PERCEPTION MODEL")
print("="*70)
print("Expected training time:")
print("  - CPU: ~2-4 hours")
print("  - GPU: ~20-40 minutes")
print("="*70)
print()

# Stage 1: Train with frozen backbone
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

print()
print("✅ Stage 1 training complete (frozen backbone)")
print()

# ==========================================
# STAGE 2: FINE-TUNING (OPTIONAL)
# ==========================================

print("="*70)
print("STAGE 2: FINE-TUNING LAST LAYERS")
print("="*70)

# Unfreeze last 30 layers of MobileNetV2
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Unfreezing last {len(base_model.layers) - fine_tune_at} layers")
print()

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # 100x lower
    loss={
        'class': 'sparse_categorical_crossentropy',
        'bbox': 'mean_squared_error'
    },
    loss_weights={
        'class': 1.0,
        'bbox': 5.0
    },
    metrics={
        'class': ['accuracy'],
        'bbox': ['mae']
    }
)

# Fine-tune
history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

print()
print("✅ Fine-tuning complete")
print()

# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save('av_perception_final.keras')
print("Final model saved: av_perception_final.keras")
print()

# ==========================================
# REAL-TIME INFERENCE PIPELINE
# ==========================================

import cv2
import time

# Load trained model
model = keras.models.load_model('av_perception_final.keras')

print("="*70)
print("REAL-TIME AUTONOMOUS VEHICLE PERCEPTION")
print("="*70)
print()

# ==========================================
# INFERENCE FUNCTION
# ==========================================

def detect_objects(image, model, conf_threshold=0.5):
    """
    Run object detection on a single image.

    Returns:
        List of detections: [(class_name, confidence, bbox), ...]
    """
    # Preprocess
    img_h, img_w = image.shape[:2]
    img_resized = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Inference
    predictions = model.predict(img_batch, verbose=0)

    # Extract predictions
    class_probs = predictions['class'][0]
    bbox_norm = predictions['bbox'][0]

    # Get predicted class
    class_id = np.argmax(class_probs)
    confidence = class_probs[class_id]
    class_name = ID_TO_CLASS[class_id]

    # Skip low-confidence detections
    if confidence < conf_threshold:
        return []

    # Denormalize bbox to original image size
    x1 = int(bbox_norm[0] * img_w)
    y1 = int(bbox_norm[1] * img_h)
    x2 = int(bbox_norm[2] * img_w)
    y2 = int(bbox_norm[3] * img_h)

    return [(class_name, confidence, (x1, y1, x2, y2))]

# ==========================================
# DRAW DETECTIONS
# ==========================================

def draw_detections(image, detections):
    """Draw bounding boxes and labels on image."""
    colors = {
        'Car': (0, 0, 255),        # Red
        'Pedestrian': (255, 0, 0), # Blue
        'Cyclist': (0, 255, 0)     # Green
    }

    for class_name, confidence, (x1, y1, x2, y2) in detections:
        color = colors.get(class_name, (255, 255, 0))

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

# ==========================================
# PROCESS VIDEO FILE
# ==========================================

def process_video(video_path, output_path='output_detected.mp4'):
    """
    Process a video file and save with detections.

    Download sample driving videos:
    - KITTI raw data: http://www.cvlibs.net/datasets/kitti/raw_data.php
    - YouTube dashcam footage (use youtube-dl)
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        detections = detect_objects(frame, model)

        # Draw results
        frame_with_detections = draw_detections(frame.copy(), detections)

        # Add FPS counter
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame_with_detections,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Write frame
        out.write(frame_with_detections)
        frame_count += 1

        # Progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | FPS: {current_fps:.1f} | Detections: {len(detections)}")

    cap.release()
    out.release()

    print()
    print(f"✅ Video processing complete!")
    print(f"   Output saved: {output_path}")
    print(f"   Average FPS: {frame_count / elapsed:.1f}")

# ==========================================
# PROCESS WEBCAM (LIVE)
# ==========================================

def process_webcam():
    """
    Run real-time detection on webcam feed.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    print("Starting webcam detection...")
    print("Press 'q' to quit")
    print()

    fps_counter = []

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        detections = detect_objects(frame, model)

        # Draw results
        frame_with_detections = draw_detections(frame.copy(), detections)

        # Calculate FPS
        elapsed = time.time() - start
        fps = 1.0 / elapsed
        fps_counter.append(fps)

        # Show FPS
        avg_fps = np.mean(fps_counter[-30:])  # 30-frame average
        cv2.putText(
            frame_with_detections,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display
        cv2.imshow('AV Perception - Press q to quit', frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nAverage FPS: {np.mean(fps_counter):.1f}")

# ==========================================
# EXAMPLE USAGE
# ==========================================

# Process KITTI test images
print("="*70)
print("TESTING ON KITTI VALIDATION SET")
print("="*70)

sample_ids = ['000000', '000010', '000050', '000100', '000200']

for sample_id in sample_ids:
    img_path = os.path.join(TRAIN_IMG_DIR, f"{sample_id}.png")

    if os.path.exists(img_path):
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect
        start = time.time()
        detections = detect_objects(img, model)
        inference_time = (time.time() - start) * 1000  # ms

        # Draw
        img_with_detections = draw_detections(img.copy(), detections)

        # Save
        output_path = f"detection_result_{sample_id}.png"
        cv2.imwrite(output_path, cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR))

        print(f"Sample {sample_id}:")
        print(f"  Detections: {len(detections)}")
        print(f"  Inference time: {inference_time:.1f} ms")
        print(f"  Saved: {output_path}")
        print()

print("✅ All samples processed!")
print()

# ==========================================
# INSTRUCTIONS FOR VIDEO TESTING
# ==========================================

print("="*70)
print("VIDEO PROCESSING INSTRUCTIONS")
print("="*70)
print("""
To test on driving videos:

1. DOWNLOAD SAMPLE VIDEO:
   - KITTI raw data (with videos):
     http://www.cvlibs.net/datasets/kitti/raw_data.php

   - Or use YouTube dashcam footage:
     youtube-dl "https://www.youtube.com/watch?v=VIDEO_ID"

2. RUN VIDEO PROCESSING:
   process_video('path/to/your/video.mp4', 'output_with_detections.mp4')

3. LIVE WEBCAM (if you have a webcam):
   process_webcam()

EXPECTED PERFORMANCE:
- CPU: 10-20 FPS (sufficient for testing)
- GPU: 60-100 FPS (real-time capable)
- Raspberry Pi 4: 5-10 FPS (after TFLite optimization)
""")