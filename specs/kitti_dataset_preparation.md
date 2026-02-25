# KITTI Dataset Preparation And Inspection (Ground Truth)

## Canonicalization Status

This spec is the canonical home for the migrated KITTI dataset tutorial notes previously kept in a root markdown file. The legacy file was removed on 2026-02-25 after consolidation into `specs/`.

## Dataset Role

The repository uses KITTI as the benchmark dataset for autonomous driving perception and specifically for the object detection tutorial workflow.

## Dataset Characteristics Noted In Docs

- 7,481 training images
- 7,518 test images (labels withheld for competition)
- Classes include vehicle/pedestrian categories and others (e.g., Car, Van, Truck, Pedestrian, Cyclist, Tram)
- Annotations include bounding boxes plus occlusion/truncation flags
- 3D locations/dimensions are available but not the main focus for the tutorial detector

## Download And Local Directory Layout

The docs specify downloading KITTI object images and labels, then extracting into `~/datasets/kitti`, with expected training subfolders:

- `~/datasets/kitti/training/image_2/`
- `~/datasets/kitti/training/label_2/`

## Verification Steps

The docs include shell checks to verify the presence and counts of image/label files, expecting `7481` for both training images and labels.

## Label Format Usage For This Project

KITTI label rows are described as containing object type, truncation, occlusion, alpha, and bounding box coordinates. For the tutorial object detector, the relevant fields are:

- object `type`
- bounding box `(x1, y1, x2, y2)`
- `occluded` (optional filtering)
- `truncated` (optional filtering)

## Training Classes (Tutorial Scope)

The tutorial narrows the training scope to three classes:

- `Car`
- `Pedestrian`
- `Cyclist`

It also states a filtering strategy for cleaner training examples (drop other classes and heavily occluded objects).

## Dataset Loader Configuration And Parsing Rules

The documented Python loader configures:

- `KITTI_DIR = ~/datasets/kitti`
- `TRAIN_IMG_DIR = training/image_2`
- `TRAIN_LABEL_DIR = training/label_2`

`parse_kitti_label(...)` returns parsed objects with class name/id, bbox, truncation, and occlusion, and applies these filters:

- keep only configured classes (optional)
- skip occluded level `>= 2`
- skip truncated `> 0.5`
- skip boxes with height `< min_height` (default `25`)

## Visualization And Dataset Stats Workflow

The docs include a visualization helper to draw class-colored bounding boxes on images and optionally save outputs, followed by:

- dataset existence check
- file counting
- sample visualization on selected IDs
- class distribution stats over parsed labels

## Output Artifacts Mentioned

The dataset inspection workflow saves example images named like `kitti_sample_<id>.png`.
