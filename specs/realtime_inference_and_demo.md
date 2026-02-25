# Real-Time Inference And Demo Workflow (Ground Truth)

## Canonicalization Status

This spec is the canonical home for the migrated inference/demo tutorial notes previously kept in a root markdown file. The legacy file was removed on 2026-02-25 after consolidation into `specs/`.

## Inference Entry Point

After training, the documented flow loads `av_perception_final.keras` and defines a real-time inference pipeline for image, video, and webcam inputs.

## `detect_objects(...)` Behavior

The inference function:

- resizes input image to the model input (`224x224`)
- normalizes pixels to `[0,1]`
- runs `model.predict(...)`
- reads `class` and `bbox` outputs
- picks the top class by `argmax`
- applies a confidence threshold (`conf_threshold=0.5` default)
- denormalizes bbox coordinates back to original image dimensions
- returns a list of detections in the form `(class_name, confidence, (x1, y1, x2, y2))`

## Rendering / Overlay Behavior

`draw_detections(...)` overlays:

- class-colored bounding boxes (`Car`, `Pedestrian`, `Cyclist`)
- text labels including class name and confidence

## Video Processing Workflow

`process_video(video_path, output_path='output_detected.mp4')`:

- opens a video file with OpenCV
- reads video FPS/resolution/frame count metadata
- writes annotated frames using `mp4v`
- runs detection per frame
- draws overlays and an FPS counter on each frame
- logs progress every 30 frames
- saves the processed output video

## Webcam Processing Workflow

`process_webcam()`:

- opens default webcam (`cv2.VideoCapture(0)`)
- runs detection continuously
- overlays detections and rolling-average FPS (last 30 frames)
- displays live window titled `AV Perception - Press q to quit`
- exits on `q`

## Demo / Evaluation Outputs Mentioned

The docs include example processing on selected KITTI sample IDs and save output images as `detection_result_<id>.png`.

## Video Test Inputs And Expected Runtime Performance

The docs recommend testing with KITTI raw videos or YouTube dashcam footage, and list expected FPS ranges:

- CPU: 10-20 FPS
- GPU: 60-100 FPS
- Raspberry Pi 4 (after TFLite optimization): 5-10 FPS

## Relation To Final Task

The documented `process_video(...)` and overlay drawing behavior align with the stated final task of overlaying pedestrian/car/cyclist detections on a YouTube video.

## Final Task Note (Migrated From Root `final_task.md`)

On 2026-02-25, the root `final_task.md` note was migrated into this spec and translated to English.

Final task (translated):

- Overlay a YouTube video with detections for `Pedestrians`, `Cars`, and `Cyclists`.
- Target video: `https://www.youtube.com/watch?v=UlnDwSQqH30`
