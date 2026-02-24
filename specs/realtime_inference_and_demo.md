# Real-Time Inference And Demo Workflow (Ground Truth)

## Inference Entry Point

After training, the documented flow loads `av_perception_final.keras` and defines a real-time inference pipeline for image, video, and webcam inputs.

Evidence:
- `building_realtime.md:378` to `building_realtime.md:387`

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

Evidence:
- `building_realtime.md:397` to `building_realtime.md:403`
- `building_realtime.md:405` to `building_realtime.md:432`

## Rendering / Overlay Behavior

`draw_detections(...)` overlays:

- class-colored bounding boxes (`Car`, `Pedestrian`, `Cyclist`)
- text labels including class name and confidence

Evidence:
- `building_realtime.md:438` to `building_realtime.md:458`

## Video Processing Workflow

`process_video(video_path, output_path='output_detected.mp4')`:

- opens a video file with OpenCV
- reads video FPS/resolution/frame count metadata
- writes annotated frames using `mp4v`
- runs detection per frame
- draws overlays and an FPS counter on each frame
- logs progress every 30 frames
- saves the processed output video

Evidence:
- `building_realtime.md:464` to `building_realtime.md:472`
- `building_realtime.md:474` to `building_realtime.md:485`
- `building_realtime.md:492` to `building_realtime.md:524`
- `building_realtime.md:525` to `building_realtime.md:531`

## Webcam Processing Workflow

`process_webcam()`:

- opens default webcam (`cv2.VideoCapture(0)`)
- runs detection continuously
- overlays detections and rolling-average FPS (last 30 frames)
- displays live window titled `AV Perception - Press q to quit`
- exits on `q`

Evidence:
- `building_realtime.md:537` to `building_realtime.md:545`
- `building_realtime.md:548` to `building_realtime.md:589`

## Demo / Evaluation Outputs Mentioned

The docs include example processing on selected KITTI sample IDs and save output images as `detection_result_<id>.png`.

Evidence:
- `building_realtime.md:595` to `building_realtime.md:626`

## Video Test Inputs And Expected Runtime Performance

The docs recommend testing with KITTI raw videos or YouTube dashcam footage, and list expected FPS ranges:

- CPU: 10-20 FPS
- GPU: 60-100 FPS
- Raspberry Pi 4 (after TFLite optimization): 5-10 FPS

Evidence:
- `building_realtime.md:641` to `building_realtime.md:652`
- `building_realtime.md:654` to `building_realtime.md:657`

## Relation To Final Task

The documented `process_video(...)` and overlay drawing behavior align with the stated final task of overlaying pedestrian/car/cyclist detections on a YouTube video.

Evidence:
- `final_task.md:2`
- `final_task.md:4`
- `building_realtime.md:464`
- `building_realtime.md:498`
- `building_realtime.md:501`

