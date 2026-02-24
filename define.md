1. Lane Detection & Segmentation

Task: Identify drivable surface, lane markings, road boundaries

CNN Approach: Semantic segmentation—classify each pixel as road/lane/other. Output: Pixel-wise mask showing where the car can drive.

2. Object Detection

Task: Locate and classify vehicles, pedestrians, cyclists, traffic signs

CNN Approach: Object detection—output bounding boxes + class labels. Models: YOLO, SSD, Faster R-CNN (all use CNN backbones).

3. Depth Estimation

Task: Estimate distance to objects (critical for collision avoidance)

CNN Approach: Monocular depth estimation—predict depth map from single camera. Tesla's approach: Learn depth from stereo pairs during training, deploy monocular.
4. Motion Prediction

Task: Predict where vehicles/pedestrians will move next

CNN Approach: Temporal CNNs or RNNs—analyze video sequences to predict future positions. Critical for safe path planning.

---

Our Focus Today: Object Detection with Transfer Learning

We'll build a real-time object detection system that identifies vehicles, pedestrians, and traffic signs from camera feeds. This is the foundation of autonomous perception—once you can detect objects, you can:

    Track their motion over time (Kalman filters, SORT algorithm)
    Estimate their distance (depth networks or stereo vision)
    Predict their future positions (temporal models)
    Plan safe trajectories around them (path planning algorithms)