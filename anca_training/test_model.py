# test_model.py
# Testeaza daca YOLOv8 detecteaza Car, Pedestrian, Cyclist
# Anca - Model Training Engineer

from ultralytics import YOLO
import cv2
import os

TARGET_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

OUR_MAPPING = {
    0:  "Pedestrian",
    1:  "Cyclist",
    2:  "Car",
    3:  "Cyclist",
    5:  "Car",
    7:  "Car",
}

COLORS = {
    "Pedestrian": (0, 255, 0),    # verde
    "Cyclist":    (255, 165, 0),  # portocaliu
    "Car":        (0, 0, 255),    # rosu
}

def download_video(url, output_path="test_video.mp4"):
    if os.path.exists(output_path):
        print(f"Video deja descarcat: {output_path}")
        return output_path

    import yt_dlp
    print("Descarc video de pe YouTube...")
    ydl_opts = {
        "format": "best[height<=480]",
        "outtmpl": output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Video descarcat!")
    return output_path


def draw_boxes(frame, results):
    """Deseneaza bounding boxes pe frame."""
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)

        if cls_id not in TARGET_CLASSES or conf < 0.3:
            continue

        label = OUR_MAPPING[cls_id]
        color = COLORS[label]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(text) * 10, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def test_on_video(video_path, num_frames=50, save_every=10):
    model = YOLO("yolov8n.pt")
    print(f"\nModel incarcat. Testez pe {num_frames} frame-uri...\n")

    # Folder pentru poze salvate
    os.makedirs("rezultate_poze", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS video: {fps:.1f}")

    detections_count = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
    frames_with_detections = 0
    saved_count = 0
    frame_idx = 0

    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        found_something = False

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if cls_id in TARGET_CLASSES and conf > 0.3:
                label = OUR_MAPPING[cls_id]
                detections_count[label] += 1
                found_something = True

        if found_something:
            frames_with_detections += 1

            # Salveaza poza la fiecare save_every frame-uri cu detectii
            if frames_with_detections % save_every == 1:
                annotated = draw_boxes(frame.copy(), results)
                poza_path = f"rezultate_poze/frame_{frame_idx:04d}.jpg"
                cv2.imwrite(poza_path, annotated)
                saved_count += 1
                print(f"  Salvat: {poza_path}")

        frame_idx += 1

    cap.release()

    # Raport final
    print("\n" + "=" * 40)
    print("RAPORT DETECTII")
    print("=" * 40)
    print(f"Frame-uri analizate:     {frame_idx}")
    print(f"Frame-uri cu detectii:   {frames_with_detections} ({100*frames_with_detections//max(frame_idx,1)}%)")
    print(f"Poze salvate:            {saved_count} (in folderul rezultate_poze/)")
    print()
    print("Obiecte detectate total:")
    for cls, count in detections_count.items():
        bar = "█" * min(count, 40)
        print(f"  {cls:<12} {count:>4}x  {bar}")
    print()

    total = sum(detections_count.values())
    if total == 0:
        print("CONCLUZIE: ❌ Modelul nu a detectat nimic. Avem nevoie de fine-tuning.")
    elif detections_count["Cyclist"] == 0:
        print("CONCLUZIE: ⚠️  Car si Pedestrian detectate, dar Cyclist lipseste.")
        print("           → Fine-tuning pe KITTI recomandat pentru Cyclist.")
    else:
        print("CONCLUZIE: ✅ Modelul detecteaza toate 3 clasele!")


if __name__ == "__main__":
    VIDEO_URL = "https://www.youtube.com/watch?v=UlnDwSQqH30"

    try:
        import yt_dlp
    except ImportError:
        os.system("pip install yt-dlp")

    video_path = download_video(VIDEO_URL)
    test_on_video(video_path, num_frames=50, save_every=5)