# models/

Acest folder conține modelele antrenate, gata de folosit pentru inferență.

## Fișiere

- `best.pt` — modelul cu cea mai bună performanță (generat de Anca după training)

## Cum folosești modelul (Paul)
```python
from ultralytics import YOLO

# Incarca modelul antrenat
model = YOLO("models/best.pt")

# Inferenta pe o imagine
results = model("imagine.jpg")

# Inferenta pe un video
results = model("video.mp4")

# Inferenta pe un frame (numpy array)
results = model(frame)  # frame din cv2.VideoCapture

# Acceseaza detectiile
for box in results[0].boxes:
    cls_id = int(box.cls)      # 0=Car, 1=Pedestrian, 2=Cyclist
    conf   = float(box.conf)   # confidenta 0-1
    x1,y1,x2,y2 = map(int, box.xyxy[0])  # coordonate bbox
```

## Clase detectate

| ID | Clasă |
|----|-------|
| 0  | Car |
| 1  | Pedestrian |
| 2  | Cyclist |

## Cum folosești dataloader-ul (Mihaela)

Pentru a conecta dataloader-ul tău la pipeline-ul de training, 
`train_model()` din `src/training.py` acceptă un `dataset_path` 
către un fișier `dataset.yaml` în format YOLO.

Fișierul `dataset.yaml` trebuie să arate așa:
```yaml
path: C:/curs_python/datasets/kitti_yolo
train: train/images
val:   val/images
test:  test/images

nc: 3
names:
  0: Car
  1: Pedestrian
  2: Cyclist
```

Odată ce ai `dataset.yaml` gata, dai path-ul către Anca sau rulezi direct:
```bash
python src/training.py --dataset "path/catre/dataset.yaml" --epochs 20
```