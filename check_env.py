import torch
from ultralytics import YOLO

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("→ Vei antrena pe GPU (mult mai rapid!)")
else:
    print("→ Vei antrena pe CPU (mai lent, dar merge)")