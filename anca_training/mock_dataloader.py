# mock_dataloader.py
# Simulează date reale până când Mihaela termină dataloader-ul ei
# Anca - Model Training Engineer

import torch

# Clasele pe care le detectăm
CLASSES = ["Car", "Pedestrian", "Cyclist"]
NUM_CLASSES = len(CLASSES)

def get_mock_batch(batch_size=4):
    """
    Returnează un batch fals de imagini + etichete.
    Format identic cu ce va livra Mihaela mai târziu.
    """
    # Imagini random: batch_size poze de 640x640 cu 3 canale RGB
    images = torch.rand(batch_size, 3, 640, 640)

    # Etichete random: fiecare imagine are câteva obiecte detectate
    labels = []
    for _ in range(batch_size):
        labels.append([
            {"class": 0, "class_name": "Car",        "bbox": [0.3, 0.4, 0.2, 0.15]},
            {"class": 1, "class_name": "Pedestrian", "bbox": [0.6, 0.5, 0.05, 0.1]},
            {"class": 2, "class_name": "Cyclist",    "bbox": [0.1, 0.7, 0.08, 0.12]},
        ])

    return images, labels


if __name__ == "__main__":
    # Test rapid să vedem că merge
    images, labels = get_mock_batch(batch_size=2)
    print("Imagini shape:", images.shape)
    print("Etichete batch 0:", labels[0])
    print("Mock dataloader OK!")