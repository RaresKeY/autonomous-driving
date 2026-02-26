# scripts/serve_dataset.py
# Interfata intre dataset si pipeline-ul de training
# Anca - Model Training Engineer
#
# ============================================================
# CUM SE PLUGUIESTE (pentru Mihaela):
#
# Pasul 1: Implementeaza functia `load_dataset_yaml()` de mai jos
#          ca sa returneze calea catre dataset.yaml-ul tau real.
#
# Pasul 2: Implementeaza `get_dataloader()` ca sa returneze
#          un generator/dataloader compatibil cu formatul YOLO.
#
# Pasul 3: Ruleaza acest script ca sa verifici ca datele
#          sunt servite corect inainte de training:
#          python scripts/serve_dataset.py
#
# Pasul 4: Training-ul se face cu:
#          python src/training.py --dataset <calea returnata de load_dataset_yaml()>
# ============================================================

import sys
from pathlib import Path
import yaml
import torch

# Adauga root-ul proiectului in path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# SECTIUNEA PENTRU MIHAELA
# Modifica functiile de mai jos cu implementarea ta reala
# ============================================================

def load_dataset_yaml(dataset_root=None):
    """
    Returneaza calea catre dataset.yaml pentru training.

    MIHAELA: Inlocuieste aceasta functie cu calea reala catre
             dataset.yaml-ul generat de tine.

    Args:
        dataset_root: folderul radacina al datasetului (optional)

    Returns:
        Path catre dataset.yaml
    """
    if dataset_root is not None:
        yaml_path = Path(dataset_root) / "dataset.yaml"
        if yaml_path.exists():
            return yaml_path

    # Locatii default unde cautam dataset.yaml
    default_locations = [
        Path("C:/Curs Python/datasets/kitti_yolo/dataset.yaml"),
        Path("datasets/kitti_yolo/dataset.yaml"),
        Path("data/dataset.yaml"),
    ]

    for path in default_locations:
        if path.exists():
            print(f"[serve_dataset] Gasit dataset.yaml la: {path}")
            return path

    print("[serve_dataset] ATENTIE: dataset.yaml nu a fost gasit.")
    print("               Ruleaza prepare_kitti.py sau furnizeaza dataset_root.")
    return None


def get_dataloader(split="train", batch_size=4, img_size=640):
    """
    Returneaza un batch de date pentru training.

    MIHAELA: Inlocuieste aceasta functie cu dataloader-ul tau real.
             Formatul returnat trebuie sa fie:
             - images: torch.Tensor de shape (batch_size, 3, img_size, img_size)
             - labels: lista de dict cu keys 'class' si 'bbox'

    Args:
        split: 'train', 'val', sau 'test'
        batch_size: numarul de imagini per batch
        img_size: dimensiunea imaginilor

    Returns:
        (images, labels) — batch de date
    """
    print(f"[serve_dataset] MOCK dataloader activ pentru split='{split}'")
    print(f"                Inlocuieste get_dataloader() cu implementarea Mihaelei!")

    # Mock data — imagini random + etichete fictive
    images = torch.rand(batch_size, 3, img_size, img_size)
    labels = []
    for _ in range(batch_size):
        labels.append([
            {"class": 0, "class_name": "Car",        "bbox": [0.3, 0.4, 0.2, 0.15]},
            {"class": 1, "class_name": "Pedestrian", "bbox": [0.6, 0.5, 0.05, 0.1]},
            {"class": 2, "class_name": "Cyclist",    "bbox": [0.1, 0.7, 0.08, 0.12]},
        ])

    return images, labels


# ============================================================
# SECTIUNEA ANCA — nu modifica
# Verifica ca datele sunt servite corect
# ============================================================

def verify_dataset_yaml(yaml_path):
    """Verifica ca dataset.yaml are structura corecta pentru YOLO."""
    print(f"\n[verify] Verific {yaml_path}...")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    errors = []

    if "nc" not in cfg:
        errors.append("Lipseste 'nc' (number of classes)")
    elif cfg["nc"] != 3:
        errors.append(f"'nc' trebuie sa fie 3, e {cfg['nc']}")

    if "names" not in cfg:
        errors.append("Lipseste 'names'")
    else:
        expected = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        for idx, name in expected.items():
            if cfg["names"].get(idx) != name:
                errors.append(f"names[{idx}] trebuie sa fie '{name}', e '{cfg['names'].get(idx)}'")

    for split in ["train", "val"]:
        if split not in cfg:
            errors.append(f"Lipseste sectiunea '{split}'")

    if errors:
        print("  ERORI gasite:")
        for e in errors:
            print(f"    ❌ {e}")
        return False
    else:
        print("  ✅ dataset.yaml valid!")
        print(f"     Clase: {cfg['names']}")
        print(f"     Train: {cfg['train']}")
        print(f"     Val:   {cfg['val']}")
        return True


def verify_dataloader():
    """Verifica ca dataloader-ul returneaza formatul corect."""
    print("\n[verify] Verific dataloader-ul...")

    images, labels = get_dataloader(split="train", batch_size=4)

    errors = []

    if not isinstance(images, torch.Tensor):
        errors.append(f"images trebuie sa fie torch.Tensor, e {type(images)}")
    elif images.shape != (4, 3, 640, 640):
        errors.append(f"images.shape trebuie sa fie (4, 3, 640, 640), e {images.shape}")

    if len(labels) != 4:
        errors.append(f"labels trebuie sa aiba 4 elemente, are {len(labels)}")
    else:
        for obj in labels[0]:
            if "class" not in obj:
                errors.append("Fiecare obiect trebuie sa aiba cheia 'class'")
            if "bbox" not in obj:
                errors.append("Fiecare obiect trebuie sa aiba cheia 'bbox'")

    if errors:
        print("  ERORI gasite:")
        for e in errors:
            print(f"    ❌ {e}")
        return False
    else:
        print(f"  ✅ Dataloader valid!")
        print(f"     images.shape: {images.shape}")
        print(f"     labels[0]: {labels[0]}")
        return True


if __name__ == "__main__":
    print("=" * 50)
    print("SERVE DATASET - Verificare interfata")
    print("=" * 50)

    # 1. Verifica dataloader
    dl_ok = verify_dataloader()

    # 2. Verifica dataset.yaml daca exista
    yaml_path = load_dataset_yaml()
    if yaml_path:
        yaml_ok = verify_dataset_yaml(yaml_path)
    else:
        print("\n[verify] dataset.yaml nu exista inca — normal daca Claudia")
        print("         nu a trimis datele. Mock dataloader activ.")
        yaml_ok = True

    # Concluzie
    print("\n" + "=" * 50)
    if dl_ok and yaml_ok:
        print("✅ Interfata OK — gata de conectat cu src/training.py")
        print("\nPentru training ruleaza:")
        if yaml_path:
            print(f'   python src/training.py --dataset "{yaml_path}"')
        else:
            print("   python src/training.py --dataset <calea ta catre dataset.yaml>")
    else:
        print("❌ Interfata are erori — verifica mesajele de mai sus")
    print("=" * 50)