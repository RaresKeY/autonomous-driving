# Mesaj Discord - Taskuri Echipa

## Toti (Obligatoriu, intai)

- Acceptati invitatia GitHub la repository.
- Verificati setup-ul Git/GitHub si faceti `pull` la ultima versiune.
- Faceti push la un fisier de test (ex: `check/<nume>_access_check.txt`) ca sa confirmati accesul.
- Scripturile temporare / experimentele care NU intra in pipeline-ul real se pun in `scripts/`.
- Daca mock testele nu se potrivesc cu implementarea valida, le puteti modifica si notati motivul in PR/commit.

## Rares (Team Lead / Integrator)

- Confirma ca toti au acces GitHub si pot face push.
- Urmareste cine a facut access-check push.
- Deblocheaza probleme de integrare si aproba update-uri la teste cand sunt justificate.

## Paul (`src/inference.py`)

- Scop: demo video cu detectii (`Car`, `Pedestrian`, `Cyclist`) pentru validare end-to-end.
- Minim:
  - `detect_objects(...)` -> `list[(class_name, confidence, (x1,y1,x2,y2))]`
  - `draw_detections(...)` -> returneaza frame anotat
  - `process_video(...)` -> proceseaza toate frame-urile, scrie output, elibereaza `VideoCapture`/`VideoWriter`
- Cerinte:
  - bbox denormalizat la frame original
  - prag `conf_threshold`
  - output stabil/usor de folosit
  - target real-time (ideal CPU `~10+ FPS`); daca nu, raportati FPS real + blocaje
- Daca exista `main()`: `0` la succes, non-zero la eroare
- Test mock tinta: `tests/inference_tests.py`

## Anca (`src/training.py`)

- Scop: pipeline de training reproducibil, apelabil din CLI si reutilizabil la integrare.
- Minim:
  - `parse_args()` si `main()`
  - o functie entrypoint de training: `train_model()` / `train()` / `run_training()`
  - o functie de construire model: `build_detection_model()` / `build_model()` / `create_model()`
- Cerinte:
  - `main()` apeleaza entrypoint-ul de training cu args/config
  - entrypoint-ul accepta config (dataset path, epochs, output model etc.)
  - `main()` -> `0`/`None` la succes, non-zero la eroare (fara crash necontrolat)
  - recomandat: return info artefacte (`best_model`, `final_model`)
- Test mock tinta: `tests/training_tests.py`

## Mihaela (`src/parse_kitti_label.py`)

- Scop: parsare/traversare KITTI pentru consum de training.
- Minim (conform `tests/parse_tests.py` curent): `parse_args()`, `main()`, functie de traversare (ex: `iter_kitti_samples()`).
- Cerinte:
  - traversare determinista/sortata dupa sample id
  - ignora imagini fara label pereche
  - suport mod partial (`limit`/subset)
  - `main()` -> `0`/`None` la succes, non-zero la eroare
- Test mock tinta: `tests/parse_tests.py`

## Claudia (`src/download.py`)

- Scop: download KITTI complet/partial, configurabil.
- Minim (conform `tests/download_tests.py` curent):
  - `COMPONENTS` (`images`, `labels`)
  - `parse_args()` cu `--output-dir`, `--components`, `--no-extract`
  - `main()` cu download partial + extract optional + verificare directoare
  - `main()` -> `0` la succes, non-zero la eroare (download/verify)
- Test mock tinta: `tests/download_tests.py`
