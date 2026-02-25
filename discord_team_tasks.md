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

- Scop: demo video cu detectii (`Car`, `Pedestrian`, `Cyclist`) pentru validare vizuala end-to-end.
- Livrabile minime:
  - `detect_objects(...)` -> returneaza `list` de detectii: `(class_name, confidence, (x1, y1, x2, y2))`
  - `draw_detections(frame, detections, ...)` -> returneaza frame-ul anotat
  - `process_video(...)` -> proceseaza toate frame-urile, scrie video output si elibereaza resursele (`VideoCapture`/`VideoWriter`)
- Contract practic:
  - bbox-urile trebuie denormalizate la dimensiunea frame-ului original
  - aplicati prag de incredere (`conf_threshold`)
  - outputul trebuie sa fie stabil/usor de folosit de demo
  - tintiti rulare in timp real (ideal CPU `~10+ FPS`); daca nu se atinge, raportati FPS-ul real masurat + blocajele
- Return/exit:
  - functiile returneaza structuri utile (vezi mai sus); daca exista `main()`, return `0` la succes, non-zero la eroare
- Test mock tinta: `tests/inference_tests.py`

## Anca (`src/training.py`)

- Scop: pipeline de training reproducibil, apelabil din CLI si reutilizabil la integrare.
- Livrabile minime:
  - `parse_args()` si `main()`
  - o functie entrypoint de training: `train_model()` / `train()` / `run_training()`
  - o functie de construire model: `build_detection_model()` / `build_model()` / `create_model()`
- Contract practic:
  - `main()` trebuie sa apeleze entrypoint-ul de training cu args/config
  - entrypoint-ul de training trebuie sa accepte parametri de configurare (dataset path, epochs, output model etc.)
  - pe succes, `main()` returneaza `0`/`None`; pe eroare returneaza non-zero (fara crash necontrolat)
  - recomandat: entrypoint-ul sa returneze info despre artefacte (ex: `best_model`, `final_model`)
- De ce: Team Lead + inference au nevoie de un punct de intrare clar si de artefacte predictibile.
- Test mock tinta: `tests/training_tests.py`

## Mihaela (`src/parse_kitti_label.py`)

- Scop (intentia rolului): parsare/traversare KITTI pentru consum de training.
- Atentie (mock tests curente): `tests/parse_tests.py` verifica in prezent comportament de download/selectie componente pe `src/parse_kitti_label.py`.
- Livrabile minime (conform mock tests curente):
  - `COMPONENTS` cu optiuni de selectie (minim `images`, `labels`)
  - `parse_args()` cu suport pentru `--output-dir`, `--components`, `--no-extract`
  - `main()` care descarca doar componentele selectate, optional extrage, verifica directoarele si returneaza:
    - `0` la succes
    - non-zero la eroare (ex: download failure)
- De ce: permite download partial/repetabil pentru setup rapid.
- Nota: daca aliniati rolul la semantica reala a fisierului (`parse_kitti_label.py` = parsare), puteti modifica testele si documentati motivul.
- Test mock tinta: `tests/parse_tests.py`

## Claudia (`src/download.py`)

- Scop (intentia rolului): download KITTI complet/partial, configurabil.
- Atentie (mock tests curente): `tests/download_tests.py` verifica in prezent traversare dataset pe `src/download.py`.
- Livrabile minime (conform mock tests curente):
  - `parse_args()` si `main()`
  - o functie de traversare (ex: `iter_kitti_samples()` / `iter_kitti_pairs()` / similar)
  - traversarea trebuie sa:
    - scaneze deterministic si sortat dupa sample id
    - ignore imagini fara label pereche
    - suporte mod partial (`limit` / subset)
  - `main()` trebuie sa ruleze fara erori cu traversal mock-uit si sa returneze `0`/`None` la succes
- De ce: training/parsing au nevoie de listare stabila a sample-urilor pereche.
- Nota: daca aliniati rolul la semantica reala a fisierului (`download.py` = download), puteti modifica testele si documentati motivul.
- Test mock tinta: `tests/download_tests.py`
