# Separate Role Test Suite (Ground Truth)

## Purpose

This spec documents the separate pytest-based mock/contract tests for each role, plus the extra final "run all tests" checks.

It is the canonical test-structure spec for the current role implementation workflow.

## Test Layout (Separate Files)

Per-role test files:

- `tests/inference_tests.py` (Paul / `src/inference.py`)
- `tests/training_tests.py` (Anca / `src/training.py`)
- `tests/parse_tests.py` (Mihaela / `src/parse_kitti_label.py`)
- `tests/download_tests.py` (Claudia / `src/download.py`)

Extra final test file:

- `tests/zz_all_roles_tests.py` (meta checks; named to run last alphabetically)

Shared test utilities:

- `tests/conftest.py`

## Discovery / Runner Configuration

- `pytest.ini` enables pytest discovery for `*_tests.py`
- `testpaths = tests`
- `addopts = -ra`

This allows both per-file test execution and a final all-tests execution without renaming files to pytest defaults.

## Coverage Contract By Role

### `tests/inference_tests.py`

Validates `src/inference.py` contract behavior using mocks/stubs:

- required callables exist: `detect_objects`, `draw_detections`, `process_video`
- detection output format and bbox denormalization behavior
- video processing smoke flow with mocked `cv2` capture/writer

### `tests/training_tests.py`

Validates `src/training.py` CLI/entrypoint contract using mocks/stubs:

- required callables exist: `parse_args`, `main`
- at least one training entrypoint exists (`train_model` / `train` / `run_training`)
- at least one model builder exists (`build_detection_model` / `build_model` / `create_model`)
- `main()` success/failure handling around training entrypoint

### `tests/parse_tests.py`

Validates `src/parse_kitti_label.py` download-role contract (current assignment) using mocks:

- `COMPONENTS` supports partial download selection (`images`, `labels`)
- `parse_args()` supports `--output-dir`, `--components`, `--no-extract`
- `main()` orchestrates selected-component downloads/extraction/verification
- `main()` returns non-zero on download failure

### `tests/download_tests.py`

Validates `src/download.py` dataset traversal contract (current assignment) using a temporary KITTI-like tree:

- traversal function exists (accepted candidate names)
- complete traversal returns deterministic sample IDs and ignores orphan image files
- partial/subset traversal mode is supported
- `main()` smoke path works with mocked traversal

## Final "Run All" Step

After per-role mock tests pass, run a full test pass as an extra final check.

The meta test file `tests/zz_all_roles_tests.py` verifies:

- separate test files exist (no collapsed single-file test layout)
- `pytest.ini` supports `*_tests.py` discovery
- role-to-target mapping matches current role assignment spec

## Execution Pattern

Per-role tests (during implementation):

- `python3 -m pytest tests/inference_tests.py`
- `python3 -m pytest tests/training_tests.py`
- `python3 -m pytest tests/parse_tests.py`
- `python3 -m pytest tests/download_tests.py`

Final extra check:

- `python3 -m pytest tests`

## Notes

- These are contract/mock tests and are expected to fail until the corresponding `src/*.py` role files exist.
- `pytest` is declared in `requirements.txt`.
