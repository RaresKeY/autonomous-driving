# Testing Role Contracts And Modular Layout (Ground Truth)

## Purpose

This spec documents the repository's role-based mock test suite structure, including the top-level role test entry files and the modularized component contract checks under `tests/role_contracts/`.

## Current Test Layout (Observed)

- Top-level role test entry files remain in `tests/`:
  - `tests/inference_tests.py`
  - `tests/training_tests.py`
  - `tests/parse_tests.py`
  - `tests/download_tests.py`
- These top-level files are thin wrappers that import and run the real contract checks from `tests/role_contracts/`.
- Modular role contract test implementations live under:
  - `tests/role_contracts/inference_contract_checks.py`
  - `tests/role_contracts/training_contract_checks.py`
  - `tests/role_contracts/parse_contract_checks.py`
  - `tests/role_contracts/download_contract_checks.py`
- Shared fixtures/helpers remain in `tests/conftest.py`.
- `tests/conftest.py` also provides `sample_kitti_tree`, which generates aligned synthetic KITTI-style paired image/label mock data for traversal tests (including an orphan-image edge case).
- `tests/zz_all_roles_tests.py` provides meta-tests for test-file presence, mapping consistency, and test layout expectations.

## Why The Modular Layout Exists

- Keeps per-role contracts isolated and easier to extend as new role asks are added.
- Preserves the original top-level role test filenames used in team communication and specs.
- Allows future split of role contracts into smaller components without changing the public role-test entry filenames.

## Role Contract Coverage (Current)

### Paul (`tests/inference_tests.py` -> `tests/role_contracts/inference_contract_checks.py`)

- Verifies inference module exposes `detect_objects`, `draw_detections`, `process_video`.
- Verifies `detect_objects(...)` returns normalized detection tuples and denormalized bbox coordinates.
- Verifies `process_video(...)`:
  - processes frames and writes output
  - releases OpenCV resources
  - returns a runtime summary dict with FPS and processed-frame metrics (to support real-time validation)

### Anca (`tests/training_tests.py` -> `tests/role_contracts/training_contract_checks.py`)

- Verifies training module exposes CLI entrypoints and model-builder/training entrypoints.
- Verifies `main()` calls the training entrypoint, returns success/non-zero codes appropriately, and handles exceptions.
- Verifies training entrypoint signature accepts config-like inputs.
- Accepts either CLI-style success return (`0`/`None`) or a summary dict (artifact metadata) from `main()`.

### Mihaela (`tests/parse_tests.py` -> `tests/role_contracts/parse_contract_checks.py`)

- Current mock-test coverage validates traversal/listing behavior on `src/parse_kitti_label.py`.
- Verifies:
  - traversal function exists
  - deterministic paired image/label traversal
  - partial/subset traversal support
  - integration-friendly traversal item shape
  - `main()` success and handled non-zero failure on traversal errors

### Claudia (`tests/download_tests.py` -> `tests/role_contracts/download_contract_checks.py`)

- Current mock-test coverage validates a download-style CLI contract on `src/download.py`.
- Verifies:
  - `COMPONENTS` selectable downloads
  - CLI args for output dir / components / no-extract
  - partial component download behavior
  - extract-skip behavior
  - non-zero returns on download failure and verify failure

## Execution Status Snapshot (2026-02-25)

- Verified in project venv (`pytest 8.4.2`)
- Historical passing snapshot (before empty-file revert): full `tests/` run passed (`23 passed`)
- Current snapshot after user-requested empty `src/*.py` placeholders: `19 failed, 4 passed`
- Failure mode is expected: contract tests require callable APIs in role-owned modules.
