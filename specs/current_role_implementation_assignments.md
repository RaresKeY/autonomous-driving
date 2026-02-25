# Current Role Implementation Assignments (Reality Override)

## Purpose

This document captures the current execution reality for team assignments (exact script ownership and mock-test expectations). It refines the broader planning contracts in `specs/team_lead_contracts.md`.

Primary teammate-facing task wording source for this revision: `discord_team_tasks.md`.

Date captured: `2026-02-25`

## Team-Wide GitHub Onboarding / Access Check (User Instruction, 2026-02-25)

Before role-specific implementation work begins, all team members are expected to:

- accept the GitHub repository invite
- ensure local Git/GitHub setup is working
- pull the latest repository state
- push a small test file to confirm write access (access-check validation)

## Test Alignment Rule (User Instruction, 2026-02-25)

If mock tests do not align with validated implementation findings, the tests may be modified. Changes should be accompanied by a short reason (for example in commit/PR notes) so the team can track why the mock expectations changed.

## Temporary Script Placement Rule (User Instruction, 2026-02-25)

Temporary scripts, experiments, or one-off helpers that are not part of the actual pipeline contract should be placed under `scripts/` rather than `src/`. The `src/` directory should remain focused on role-owned pipeline modules and integration-facing code.

## Deliverable Structure Guidance (User Instruction, 2026-02-25)

The team requested role tasks to include concrete deliverable structure details (what scripts must do, what they should return, and why). The guidance below should be used for assignment messages and implementation planning.

This section is aligned to the current `discord_team_tasks.md` content (2026-02-25).

### Paul (`src/inference.py`)

- Purpose: provide an end-to-end demo path for object-detection overlays on video/webcam inputs.
- Expected callable surface (from current mock-test contract):
  - `detect_objects(...)` -> returns `list` of detections `(class_name, confidence, (x1, y1, x2, y2))`
  - `draw_detections(...)` -> returns annotated frame/image
  - `process_video(...)` -> processes frames, writes output video, releases OpenCV resources
- Real-time expectation (user instruction refinement, 2026-02-25):
  - target practical real-time behavior for demo use (ideally ~`10+ FPS` on CPU when feasible)
  - if target is not met, report measured FPS and main bottlenecks instead of silently degrading
- Why this shape: it gives a stable integration contract for rendering + demo automation and is easy to mock in tests.

### Anca (`src/training.py`)

- Purpose: expose a CLI-usable and integration-friendly training pipeline entrypoint.
- Expected callable surface (from current mock-test contract):
  - `parse_args()`
  - `main()` (returns `0`/`None` on success, non-zero on handled failure)
  - one training entrypoint: `train_model()` / `train()` / `run_training()`
  - one model builder: `build_detection_model()` / `build_model()` / `create_model()`
- Training entrypoint should accept config-like parameters (dataset path / epochs / output path etc.).
- Why this shape: Team Lead and inference integration need reproducible training invocation and predictable artifact handoff.

### Mihaela (`src/parse_kitti_label.py`)

- Intended and current mock-test semantics: KITTI parsing/traversal support for downstream training preprocessing.
- Current mock-test contract (`tests/parse_tests.py`) validates:
  - `parse_args()`, `main()`
  - traversal/listing function (for example `iter_kitti_samples()`)
  - deterministic sorted paired image+label traversal
  - partial/subset mode support
  - handled non-zero return on traversal failure

### Claudia (`src/download.py`)

- Intended and current mock-test semantics: KITTI downloader (complete/partial, configurable destination/components).
- Current mock-test contract (`tests/download_tests.py`) validates:
  - `COMPONENTS` (including `images`, `labels`)
  - `parse_args()` supporting `--output-dir`, `--components`, `--no-extract`
  - selected-component download/extract flow
  - handled non-zero returns on download/verify failures

## Role-to-Implementation Mapping (Current)

| Person | Responsibility (as stated) | Target File | Mock Test Gate |
|---|---|---|---|
| `Paul` | Script that renders bounding boxes over a video in real time using any model | `src/inference.py` | `tests/inference_tests.py` |
| `Anca` | Script that trains on KITTI using YOLO with MobileNetV2 | `src/training.py` | `tests/training_tests.py` |
| `Mihaela` | Script that traverses KITTI dataset (complete/partial) for another script (`parse_kitti_label`) | `src/parse_kitti_label.py` | `tests/parse_tests.py` |
| `Claudia` | Script that downloads KITTI complete/partial, configurable destination and what to download | `src/download.py` | `tests/download_tests.py` |

## Differences Vs `specs/team_lead_contracts.md`

- The team contracts spec is interface/handoff oriented; this doc is implementation-task oriented (exact file ownership + test gates).
- `Paul` and `Anca` match the expected high-level domains (inference, training), but now have explicit file targets and test files.
- `Mihaela` and `Claudia` are aligned with filename semantics:
  - `Mihaela` owns parsing/traversal work in `src/parse_kitti_label.py`.
  - `Claudia` owns download work in `src/download.py`.
- Acceptance criteria are now concretely "passes mock tests", which is stricter/more executable than the earlier checklist-style role contracts.

## Test Execution Rules (Current)

- Tests are separate per role:
  - `tests/inference_tests.py`
  - `tests/training_tests.py`
  - `tests/download_tests.py`
  - `tests/parse_tests.py`
- A "run all tests" step is an extra final check after the per-role mock tests pass.

## Implemented Test Suite Artifacts (Current Repo)

- `pytest.ini` enables discovery of `*_tests.py` files under `tests/`.
- `tests/conftest.py` contains shared fixtures/helpers for role contract tests.
- `tests/inference_tests.py`, `tests/training_tests.py`, `tests/parse_tests.py`, and `tests/download_tests.py` are top-level role test wrappers.
- Modular role-contract checks live under `tests/role_contracts/` and are imported by the top-level role wrappers.
- `tests/zz_all_roles_tests.py` is the extra final meta-test file (named to run last alphabetically).
- Current observed verification snapshot depends on implementation state:
  - contract-stub implementation state: full `tests/` suite can pass
  - current user-requested empty-file state: role-contract tests fail until APIs are implemented

## Naming Notes

- Correct path spelling is `tests/inference_tests.py` (not `tests/infernce_tests.py`).
