# Current Role Implementation Assignments (Reality Override)

## Purpose

This document captures the current execution reality for team assignments (exact script ownership and mock-test expectations). It refines the broader planning contracts in `specs/team_lead_contracts.md`.

Date captured: `2026-02-25`

## Role-to-Implementation Mapping (Current)

| Person | Responsibility (as stated) | Target File | Mock Test Gate |
|---|---|---|---|
| `Paul` | Script that renders bounding boxes over a video in real time using any model | `src/inference.py` | `tests/inference_tests.py` |
| `Anca` | Script that trains on KITTI using YOLO with MobileNetV2 | `src/training.py` | `tests/training_tests.py` |
| `Mihaela` | Script that traverses KITTI dataset (complete/partial) for another script (`parse_kitti_label`) | `src/download.py` | `tests/download_tests.py` |
| `Claudia` | Script that downloads KITTI complete/partial, configurable destination and what to download | `src/parse_kitti_label.py` | `tests/parse_tests.py` |

## Differences Vs `specs/team_lead_contracts.md`

- The team contracts spec is interface/handoff oriented; this doc is implementation-task oriented (exact file ownership + test gates).
- `Paul` and `Anca` match the expected high-level domains (inference, training), but now have explicit file targets and test files.
- `Mihaela` and `Claudia` responsibilities appear swapped relative to filename semantics:
  - `Mihaela` has dataset traversal/parsing-adjacent work but target file is `src/download.py`.
  - `Claudia` has downloading work but target file is `src/parse_kitti_label.py`.
- Acceptance criteria are now concretely "passes mock tests", which is stricter/more executable than the earlier checklist-style role contracts.

## Test Execution Rules (Current)

- Tests are separate per role:
  - `tests/inference_tests.py`
  - `tests/training_tests.py`
  - `tests/download_tests.py`
  - `tests/parse_tests.py`
- A "run all tests" step is an extra final check after the per-role mock tests pass.

## Naming Notes

- Correct path spelling is `tests/inference_tests.py` (not `tests/infernce_tests.py`).
