# autonomous_driving - Specs Index

**Tech Stack**: Python (TensorFlow/Keras, Ultralytics YOLO, OpenCV, NumPy, Matplotlib, Pillow, pytest)

---

**IMPORTANT** Before making changes or researching any part of the codebase, use the table below to find and read the relevant spec first. This ensures you understand existing patterns and constraints.

## Documentation

Populate this table using the template library as inspiration, not as a rigid file structure.
Scope `specs/` to modularly map the project, and split specs by logical area when needed (for example UI can be separated into overview/views/previews/components/tests files).

| Spec | Code | Purpose |
|------|------|---------|
| [`project_overview.md`](./project_overview.md) | `src/`, `tests/`, `requirements.txt`, `specs/realtime_inference_and_demo.md`, repo root contents | Project scope, stated objective, current repo state, migration status, and explicit gaps. |
| [`kitti_dataset_preparation.md`](./kitti_dataset_preparation.md) | Canonical spec (migrated tutorial dataset notes) | KITTI download/setup, label format, filtering rules, parsing, visualization, and dataset stats workflow. |
| [`model_training_pipeline.md`](./model_training_pipeline.md) | Canonical spec (migrated tutorial training notes) | Detector architecture, data generator behavior, training split, callbacks, staged training, and model artifacts. |
| [`realtime_inference_and_demo.md`](./realtime_inference_and_demo.md) | Canonical spec (migrated tutorial inference notes) + migrated final-task note | Inference API, overlay rendering, video/webcam processing, demo outputs, and final-task alignment. |
| [`team_lead_contracts.md`](./team_lead_contracts.md) | Canonical spec (migrated team role/contract notes) | Team Lead integration contracts for parallel workstreams: interfaces, handoffs, acceptance criteria, and integration cadence. |
| [`operations_build_release.md`](./operations_build_release.md) | `AGENTS.md`, git commit behavior observed in repo | Commit workflow policy (staged-only scope, message derivation, signing/elevation handling) and release/tagging notes. |
| [`current_role_implementation_assignments.md`](./current_role_implementation_assignments.md) | `discord_team_tasks.md` + user-provided role updates (2026-02-25) | Current execution reality: exact owner -> file path -> mock test acceptance mapping, plus mismatches vs planning roles. |
| [`testing_role_contracts.md`](./testing_role_contracts.md) | `tests/`, `pytest.ini`, `tests/conftest.py` | Canonical detailed test spec: role-based mock suite layout, modular `tests/role_contracts/` structure, fixture-backed mock data, and per-role contract coverage. |
| [`separate_role_test_suite.md`](./separate_role_test_suite.md) | `pytest.ini`, `tests/`, `specs/current_role_implementation_assignments.md` | Concise companion spec for separate per-role test execution and final run-all checks. |

## Migration Notes

- On 2026-02-25, legacy root docs (`dataset.md`, `define.md`, `roles.md`, `building_realtime.md`) were consolidated into `specs/` and removed.
- On 2026-02-25, the root `final_task.md` note was migrated into `specs/realtime_inference_and_demo.md` and translated to English.
- The specs above are now the canonical location for that content.
- `specs/operations_build_release.md` was added to satisfy the `AGENTS.md` commit-workflow reference.

## Templates

- [Specs Template Library](../specs_templates/_readme.template.md): Reusable templates kept separate from ground-truth specs.
