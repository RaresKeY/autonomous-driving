# Ground Truth Read TODO

Project root: `/home/mintmainog/workspace/vs_code_workspace/autonomous_driving`

Ground-truth rule: document `specs/` from repository evidence (code/docs).
Template rule: treat `specs_templates/` as optional inspiration for organization/coverage, not as project evidence.
Organization rule: shape `specs/` as a modular map of the actual project (split by logical area/component when useful).

## 1) Repo Top-Level (ls)
Use this first before deep reads.

- [ ] `specs/`

## 2) Documentation Files
Read all docs before writing/updating specs.

- [ ] `specs/_ground_truth_read_todo.md`
- [ ] `specs/_readme.md`
- [ ] `specs/kitti_dataset_preparation.md`
- [ ] `specs/model_training_pipeline.md`
- [ ] `specs/project_overview.md`
- [ ] `specs/realtime_inference_and_demo.md`
- [ ] `specs/team_lead_contracts.md`

Legacy docs removed after migration to `specs/` (2026-02-25):
- `building_realtime.md`
- `dataset.md`
- `define.md`
- `final_task.md` (migrated into `specs/realtime_inference_and_demo.md`)
- `roles.md`

## 3) Code & Config Files
Read implementation and configuration to establish ground truth.

- [ ] (No code/config files detected)

## 4) Other Files (Optional)
Review if needed for architecture, behavior, or operations context.

- [ ] (No additional files detected)

## Iteration Loop
- [ ] Update `specs/` from currently read evidence.
- [ ] Keep `specs/` file layout aligned to real project areas/modules (not template file names by default).
- [ ] Update `specs/_readme.md` so it indexes the actual ground-truth spec files.
- [ ] Re-scan repo for missed/new files.
- [ ] Read newly discovered files and update specs again.
- [ ] Repeat until no unread relevant files remain.
- [ ] Final pass: confirm each specs statement maps to code/docs evidence.
