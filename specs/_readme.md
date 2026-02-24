# autonomous_driving - Specs Index

**Tech Stack**: TBD

---

**IMPORTANT** Before making changes or researching any part of the codebase, use the table below to find and read the relevant spec first. This ensures you understand existing patterns and constraints.

## Documentation

Populate this table using the template library as inspiration, not as a rigid file structure.
Scope `specs/` to modularly map the project, and split specs by logical area when needed (for example UI can be separated into overview/views/previews/components/tests files).

| Spec | Code | Purpose |
|------|------|---------|
| [`project_overview.md`](./project_overview.md) | `define.md`, `final_task.md`, repo root contents | Project scope, stated objective, current repo state, and explicit gaps. |
| [`kitti_dataset_preparation.md`](./kitti_dataset_preparation.md) | `dataset.md` | KITTI download/setup, label format, filtering rules, parsing, visualization, and dataset stats workflow. |
| [`model_training_pipeline.md`](./model_training_pipeline.md) | `building_realtime.md` (+ class definitions from `dataset.md`) | Detector architecture, data generator behavior, training split, callbacks, staged training, and model artifacts. |
| [`realtime_inference_and_demo.md`](./realtime_inference_and_demo.md) | `building_realtime.md`, `final_task.md` | Inference API, overlay rendering, video/webcam processing, demo outputs, and final-task alignment. |

## Templates

- [Specs Template Library](../specs_templates/_readme.template.md): Reusable templates kept separate from ground-truth specs.
