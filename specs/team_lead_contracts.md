# Team Lead Contracts And Parallel Workstream Interfaces (Ground Truth / Planning)

## Purpose

This spec captures the Team Lead / System Integrator contracts needed to let five assigned roles work in parallel with minimal blocking. It translates `roles.md` into explicit handoff interfaces, acceptance criteria, and integration cadence.

Current planning assumption for this revision:
- team coordination/setup is timeboxed to `2 hours` (focus on contracts, interfaces, and startup tasks rather than full implementation completion)

Source:
- `roles.md`
- `specs/team_lead_interface_v0_1.md`

## Team Assignments (Current)

- Team Lead / System Integrator: `Rares`
- Dataset Collection & Curation Engineer: `Claudia`
- Data Parsing / Preprocessing / Generator Engineer: `Mihaela`
- Model Training Engineer: `Anca`
- Inference / Demo / Evaluation Engineer: `Paul`

Source:
- `roles.md`

## Team Lead Scope (Contract Owner)

The Team Lead owns:

- interface definitions between roles
- milestone sequencing and deadlines
- merge / integration decisions
- acceptance criteria for each handoff
- change-control decisions when interfaces must change

Source:
- `roles.md`

## Concrete Interface Specification

The concrete kickoff schemas, example payloads, and function signatures are defined in:

- `specs/team_lead_interface_v0_1.md`

This contracts spec remains the higher-level ownership/handoff/acceptance document.

## Contract 1: Dataset Collection -> Downstream Consumers

Owner:
- Producer: Dataset Collection & Curation Engineer (`Claudia`)
- Approver: Team Lead (`Rares`)
- Consumers: Data Parsing Engineer (`Mihaela`), Model Training Engineer (`Anca`)

### Required Deliverables

- Dataset root layout documented and consistent (KITTI paths and any added data paths)
- Data manifest file (image counts, label counts, source URLs, license notes)
- Split definitions (`train`, `val`, optional `test`) as stable file lists / IDs
- Integrity verification report (missing files, corrupted files, count mismatches)

### Interface Contract

- Primary classes for project scope must be explicitly listed: `Car`, `Pedestrian`, `Cyclist`
- Every split file must be reproducible and versioned (no ad-hoc local-only split changes)
- File naming/ID format must be stable so parsers/trainers can reference the same samples
- Any non-KITTI additions must include source and label format notes

### Acceptance Criteria (Lead Sign-off)

- Counts are reported and internally consistent
- Split files are present and readable
- Class coverage summary exists
- License/source provenance is documented

## Contract 2: Data Parsing / Preprocessing -> Model Training

Owner:
- Producer: Data Parsing / Preprocessing / Generator Engineer (`Mihaela`)
- Approver: Team Lead (`Rares`)
- Consumer: Model Training Engineer (`Anca`)

### Required Deliverables

- KITTI label parsing implementation (`parse_kitti_label` behavior documented)
- Filtering rules (occlusion, truncation, min box height, allowed classes)
- Preprocessing config (resize target, normalization, augmentations if used)
- Training dataloader/generator module with stable batch output format
- Visualization/QA outputs (sample overlays + class distribution summary)

### Interface Contract (Training Input)

- Batch images shape and dtype are fixed and documented (for example `B x H x W x C`, normalized range)
- Class labels are encoded consistently and versioned
- Bounding boxes are normalized/denormalized rules documented
- Empty-object behavior is defined (default class / zero bbox or equivalent)
- Generator must produce deterministic output when seed is fixed

### Acceptance Criteria (Lead Sign-off)

- Trainer can consume one batch without code changes to parser internals
- QA examples visually confirm bbox correctness on sample images
- Parsing/filtering rules are written in spec or config, not only in code

## Contract 3: Model Training -> Inference / Demo

Owner:
- Producer: Model Training Engineer (`Anca`)
- Approver: Team Lead (`Rares`)
- Consumer: Inference / Demo Engineer (`Paul`)

### Required Deliverables

- Training script/module and run configuration
- Saved model artifacts (`best` and `final`)
- Label/class mapping used during training
- Metrics summary (classification + bbox regression metrics)
- Inference-facing model I/O contract

### Interface Contract (Model Output)

- Model input size is fixed and documented (current tutorial baseline: `224x224`)
- Output names and shapes are fixed (class probabilities + bbox coordinates)
- Bounding box coordinate convention is documented (`[x1, y1, x2, y2]`, normalized range)
- Confidence thresholding expectations are documented for inference
- Model artifact version and training config version are recorded together

### Acceptance Criteria (Lead Sign-off)

- Inference engineer can load the delivered model and run a test prediction
- Class index mapping matches overlay labels exactly
- Artifact filenames and versions are unambiguous

## Contract 4: Inference / Demo -> Integration Demo Review

Owner:
- Producer: Inference / Demo / Evaluation Engineer (`Paul`)
- Approver: Team Lead (`Rares`)

### Required Deliverables

- Inference API module/function contract
- Image/video/webcam processing scripts
- Overlay rendering behavior (colors, labels, confidence display)
- Demo outputs (sample images/videos)
- Runtime performance report (FPS and environment notes)

### Interface Contract (Inference API)

- Inference function signature is stable and documented
- Return format is explicit (class name, confidence, bbox tuple)
- Confidence threshold is configurable
- Video processing output path behavior is documented
- Error handling behavior is defined for missing model/video source

### Acceptance Criteria (Lead Sign-off)

- A sample video can be processed end-to-end
- Overlay labels map to training classes without mismatch
- Performance report includes hardware/context notes

## Contract 5: Integration Governance (Lead-Owned)

This contract is owned entirely by the Team Lead and applies to all roles.

### Rules

- Interfaces are frozen at the start of each sprint/phase unless an approved change request is logged
- Teams can implement against mocks first (dummy generator, dummy model)
- For the current `2-hour` kickoff scope, integration happens in a single midpoint sync and a final wrap-up sync within the same session
- Interface changes require:
- version bump (even if minor)
- written changelog note
- consumer acknowledgment before merge

### Minimum Artifacts Maintained By Lead

- Integration checklist
- Interface version table
- Open risks / blockers list
- Milestone status board

## Initial Milestone Sequence (Parallel-Friendly)

1. `0:00-0:20` Team Lead publishes interface v0.1 contracts and acceptance checklist.
2. `0:20-1:10` Dataset, preprocessing, training, and inference roles work in parallel on startup artifacts / stubs (using mocks where needed).
3. `1:10-1:30` Midpoint sync:
   - confirm dataset manifest/split format
   - confirm dataloader batch schema
   - confirm model I/O contract for inference
4. `1:30-2:00` Final wrap-up:
   - document blockers and next-step handoffs
   - freeze interface v0.1
   - record owner-specific follow-up tasks for post-timebox implementation

## Notes

- This spec is a project coordination contract derived from current `roles.md`, not a code-implemented runtime API.
- The `2-hour` scope covers planning/setup and interface alignment only; it does not imply full model training or demo completion within the same session.
- As code modules are created, this spec should be updated to point to exact file paths and function signatures.
