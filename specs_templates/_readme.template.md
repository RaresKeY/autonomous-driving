# <project_name> - Specs Index

**Tech Stack**: <language/framework/runtime>

---

**IMPORTANT** Before making changes or researching any part of the codebase, use the table below to find and read the relevant spec first. This ensures you understand existing patterns and constraints.

## How To Use This Template Library

- Use these templates as inspiration and a starting point, not as a structure to copy verbatim into `specs/`.
- Adapt the number and names of spec files to the actual project. Keep only what is useful and split documents where the project needs more precision.
- The goal of `specs/` is to modularly map project specifications so each area is logically separated and easy to maintain.

### Example Modular Scope (UI)

Depending on the project, UI documentation may be split into focused files such as:

- `ui_overview.md`
- `ui_views.md`
- `ui_previews.md`
- `ui_components.md`
- `ui_tests.md`

## Documentation

| Spec | Code | Purpose |
|------|------|---------|
| [Product Context](product_context.template.md) | `<app entrypoints>` | Product goals, personas, constraints, and non-goals. |
| [System Architecture](system_architecture.template.md) | `<root modules>` | Entry point, module boundaries, and runtime data flow. |
| [Domain Logic](domain_logic.template.md) | `<core domain files>` | Business/game/domain rules and invariants. |
| [Data Model](data_model.template.md) | `<schema/models>` | Data entities, relationships, storage format, and migrations. |
| [Quality & Testing](quality_testing.template.md) | `<tests/, ci, tooling>` | Test strategy, quality gates, and release confidence checks. |
| [Operational Notes](operational_notes.template.md) | `<deploy/runtime config>` | Environment setup, observability, and runbook notes. |
