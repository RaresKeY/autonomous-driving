# Operations: Build / Release / Commit Workflow (Ground Truth)

## Purpose

This spec defines the repository commit workflow policy referenced by `AGENTS.md`.

Scope in current repo state:

- commit-message drafting from staged changes
- staged-only commits
- signed commit behavior (when local git/GPG config requires signing)
- basic release-operation notes (tags/releases remain a later step)

## Commit Workflow Policy

### 1. Source of Truth for Commit Scope

- Commit requests should operate on the **currently staged changes** only.
- Do not include unstaged/untracked changes unless the user explicitly asks to stage them.
- Validate staged state before committing.

### 2. Message Derivation

- Derive the commit message from the staged diff (files/stat/hunks), not from assumptions.
- Prefer Conventional Commit style (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Use `chore:` when staged changes are intentionally mixed (for example docs + tests + scaffolding).
- Preserve user intent wording when it matches the staged diff facts.

### 3. Commit Execution Rules

- Prefer non-interactive commit commands (for example `git commit -m ...`).
- Do not amend commits unless explicitly requested.
- Do not run destructive git commands (`reset --hard`, forced checkout/revert patterns) unless explicitly requested.
- If git is configured for commit signing, keep signing enabled unless the user asks otherwise.

### 4. Signed Commit / Permission Handling

- If `git commit` fails in sandbox due to GPG agent / keyring permissions, retry with elevated execution.
- Keep the same commit message and staged-only scope when retrying.
- Report the commit hash and subject after success.

Observed repo behavior (2026-02-25):

- local git commit signing is enabled and may require access to host GPG agent outside sandbox.

### 5. What To Report Back After Commit

- commit hash
- final subject line
- whether signing/elevation was required
- concise summary of staged scope (for example file count / high-level areas)

## Release / Tagging Notes (Current)

- Tag/release workflow is not yet fully specified in this file.
- If/when release tagging is requested, extend this spec with:
  - versioning policy (SemVer)
  - annotated tag requirements
  - push/verification steps

## Related Specs / Inputs

- `AGENTS.md` (references this file)
- `specs/current_role_implementation_assignments.md` (team workflow context)
- `specs/testing_role_contracts.md` (test expectations may affect commit scope)
