# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** Generate animatable Gaussian splat avatars driven by FACS-based expression from a single image
**Current focus:** Phase 1 — AU Deformation Basis + ARKit Delta Export

## Current Position

Phase: 1 of 3 (AU Deformation Basis + ARKit Delta Export)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-03-19 — Plan 01-01 complete (delta extraction module + integration test)

Progress: █░░░░░░░░░ 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: ~30 min
- Total execution time: ~0.5 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1/3 | ~30 min | ~30 min |

**Recent Trend:**
- Last 5 plans: 01-01
- Trend: Starting

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Design review (2026-03-19): 52 ARKit deltas as export format, built from 34 unilateral AU basis offline
- Design review (2026-03-19): shapedirs_up canonical-space extraction (not animation_forward) — composes linearly
- Design review (2026-03-19): bone_bindings manifest metadata for jaw rotation — additive with position deltas

### Deferred Issues

- Multi-view input (separate milestone — architecture change + retraining)
- Dual-head A2E model (separate effort in Muse repo)
- FLAME dims 47-99 quality tuning (post-validation visual comparison)
- Replace Blender GLB with pure-Python glTF writer (removes 1.4GB dependency)

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-19
Stopped at: Plan 01-01 complete, ready for 01-02
Resume file: .planning/phases/01-au-deformation-basis/01-01-SUMMARY.md
