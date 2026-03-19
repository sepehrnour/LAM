# Roadmap: LAM Animatable Gaussian Avatar Pipeline

## Overview

Transform LAM's static Gaussian splat export into an animatable avatar bundle that Muse/OmoteRuntime can drive in real-time. The pipeline extracts per-Gaussian expression deltas from FLAME's upsampled shapedirs, pre-multiplies them into 52 ARKit blendshape deltas, and bundles them alongside the existing SPZ and a GLB mesh fallback — all from a single inference pass.

## Domain Expertise

None (specialized 3D/ML pipeline — context gathered via design review team)

## Design Review Findings (2026-03-19)

Architecture validated by 3-specialist team (FLAME/3DGS, Muse integration, systems engineering).

**Key decisions:**
1. **Deformation basis**: 34 unilateral AUs extracted from `shapedirs_up` in canonical space (composes linearly)
2. **Export format**: 52 ARKit deltas (pre-multiplied from AU basis offline). 8.3MB fp16. Zero Muse SDK changes — same `FrameOutput.blendshapes` drives both GLB and Gaussian renderers
3. **Delta extraction**: From `shapedirs_up[:, :, n_shape:]` directly (not `animation_forward`). Canonical space deltas + client-side LBS for bone transforms
4. **Jaw rotation**: `bone_bindings` metadata in manifest maps `jawOpen` → jaw bone axis-angle. Client applies proportional rotation
5. **Buffer layout**: Gaussian-major `[V × 52 × vec4]` fp16 for WebGPU cache coherence
6. **Position-only deltas**: Correct — matches LAM's own animation (rotation/scale/SH stay canonical)
7. **VRAM**: AU delta generation adds <10MB and <10ms to export. Fits trivially

**Deferred to future milestones:**
- Multi-view input (architecture change + retraining)
- Dual-head A2E model (AU + ARKit output)
- AU expansion to 20 in Muse SDK (separate repo)
- Replace Blender GLB with pure-Python glTF writer
- Higher FLAME expression dims (47-99) quality tuning

## Phases

- [ ] **Phase 1: AU Deformation Basis + ARKit Delta Export** — Extract 34 unilateral AU deltas from shapedirs_up, pre-multiply to 52 ARKit deltas, serialize with manifest
- [ ] **Phase 2: Export Bundle Integration** — Wire arkit_deltas.bin + manifest + GLB into avatar bundle, update serve.py API
- [ ] **Phase 3: End-to-End Validation** — Visualize deltas, profile VRAM, document format for Muse integration

## Phase Details

### Phase 1: AU Deformation Basis + ARKit Delta Export
**Goal**: Generate 52 ARKit position deltas for 20K Gaussians from FLAME's expression shapedirs
**Depends on**: Nothing (first phase)
**Research**: Likely (need to verify ARKit→FLAME matrix coverage, validate AU isolation in shapedirs_up, confirm canonical-space delta correctness)
**Research topics**: FLAME shapedirs_up tensor layout verification, existing arkit_to_flame.json [52,100] matrix quality, unilateral AU construction from bilateral ARKit shapes, bone_bindings calibration for jaw/eye rotation
**Plans**: 3 plans

Plans:
- [ ] 01-01: Build AU→FLAME weight matrix from existing arkit_to_flame.json, construct 34 unilateral AU deltas from shapedirs_up expression portion
- [ ] 01-02: Pre-multiply 34 AU deltas → 52 ARKit deltas, serialize as fp16 binary (Gaussian-major vec4 layout), generate au_manifest.json with bone_bindings
- [ ] 01-03: Visual validation — render each ARKit delta individually, compare against expected deformation, verify linear composition of multiple deltas

### Phase 2: Export Bundle Integration
**Goal**: Complete avatar bundle with Gaussian deltas + GLB from single API call
**Depends on**: Phase 1
**Research**: Unlikely (extending existing export pipeline)
**Plans**: 2 plans

Plans:
- [ ] 02-01: Wire arkit_deltas.bin + manifest into export_avatar_bundle(), update serve.py /avatar/generate response with new URLs and metadata
- [ ] 02-02: Integrate Blender GLB subprocess into bundle (parallel with delta generation), handle Blender unavailability gracefully, add GPU request serialization lock

### Phase 3: End-to-End Validation
**Goal**: Verified, documented, production-ready export pipeline
**Depends on**: Phase 2
**Research**: Unlikely (testing and documentation)
**Plans**: 2 plans

Plans:
- [ ] 03-01: Python visualizer — load SPZ + arkit_deltas.bin, apply blendshape weights interactively, render with Gaussian splatting, test AU combinations
- [ ] 03-02: VRAM profiling of full pipeline, format spec documentation for Muse integration (buffer layout, manifest schema, bone_bindings protocol), test with diverse portrait images

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. AU Deformation Basis + ARKit Delta Export | 0/3 | Not started | - |
| 2. Export Bundle Integration | 0/2 | Not started | - |
| 3. End-to-End Validation | 0/2 | Not started | - |
