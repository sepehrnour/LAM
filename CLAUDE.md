# LAM Character Service

## What This Is
GPU microservice for the Omote character platform. Three capabilities:
1. Photo → 3D Gaussian Splatting avatar (.spz)
2. Video → FLAME motion capture (per-frame expression params)
3. Audio → FLAME expression params (lip sync)

All output FLAME parameters — the universal animation interface.

## Architecture
Photo → POST /avatar/generate → .spz bundle → S3/CDN → @omote SDK (browser/iOS)
Video → POST /motion/extract → FLAME params JSON → drive any avatar
Audio → POST /motion/from-audio → FLAME params JSON → real-time lip sync

Client-side (@omote SDK):
- Wav2Vec2 → 52 ARKit blendshapes (ONNX, WebGPU/CoreML)
- ARKit→FLAME matrix [100,52] → FLAME expression params
- FLAME LBS deformation → animate Gaussians
- Spark.js / Metal → render

## API
POST /avatar/generate — photo → .spz avatar bundle
POST /motion/extract — video → FLAME motion sequence
POST /motion/from-audio — audio → FLAME expression params
GET  /artifacts/{job_id}/{file} — download artifacts
GET  /health — service status

## Key Files
- serve.py — FastAPI service
- app_lam.py — Gradio dev UI
- lam/export/ — Export pipeline (avatar_bundle, spz_converter, motion_sequence)
- lam/models/rendering/ — GS renderer, FLAME model, Gaussian model
- vhap/ — Video FLAME tracking
- tools/flame_tracking_single_image.py — Single-image tracking
- configs/inference/lam-20k-8gpu.yaml — Model config

## Running
Local dev: run_lam.bat (Gradio) or uvicorn serve:app --port 8080
Docker: docker build -t lam-service . && docker run --gpus all -p 8080:8080 lam-service

## Patches (RTX 5080 / sm_120)
- compiled_autograd.h: guard fix
- nvdiffrast/torch/ops.py: DLL + import fix
- attention.py: xformers → SDPA
- XFORMERS_DISABLED=1

## @omote SDK Integration
- @omote/types/character.ts → GSplatConfig { spzUrl, gaussianCount, arkitToFlameUrl }
- @omote/types/lam.ts → LAMJob, ARKitToFLAMEMapping
- @omote/core → Wav2Vec2Inference, FaceCompositor, PlaybackPipeline

## EKS
GPU: g6.xlarge (L4 24GB). Inference: ~1.4s/avatar. Container: cuda:12.8-runtime.
