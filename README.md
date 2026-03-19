# LAM Character Service

> GPU microservice that converts photos, video, and audio into animatable 3D Gaussian avatars using FLAME parameters.

Fork of [LAM (Large Avatar Model)](https://github.com/aigc3d/LAM) — SIGGRAPH 2025. Extended with a production API and export pipeline for the [Omote](https://github.com/your-org/omote) character platform.

## Capabilities

| Endpoint | Input | Output | Time |
|----------|-------|--------|------|
| `POST /avatar/generate` | Photo (JPEG/PNG) | `.spz` avatar bundle + rigging data | ~5-8s |
| `POST /motion/extract` | Video (MP4) | Per-frame FLAME motion JSON | ~30-120s |
| `POST /motion/from-audio` | Audio (WAV/MP3) | FLAME expression params JSON | ~2-5s |

All outputs use FLAME parameters as the universal animation interface — any motion source can drive any avatar.

## Quick Start

### Local (requires RTX GPU with ≥16GB VRAM)

```bash
# Install dependencies
pip install -r requirements-serve.txt

# Start the API server
uvicorn serve:app --host 0.0.0.0 --port 8080

# Or use Gradio dev UI
python run_lam.bat
```

### Docker

```bash
docker build -t lam-service .
docker run --gpus all -p 8080:8080 lam-service
```

### Generate an Avatar

```bash
curl -X POST -F "image=@photo.jpg" http://localhost:8080/avatar/generate
```

Response:
```json
{
  "job_id": "abc123",
  "spz_url": "/artifacts/abc123/canonical.spz",
  "gaussian_count": 20000,
  "lbs_weights_url": "/artifacts/abc123/lbs_weights.json",
  "bone_tree_url": "/artifacts/abc123/bone_tree.json",
  "flame_identity_url": "/artifacts/abc123/flame_identity.json",
  "arkit_to_flame_url": "/artifacts/abc123/arkit_to_flame.json",
  "thumbnail_url": "/artifacts/abc123/thumbnail.png"
}
```

### Extract Motion from Video

```bash
curl -X POST -F "video=@clip.mp4" http://localhost:8080/motion/extract
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Photo/Video │────▶│  LAM Service     │────▶│  S3 / CDN   │
│  Audio       │     │  (GPU, FastAPI)  │     │             │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                     │
                    ┌──────────────────┐              │
                    │  @omote SDK      │◀─────────────┘
                    │  (Browser/iOS)   │
                    │  - Wav2Vec2      │
                    │  - ARKit→FLAME   │
                    │  - GS Rendering  │
                    └──────────────────┘
```

## API Reference

### `POST /avatar/generate`
Generates a 3D Gaussian Splatting avatar from a single photo.

**Input**: `multipart/form-data` with `image` field (JPEG, PNG)

**Output**: JSON with URLs to all avatar artifacts:
- `.spz` — compressed Gaussian splat (Niantic SPZ format)
- `lbs_weights.json` — linear blend skinning weights `[V, 5]`
- `bone_tree.json` — skeleton hierarchy (5 joints)
- `flame_identity.json` — FLAME shape parameters
- `arkit_to_flame.json` — ARKit blendshape → FLAME expression mapping
- `thumbnail.png` — preview render

### `POST /motion/extract`
Extracts per-frame FLAME parameters from video using VHAP tracking.

**Input**: `multipart/form-data` with `video` field (MP4)

**Output**: JSON with URL to motion sequence file containing per-frame `expr[100]`, `rotation[3]`, `jaw_pose[3]`, `neck_pose[3]`, `eyes_pose[6]`, `translation[3]`.

### `POST /motion/from-audio`
Generates FLAME expression parameters from audio input.

**Input**: `multipart/form-data` with `audio` field (WAV, MP3)

**Output**: JSON with URL to motion sequence with expression-only FLAME params.

### `GET /artifacts/{job_id}/{filename}`
Downloads a specific artifact file.

### `GET /health`
Returns service status, GPU info, and available capabilities.

## Key Files

| File | Purpose |
|------|---------|
| `serve.py` | FastAPI production service |
| `app_lam.py` | Gradio development UI |
| `lam/export/` | Export pipeline (avatar bundle, SPZ, motion) |
| `lam/models/` | LAM transformer + Gaussian renderer |
| `lam/models/rendering/flame_model/` | FLAME head model + LBS |
| `vhap/` | Video-based FLAME tracking |
| `tools/flame_tracking_single_image.py` | Single-image FLAME tracking |
| `configs/inference/lam-20k-8gpu.yaml` | Model configuration |

## Hardware Requirements

- **GPU**: NVIDIA RTX with ≥16GB VRAM (tested on RTX 5080)
- **CUDA**: 12.8+
- **RAM**: 32GB recommended
- **Inference**: ~1.4s per avatar generation

## Credits

Based on [LAM: Large Avatar Model for One-shot Animatable Gaussian Head Avatar](https://github.com/aigc3d/LAM) (SIGGRAPH 2025).
