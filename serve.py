"""
LAM Character Service — FastAPI production API.

Endpoints:
  POST /avatar/generate   — photo → .spz avatar bundle
  POST /motion/extract    — video → FLAME motion sequence
  POST /motion/from-audio — audio → FLAME expression params
  GET  /artifacts/{job_id}/{filename}
  GET  /health
"""

import os
import sys
import uuid
import time
import tempfile
import pathlib
import json
import shutil
from collections import defaultdict
from contextlib import asynccontextmanager

import numpy as np
import torch
import cv2
from PIL import Image
from omegaconf import OmegaConf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# Disable dynamo / xformers early (same as app_lam.py)
# ---------------------------------------------------------------------------
os.environ.setdefault("XFORMERS_DISABLED", "1")
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
_lam_model = None
_flame_tracking = None
_cfg = None
_artifacts_root = pathlib.Path("artifacts")


# ---------------------------------------------------------------------------
# Model loading helpers (mirrors app_lam.py:503-552)
# ---------------------------------------------------------------------------
def _build_model(cfg):
    from lam.models import ModelLAM
    from safetensors.torch import load_file

    model = ModelLAM(**cfg.model)
    resume = os.path.join(cfg.model_name, "model.safetensors")
    print("=" * 80)
    print("loading pretrained weight from:", resume)
    if resume.endswith("safetensors"):
        ckpt = load_file(resume, device="cpu")
    else:
        ckpt = torch.load(resume, map_location="cpu")
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f"WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
        else:
            print(f"WARN] unexpected param {k}: {v.shape}")
    print("finish loading pretrained weight from:", resume)
    print("=" * 80)
    return model


def _load_cfg():
    """Load config the same way app_lam.py does, but without argparse."""
    infer_path = os.environ.get(
        "APP_INFER", "./configs/inference/lam-20k-8gpu.yaml"
    )
    model_name = os.environ.get(
        "APP_MODEL_NAME",
        "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
    )

    cfg = OmegaConf.create()
    cfg_train = OmegaConf.load(infer_path)

    cfg.source_size = cfg_train.dataset.source_image_res
    cfg.src_head_size = cfg_train.dataset.get("src_head_size", 112)
    cfg.render_size = cfg_train.dataset.render_image.high
    cfg.model_name = model_name
    cfg.model = cfg_train.model
    cfg.motion_video_read_fps = 30
    return cfg, cfg_train


# ---------------------------------------------------------------------------
# Lifespan: load models once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lam_model, _flame_tracking, _cfg

    os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

    # Pre-compile nvdiffrast CUDA extension before anything loads the .pyd
    print("[serve] Pre-compiling nvdiffrast CUDA extension …")
    try:
        import nvdiffrast.torch as dr
        _glctx = dr.RasterizeCudaContext()
        del _glctx
        print("[serve] nvdiffrast compiled OK.")
    except Exception as e:
        print(f"[serve] nvdiffrast warmup warning: {e}")

    _cfg, _ = _load_cfg()

    # LAM model
    print("[serve] Loading LAM model …")
    _lam_model = _build_model(_cfg)
    _lam_model.to("cuda")
    _lam_model.eval()
    print("[serve] LAM model ready.")

    # FLAME tracking
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    print("[serve] Loading FLAME tracking …")
    _flame_tracking = FlameTrackingSingleImage(
        output_dir="output/tracking",
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
        detect_iris_landmarks=False,
    )
    print("[serve] FLAME tracking ready.")

    _artifacts_root.mkdir(parents=True, exist_ok=True)
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LAM Character Service", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# POST /avatar/generate
# ---------------------------------------------------------------------------
@app.post("/avatar/generate")
async def avatar_generate(image: UploadFile = File(...)):
    """Photo → .spz avatar bundle with FLAME rigging data."""
    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
    from lam.export import export_avatar_bundle, ply_to_spz, get_default_mapping_path

    job_id = uuid.uuid4().hex[:12]
    job_dir = _artifacts_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image
    suffix = pathlib.Path(image.filename or "photo.jpg").suffix or ".jpg"
    raw_path = job_dir / f"input{suffix}"
    contents = await image.read()
    raw_path.write_bytes(contents)

    try:
        # --- FLAME tracking (preprocess → optimize → export) ---
        return_code = _flame_tracking.preprocess(str(raw_path))
        if return_code != 0:
            raise HTTPException(500, "FLAME tracking preprocess failed")
        return_code = _flame_tracking.optimize()
        if return_code != 0:
            raise HTTPException(500, "FLAME tracking optimize failed")
        return_code, output_dir = _flame_tracking.export()
        if return_code != 0:
            raise HTTPException(500, "FLAME tracking export failed")

        image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")

        # --- Prepare reference image ---
        source_size = _cfg.source_size
        render_size = _cfg.render_size
        aspect_standard = 1.0

        img_tensor, _, _, shape_param = preprocess_image(
            image_path,
            mask_path=mask_path,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=None,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
            get_shape_param=True,
        )

        # --- Prepare a single-frame motion sequence for canonical pose ---
        flame_params_dir = os.path.join(output_dir, "flame_param")
        src_name = pathlib.Path(image_path).parent.parent.name
        driven_name = pathlib.Path(flame_params_dir).parent.name
        src_driven = [src_name, driven_name]

        motion_seq = prepare_motion_seqs(
            flame_params_dir,
            None,
            save_root=str(job_dir),
            fps=30,
            bg_color=1.0,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size,
            multiply=16,
            need_mask=False,
            vis_motion=False,
            shape_param=shape_param,
            test_sample=False,
            cross_id=False,
            src_driven=src_driven,
        )

        # --- LAM inference ---
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device, dtype = "cuda", torch.float32

        with torch.no_grad():
            res = _lam_model.infer_single_view(
                img_tensor.unsqueeze(0).to(device, dtype),
                None,
                None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={
                    k: v.to(device) for k, v in motion_seq["flame_params"].items()
                },
            )

        # --- Export avatar bundle ---
        flame_model = _lam_model.renderer.flame_model
        bundle = export_avatar_bundle(
            res=res,
            flame_model=flame_model,
            shape_param=shape_param,
            output_dir=str(_artifacts_root),
            job_id=job_id,
        )

        # --- PLY → SPZ ---
        ply_path = bundle["files"]["canonical_ply"]
        spz_path = str(job_dir / "canonical.spz")
        ply_to_spz(ply_path, spz_path)

        # --- Copy ARKit→FLAME mapping ---
        mapping_src = get_default_mapping_path()
        mapping_dst = str(job_dir / "arkit_to_flame.json")
        if os.path.exists(mapping_src):
            shutil.copy2(mapping_src, mapping_dst)

        # --- Save thumbnail from first rendered frame ---
        try:
            rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3]
            thumb = (np.clip(rgb[0], 0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(thumb).save(str(job_dir / "thumbnail.png"))
        except Exception:
            pass

        # Load metadata from exported file
        metadata = {}
        metadata_path = bundle["files"].get("metadata")
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)

        # Build response
        base = f"/artifacts/{job_id}"
        resp = {
            "job_id": job_id,
            "spz_url": f"{base}/canonical.spz",
            "thumbnail_url": f"{base}/thumbnail.png",
            "gaussian_count": bundle["gaussian_count"],
            "lbs_weights_url": f"{base}/lbs_weights.json",
            "bone_tree_url": f"{base}/bone_tree.json",
            "flame_identity_url": f"{base}/flame_identity.json",
            "arkit_to_flame_url": f"{base}/arkit_to_flame.json",
            "metadata": metadata,
        }
        return JSONResponse(resp)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Avatar generation failed: {e}")


# ---------------------------------------------------------------------------
# POST /motion/extract
# ---------------------------------------------------------------------------
@app.post("/motion/extract")
async def motion_extract(video: UploadFile = File(...)):
    """Video → per-frame FLAME motion sequence."""
    from lam.runners.infer.head_utils import load_flame_params
    from lam.export import export_motion_sequence

    job_id = uuid.uuid4().hex[:12]
    job_dir = _artifacts_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video
    suffix = pathlib.Path(video.filename or "clip.mp4").suffix or ".mp4"
    video_path = job_dir / f"input{suffix}"
    contents = await video.read()
    video_path.write_bytes(contents)

    try:
        # Extract frames from video
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.png"), frame)
            frame_idx += 1
        cap.release()

        if frame_idx == 0:
            raise HTTPException(400, "Could not extract frames from video")

        # Track each frame with FlameTrackingSingleImage
        flame_params_list = []
        for i in range(frame_idx):
            frame_path = str(frames_dir / f"{i:06d}.png")
            rc = _flame_tracking.preprocess(frame_path)
            if rc != 0:
                continue
            rc = _flame_tracking.optimize()
            if rc != 0:
                continue
            rc, export_dir = _flame_tracking.export()
            if rc != 0:
                continue

            # Load the exported FLAME params
            flame_param_dir = os.path.join(export_dir, "flame_param")
            npz_files = sorted(pathlib.Path(flame_param_dir).glob("*.npz"))
            if npz_files:
                params = load_flame_params(str(npz_files[0]))
                flame_params_list.append(params)

        if not flame_params_list:
            raise HTTPException(500, "No frames could be tracked")

        # Assemble into dict-of-lists for export
        params_dict = defaultdict(list)
        for fp in flame_params_list:
            for k, v in fp.items():
                params_dict[k].append(v)
        for k, v in params_dict.items():
            params_dict[k] = torch.stack(v)

        # Export motion sequence
        motion_path = str(job_dir / "motion.json")
        result = export_motion_sequence(
            source=dict(params_dict),
            output_path=motion_path,
            fps=int(round(fps)),
        )

        base = f"/artifacts/{job_id}"
        return JSONResponse({
            "job_id": job_id,
            "motion_url": f"{base}/motion.json",
            "frame_count": result["frame_count"],
            "fps": result["fps"],
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Motion extraction failed: {e}")


# ---------------------------------------------------------------------------
# POST /motion/from-audio
# ---------------------------------------------------------------------------
@app.post("/motion/from-audio")
async def motion_from_audio(audio: UploadFile = File(...)):
    """Audio → FLAME expression params (lip sync)."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = _artifacts_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = pathlib.Path(audio.filename or "audio.wav").suffix or ".wav"
    audio_path = job_dir / f"input{suffix}"
    contents = await audio.read()
    audio_path.write_bytes(contents)

    try:
        # Check if Audio2Expression model is available
        a2e_path = os.environ.get(
            "A2E_MODEL_PATH",
            "./model_zoo/audio2expression/",
        )
        if not os.path.isdir(a2e_path):
            raise HTTPException(
                501,
                "Audio2Expression model not installed. "
                "See https://github.com/aigc3d/LAM_Audio2Expression",
            )

        # Try to import and run Audio2Expression
        try:
            sys.path.insert(0, a2e_path)
            from audio2expression import Audio2ExpressionModel

            a2e_model = Audio2ExpressionModel(a2e_path)
            result = a2e_model.infer(str(audio_path))

            # result expected: dict with 'expr' [N, 100], 'jaw_pose' [N, 3], etc.
            from lam.export import export_motion_sequence

            motion_path = str(job_dir / "motion.json")
            export_result = export_motion_sequence(
                source=result,
                output_path=motion_path,
                fps=30,
            )

            base = f"/artifacts/{job_id}"
            return JSONResponse({
                "job_id": job_id,
                "motion_url": f"{base}/motion.json",
                "frame_count": export_result["frameCount"],
                "fps": export_result["fps"],
            })
        except ImportError:
            raise HTTPException(
                501,
                "Audio2Expression model found but cannot be imported. "
                "Check installation.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Audio-to-expression failed: {e}")


# ---------------------------------------------------------------------------
# GET /artifacts/{job_id}/{filename}
# ---------------------------------------------------------------------------
@app.get("/artifacts/{job_id}/{filename}")
async def get_artifact(job_id: str, filename: str):
    """Serve exported artifact files."""
    file_path = _artifacts_root / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, f"Artifact not found: {job_id}/{filename}")
    # Prevent path traversal
    try:
        file_path.resolve().relative_to(_artifacts_root.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    return FileResponse(str(file_path))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Service health check."""
    gpu_name = "unknown"
    gpu_mem = 0
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem // (1024 ** 2)
    except Exception:
        pass

    capabilities = ["avatar_generate", "motion_extract"]
    a2e_path = os.environ.get("A2E_MODEL_PATH", "./model_zoo/audio2expression/")
    if os.path.isdir(a2e_path):
        capabilities.append("motion_from_audio")

    return {
        "status": "ok",
        "gpu": gpu_name,
        "gpu_memory_mb": gpu_mem,
        "model": "LAM-20K",
        "capabilities": capabilities,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        workers=1,
    )
