import os
import json
import datetime
import numpy as np
import torch


def export_avatar_bundle(res, flame_model, shape_param, output_dir, job_id):
    """
    Export a complete avatar bundle from inference results.

    Args:
        res: dict from model.infer_single_view(), contains res['cano_gs_lst'][0] (GaussianModel)
        flame_model: FlameHead or FlameHeadSubdivided instance
        shape_param: tensor of shape [1, 10] or [1, 100]
        output_dir: base output directory path
        job_id: unique string identifier for this export

    Returns:
        dict with paths to all exported files and key metadata
    """
    bundle_dir = os.path.join(output_dir, job_id)
    os.makedirs(bundle_dir, exist_ok=True)

    exported = {}

    # 1. canonical.ply - save the canonical gaussian splat
    gs = res["cano_gs_lst"][0]
    ply_path = os.path.join(bundle_dir, "canonical.ply")
    gs.save_ply(ply_path)
    exported["canonical_ply"] = ply_path
    gaussian_count = gs.xyz.shape[0]

    # 2. lbs_weights.json - linear blend skinning weights [V, J]
    lbs_weights_path = os.path.join(bundle_dir, "lbs_weights.json")
    if hasattr(flame_model, "lbs_weights_up"):
        lbs = flame_model.lbs_weights_up.cpu().float().numpy().tolist()
    else:
        lbs = flame_model.lbs_weights.cpu().float().numpy().tolist()
    with open(lbs_weights_path, "w") as f:
        json.dump(lbs, f)
    exported["lbs_weights"] = lbs_weights_path

    # 3. bone_tree.json - skeleton hierarchy with joint positions
    bone_tree_path = os.path.join(bundle_dir, "bone_tree.json")
    joint_names = ["root", "neck", "jaw", "leftEye", "rightEye"]
    parents_buf = flame_model.parents.cpu().numpy().tolist()
    bone_tree = {
        "bones": [
            {
                "name": "root",
                "position": [0, 0, 0],
                "children": [
                    {
                        "name": "neck",
                        "position": [0, 0, 0],
                        "children": [
                            {"name": "jaw", "position": [0, 0, 0]},
                            {"name": "leftEye", "position": [0, 0, 0]},
                            {"name": "rightEye", "position": [0, 0, 0]},
                        ],
                    }
                ],
            }
        ],
        "joint_names": joint_names,
        "parents": parents_buf,
    }

    # Compute joint positions from shape params if possible
    try:
        from lam.models.rendering.flame_model.flame import vertices2joints, blend_shapes

        if hasattr(flame_model, "v_template_up"):
            v_template = flame_model.v_template_up
            shapedirs = flame_model.shapedirs_up
        else:
            v_template = flame_model.v_template
            shapedirs = flame_model.shapedirs

        n_shape = flame_model.n_shape_params
        sp = shape_param.to(v_template.device)
        batch_size = sp.shape[0]
        template_vertices = v_template.unsqueeze(0).expand(batch_size, -1, -1)
        v_shaped = template_vertices + blend_shapes(sp, shapedirs[:, :, :n_shape])
        J = vertices2joints(flame_model.J_regressor, v_shaped)

        bone_tree["bones"][0]["position"] = J[0, 0, :].cpu().tolist()
        bone_tree["bones"][0]["children"][0]["position"] = J[0, 1, :].cpu().tolist()
        bone_tree["bones"][0]["children"][0]["children"][0]["position"] = J[0, 2, :].cpu().tolist()
        bone_tree["bones"][0]["children"][0]["children"][1]["position"] = J[0, 3, :].cpu().tolist()
        bone_tree["bones"][0]["children"][0]["children"][2]["position"] = J[0, 4, :].cpu().tolist()
    except Exception:
        pass

    with open(bone_tree_path, "w") as f:
        json.dump(bone_tree, f, indent=2)
    exported["bone_tree"] = bone_tree_path

    # 4. flame_identity.json - shape parameters
    identity_path = os.path.join(bundle_dir, "flame_identity.json")
    if isinstance(shape_param, torch.Tensor):
        shape_list = shape_param.detach().cpu().float().numpy().flatten().tolist()
    else:
        shape_list = np.array(shape_param).flatten().tolist()
    with open(identity_path, "w") as f:
        json.dump({"shape_params": shape_list}, f)
    exported["flame_identity"] = identity_path

    # 5. metadata.json
    metadata_path = os.path.join(bundle_dir, "metadata.json")
    metadata = {
        "job_id": job_id,
        "gaussian_count": int(gaussian_count),
        "model_version": "lam-v1",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "shape_param_dim": len(shape_list),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    exported["metadata"] = metadata_path

    # 6. thumbnail.png - first rendered frame if available
    try:
        rendered = None
        if "render_images" in res and res["render_images"] is not None:
            rendered = res["render_images"]
        elif "img" in res and res["img"] is not None:
            rendered = res["img"]

        if rendered is not None:
            from PIL import Image

            if isinstance(rendered, torch.Tensor):
                img_np = rendered[0].detach().cpu().float().numpy()
                if img_np.shape[0] in (3, 4):
                    img_np = np.transpose(img_np, (1, 2, 0))
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            else:
                img_np = np.array(rendered[0])
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)

            thumb_path = os.path.join(bundle_dir, "thumbnail.png")
            Image.fromarray(img_np).save(thumb_path)
            exported["thumbnail"] = thumb_path
    except Exception:
        pass

    return {
        "bundle_dir": bundle_dir,
        "files": exported,
        "gaussian_count": int(gaussian_count),
        "shape_param_dim": len(shape_list),
    }
