"""Run Pigey's ER localization on real LIBERO frames with the local VLM.

Uses Pigey's own _gemini_er_localize (parsing, SAM-masked bbox -> world
unprojection, filters), so only the detector differs from upstream; SAM runs
Pigey's own function through sam_server.py. Each frame is checked at reset and
with the arm moved into the workspace (where the detector has looped), under
each requested sampling mode.

Every intermediate is saved per frame: the frame the detector sees, its raw
reply, the Gemini-format conversion, each SAM mask, the resulting world
positions, and the annotated image Claude receives. Simulator ground truth
(segmentation and object positions) is read for this report only; it is never
given to an agent.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

SETUP_ROOT = Path(__file__).resolve().parents[1]  # baselines/pigey_setup
if str(SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(SETUP_ROOT))

from runner.perception import (CLAUDE_ER_SYSTEM, ER_PROMPT, LOCAL_ER_MODEL, ClaudeER, LocalER, SamClient, route_local_er,
                        route_sam)
from runner.run_claude import load_harness

MJOBJ_GEOM = 5
# Arm poses (world EE targets) that put the gripper over the objects, as after
# a VLA rollout or GoHome; None is the settled reset frame.
POSES = {"reset": None, "arm_center": (-0.05, 0.05, 1.02), "arm_left": (-0.15, 0.22, 1.05)}
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
          (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def entity_names(env):
    inner = env.env
    return list(getattr(inner, "objects_dict", {})) + list(getattr(inner, "fixtures_dict", {}))


def object_positions(env):
    positions = {}
    for name in getattr(env.env, "objects_dict", {}):
        try:
            body = env.sim.model.body_name2id(f"{name}_main")
        except Exception:
            continue
        positions[name] = np.array(env.sim.data.body_xpos[body], dtype=float)
    return positions


def segmentation(env, names, size):
    """Per-pixel entity name in native (sim.render RGB) orientation, '' elsewhere."""
    # The segmentation buffer is vertically flipped relative to the RGB render;
    # robosuite's camera_utils.get_camera_segmentation applies the same flip.
    seg = env.sim.render(camera_name="agentview", height=size, width=size, segmentation=True)[::-1]
    model = env.sim.model
    body_entity = {}
    for body in range(model.nbody):
        body_name = model.body_id2name(body) or ""
        body_entity[body] = next((n for n in names if body_name == n or body_name.startswith(n + "_")), "")
    out = np.full(seg.shape[:2], "", dtype=object)
    geoms = seg[..., 0] == MJOBJ_GEOM
    for geom in np.unique(seg[..., 1][geoms]):
        # Collision geoms (robosuite group 0) are drawn in the segmentation
        # buffer but are invisible to the camera; keep visual geoms only.
        if model.geom_group[geom] == 0 or model.geom_rgba[geom, 3] == 0:
            continue
        out[geoms & (seg[..., 1] == geom)] = body_entity.get(int(model.geom_bodyid[geom]), "")
    return out


def crop_panel(frame, detections, masks, seg_llm, path, pad=12, scale=3):
    """One close-up per detection: Qwen box (green), SAM mask (red), true outline (blue)."""
    tiles = []
    rgb = np.asarray(frame).copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox_px"]
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(rgb.shape[1], x2 + pad + 1), min(rgb.shape[0], y2 + pad + 1)
        tile = rgb[cy1:cy2, cx1:cx2].astype(float)
        mask = masks.get(tuple(d["bbox_px"]))
        if mask is not None:
            m = mask[:, ::-1][cy1:cy2, cx1:cx2]
            tile[m] = 0.5 * tile[m] + 0.5 * np.array([255, 0, 0])
        if d["gt_entity"]:
            gt = seg_llm[cy1:cy2, cx1:cx2] == d["gt_entity"]
            edge = gt & ~(np.roll(gt, 1, 0) & np.roll(gt, -1, 0) & np.roll(gt, 1, 1) & np.roll(gt, -1, 1))
            tile[edge] = [0, 120, 255]
        image = Image.fromarray(tile.astype(np.uint8)).resize(((cx2 - cx1) * scale, (cy2 - cy1) * scale), Image.NEAREST)
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1 - cx1) * scale, (y1 - cy1) * scale, (x2 - cx1) * scale, (y2 - cy1) * scale],
                       outline=(0, 255, 0), width=2)
        draw.text((3, 3), f"#{d['index']} {d['label'][:22]}", fill=(255, 255, 0))
        tiles.append(image)
    if not tiles:
        return
    rows = [tiles[i:i + 6] for i in range(0, len(tiles), 6)]
    panel = Image.new("RGB", (max(sum(t.width for t in r) for r in rows), sum(max(t.height for t in r) for r in rows)))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            panel.paste(t, (x, y))
            x += t.width
        y += max(t.height for t in r)
    panel.save(path)


def box_of(mask):
    """Box of the largest connected region (stray specks would inflate it)."""
    from scipy.ndimage import label
    regions, count = label(mask)
    if count > 1:
        mask = regions == 1 + int(np.argmax(np.bincount(regions.flat)[1:]))
    ys, xs = np.where(mask)
    return None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return inter / float(area(a) + area(b) - inter) if inter else 0.0


def draw_boxes(image, boxes, labels):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = COLORS[i % len(COLORS)]
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0] + 2, box[1] + 2), label, fill=color)
    return image


def mask_scores(mask_llm, gt):
    union = (mask_llm | gt).sum()
    return (round(float((mask_llm & gt).sum() / union), 3) if union else 0.0,
            round(float((mask_llm & gt).sum() / max(1, mask_llm.sum())), 3))


def check_frame(module, runner_obs, env, local_er, sam, sam_log, seg_native, truth, frame_dir, sampling, tag):
    """Run upstream localization once and score it; returns the frame record."""
    local_er.sampling = sampling
    sam_log.clear()
    # Minimal stand-in for LiberoRunner, whose constructor connects to the
    # pi0.5 server; localization only reads obs, env, and t.
    runner = SimpleNamespace(obs=runner_obs, env=env, t=0)
    world = module.LiberoRunner._gemini_er_localize(runner)
    call = local_er.calls[-1]
    (frame_dir / f"qwen_raw_{tag}.txt").write_text(local_er.last_raw)
    (frame_dir / f"gemini_format_{tag}.json").write_text(getattr(runner, "_gemini_er_raw_text", ""))

    seg_llm = seg_native[:, ::-1]
    gt_boxes = {}
    for name in set(seg_llm.flat) - {""}:
        if (seg_llm == name).sum() >= 20:
            gt_boxes[name] = box_of(seg_llm == name)
    frame = Image.fromarray(np.ascontiguousarray(runner_obs["agentview_image"][::-1, ::-1]))
    masks = {tuple(int(v) for v in box): mask for box, mask in sam_log}
    np.savez_compressed(frame_dir / f"sam_masks_{tag}.npz",
                        **{"x".join(map(str, box)): mask for box, mask in masks.items()})
    # Image in the orientation Pigey's bbox mapping assumes for SAM (LLM view
    # mirrored left-right); Pigey itself passes a fresh sim.render instead.
    assumed = np.ascontiguousarray(np.asarray(frame)[:, ::-1])
    oriented = {}
    detections, mask_layer = [], np.zeros((*seg_native.shape, 3), dtype=np.uint8)
    for i, (label, xyz) in enumerate(world.items()):
        box = [int(v) for v in runner._gemini_er_bboxes[label]]
        entity, best = max(((n, iou(box, b)) for n, b in gt_boxes.items()), key=lambda t: t[1], default=("", 0.0))
        entry = {"index": i, "label": label, "bbox_px": box, "world": [round(float(v), 4) for v in xyz],
                 "gt_entity": entity if best > 0 else "", "box_iou": round(best, 3)}
        mask = masks.get(tuple(box))
        if mask is not None:
            mask_llm = mask[:, ::-1]
            mask_layer[mask_llm] = COLORS[i % len(COLORS)]
            if entry["gt_entity"]:
                entry["sam_iou"], entry["sam_on_entity"] = mask_scores(mask_llm, seg_llm == entry["gt_entity"])
        entry["sam_mask"] = mask is not None
        size = seg_native.shape[0]
        alt = sam(assumed, box, llm_view_rotated=True, img_h=size, img_w=size)
        if alt is not None:
            oriented[tuple(box)] = alt
            if entry["gt_entity"]:
                entry["sam_iou_oriented"], entry["sam_on_entity_oriented"] = mask_scores(
                    alt[:, ::-1], seg_llm == entry["gt_entity"])
        if entry["gt_entity"] in truth:
            entry["world_error_m"] = round(float(np.linalg.norm(truth[entry["gt_entity"]] - np.array(xyz))), 3)
        detections.append(entry)
    found = {d["gt_entity"] for d in detections if d["box_iou"] >= 0.5}
    record = {
        "sampling": sampling, "truncated": call["truncated"], "output_tokens": call["output_tokens"],
        "duration_s": call["duration_s"], "detections": detections,
        "objects_found": sorted(found & set(truth)), "objects_missed": sorted(set(truth) - found),
        "visible_objects": sorted(set(truth) & set(gt_boxes)),
        "fixtures_detected": sorted(found - set(truth)),
        "false_detections": [d["label"] for d in detections if d["box_iou"] < 0.5],
    }
    labels = [f"#{d['index']} {d['label']}" for d in detections]
    draw_boxes(frame, [d["bbox_px"] for d in detections], labels).save(frame_dir / f"boxes_{tag}.png")
    Image.blend(frame, Image.fromarray(mask_layer), 0.45).save(frame_dir / f"sam_masks_{tag}.png")
    crop_panel(frame, detections[:12], masks, seg_llm, frame_dir / f"sam_detail_{tag}.png")
    crop_panel(frame, detections[:12], oriented, seg_llm, frame_dir / f"sam_detail_oriented_{tag}.png")
    # What Claude receives: Pigey's own annotation of the same detections.
    runner._gemini_er_bboxes_ordered = [d["bbox_px"] for d in detections]
    runner._perceive_pngs_skip_detect = True
    agent_png, _ = module.LiberoRunner.perceive_pngs(runner)
    (frame_dir / f"claude_agentview_{tag}.png").write_bytes(agent_png)
    (frame_dir / f"claude_objects_{tag}.txt").write_text("objects: " + "; ".join(
        f"#{d['index']} world=({d['world'][0]:+.3f},{d['world'][1]:+.3f},{d['world'][2]:+.3f}) "
        f"bbox=[{','.join(map(str, d['bbox_px']))}]" for d in detections))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_goal_task")
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--poses", nargs="+", default=list(POSES), choices=list(POSES))
    parser.add_argument("--samplings", nargs="+", default=["greedy", "default"], choices=["greedy", "default"])
    parser.add_argument("--sam-inputs", nargs="+", default=["upstream", "flipped"], choices=["upstream", "flipped"],
                        help="upstream: SAM gets the image Pigey passes; flipped: that image flipped vertically, "
                             "matching the box orientation Pigey's mapping assumes")
    parser.add_argument("--er-backend", choices=["local", "claude"], default="local",
                        help="local: VLM on the vLLM server; claude: Claude Code CLI with --claude-model")
    parser.add_argument("--claude-model", default="claude-sonnet-5-5")
    parser.add_argument("--er-url", default=os.environ.get("PIGEY_ER_URL"),
                        help="ER server URL; serve_perception.sh writes it to perception.env")
    parser.add_argument("--er-model", default=os.environ.get("PIGEY_ER_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--sam-url", default=os.environ.get("PIGEY_SAM_URL"),
                        help="SAM server URL; serve_perception.sh writes it to perception.env")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not args.sam_url or (args.er_backend == "local" and not args.er_url):
        parser.error("Set --er-url and --sam-url, or source perception.env from serve_perception.sh.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qwen_prompt.txt").write_text(ER_PROMPT if args.er_backend == "local" else CLAUDE_ER_SYSTEM)

    import litellm
    if args.er_backend == "claude":
        from runner.claude_code import ClaudeCodeBackend
        root = Path(os.environ.get("PIGEY_ROOT", "/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey"))
        local_er = ClaudeER(ClaudeCodeBackend(root / "tmp/claude-code", model=args.claude_model))
    else:
        local_er = LocalER(args.er_url, args.er_model)
    route_local_er(litellm, local_er)
    os.environ["GEMINI_ER_MODEL"] = LOCAL_ER_MODEL
    os.environ.setdefault("GEMINI_API_KEY", "pigey-local-er-unused")
    os.environ["HARNESS_PERCEPTION"] = "gemini_er"
    pigey = Path(os.environ["PIGEY_REPO"])
    module = load_harness(pigey / "sim/agent_sim.py")
    sam = SamClient(args.sam_url)
    sam_log = []
    sam_input = {"flip": False}

    def recording_sam(rgb_image, bbox, **kwargs):
        if sam_input["flip"]:
            rgb_image = np.ascontiguousarray(rgb_image[::-1])
        mask = sam(rgb_image, bbox, **kwargs)
        if mask is not None:
            sam_log.append((bbox, mask))
        return mask

    route_sam(recording_sam)
    import spatial_tools
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    size = module.LIBERO_ENV_RESOLUTION
    suite = benchmark.get_benchmark_dict()[args.suite]()
    report = []
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=size, camera_widths=size,
                                 camera_depths=True)
        try:
            env.seed(7)
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(task_id)[args.episode])
            for _ in range(module.NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(module.LIBERO_DUMMY_ACTION)
            for pose in args.poses:
                if POSES[pose] is not None:
                    spatial_tools.move_ee_to(env, POSES[pose], max_steps=150)
                    obs, _, _, _ = env.step(module.LIBERO_DUMMY_ACTION)
                frame_dir = out / f"t{task_id}_{pose}"
                frame_dir.mkdir(exist_ok=True)
                Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).save(frame_dir / "frame.png")
                names = entity_names(env)
                truth = object_positions(env)
                seg_native = segmentation(env, names, size)
                seg_names = sorted(set(seg_native.flat) - {""})
                np.savez_compressed(frame_dir / "ground_truth_native.npz", names=np.array(seg_names),
                                    index=np.vectorize(lambda n: seg_names.index(n) + 1 if n else 0)(seg_native))
                # Self-check of the ground truth: Pigey's own projection of each
                # object's position should land on that object's segment.
                aligned = {}
                for name, pos in truth.items():
                    x, y = spatial_tools.project_world_to_pixel(env, pos, img_h=size, img_w=size)
                    patch = seg_native[:, ::-1][max(0, y - 6):y + 7, max(0, x - 6):x + 7]
                    aligned[name] = bool((patch == name).any()) if 0 <= x < size and 0 <= y < size else None
                # Which orientation is the RGB that Pigey renders for SAM in?
                llm = np.asarray(obs["agentview_image"][::-1, ::-1], dtype=float)
                rendered = np.asarray(env.sim.render(camera_name="agentview", height=size, width=size), dtype=float)
                Image.fromarray(rendered.astype(np.uint8)).save(frame_dir / "sam_input_as_rendered.png")
                render_orientation = {
                    "mean_abs_diff_vs_llm_mirrored_lr": round(float(np.abs(rendered - llm[:, ::-1]).mean()), 2),
                    "mean_abs_diff_vs_llm_rotated_180": round(float(np.abs(rendered - llm[::-1, ::-1]).mean()), 2),
                }
                gt_llm = seg_native[:, ::-1]
                gt_names = sorted(set(gt_llm.flat) - {""})
                draw_boxes(Image.open(frame_dir / "frame.png"), [box_of(gt_llm == n) for n in gt_names],
                           gt_names).save(frame_dir / "ground_truth.png")
                for sampling in args.samplings:
                  for mode in args.sam_inputs:
                    sam_input["flip"] = mode == "flipped"
                    tag = sampling if mode == "upstream" else f"{sampling}_flipped"
                    record = check_frame(module, obs, env, local_er, sam, sam_log, seg_native, truth, frame_dir,
                                         sampling, tag)
                    record.update(task=task_id, language=task.language, pose=pose, gt_alignment=aligned,
                                  render_orientation=render_orientation, sam_input=mode)
                    report.append(record)
                    print(json.dumps({k: record[k] for k in ("task", "pose", "sampling", "truncated",
                                                             "objects_found", "objects_missed",
                                                             "false_detections")}), flush=True)
        finally:
            env.close()
    errors = local_er.errors + sam.errors
    summary = {"er_model": local_er.model, "er_calls": local_er.calls, "sam_calls": sam.calls,
               "errors": errors, "frames": report}
    (out / "perception_check.json").write_text(json.dumps(summary, indent=2))
    if errors or not sam.calls or not any(frame["detections"] for frame in report):
        print("Perception check failed: no detections, no SAM calls, or server errors", file=sys.stderr)
        return 1
    print(f"Perception check finished; review {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
