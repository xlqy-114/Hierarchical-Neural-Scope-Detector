"""
Full_S_S0_v26_online_heads_feature_rich

Self-contained S0-triggered can_bus harmful-fault monitor for OpenDriveVLA.

This file intentionally contains the v19 can_bus multi-stage harmful-monitor path:
  - fault target: can_bus only;
  - detector object: can_bus only;
  - value-level can_bus detector retained from the proven v7 line;
  - can_bus_mlp hook retained for recording/debug only, not as a hard alarm;
  - small_signed_state structure detector added for harmful underflow/collapse cases;
  - translation_xyz shape detector added for harmful x/y collapse cases;
  - detector reports separate raw injected-fault recall from harmful-fault recall;
  - post-eval utilities integrated in this same file;
  - v19 adds E-output semantic decomposition, context proxies, and temporal
    features for distinguishing fault-induced output collapse from
    pre-existing/scene-consistent output anomalies.

It does not import any previous S0 implementation file.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
# v35: subset1200 S0+S1 mixed fault inference with effective S1 activation injection.
from typing import Any, Dict, Iterable, List, Optional, Tuple

import deepspeed
import numpy as np
import pandas as pd
import joblib
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmengine import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "drivevla")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset
from data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from data_utils.nuscenes_llava_distributed_sampler import ContinuousSceneDistributedSampler
from llava.model.builder import load_pretrained_model
from llava.train.train import DataArguments
from llava.utils import disable_torch_init
from utils.tensor_utils import move_data_to_device

VERSION = "Subset1200_v53_sample_inline_online_v35_1_scdout_exact"

DETECTOR_OBJECTS = ["can_bus"]
# v44 keeps the proven v35.1 online detector objects unchanged, but expands
# fault sources for OOD sensitivity testing.  C/D faults are activation-output
# mutations through generic forward hooks.  E faults are post-generation output
# perturbations applied to the predicted trajectory text before post-eval.
FAULT_TARGETS = ["can_bus", "s1_activation", "c_activation", "d_activation"]
S0_TARGET_MODE_TO_TARGETS = {
    "A1_can_bus": ["can_bus"],
    "A1_can_bus_s1_mixed": ["can_bus", "s1_activation"],
    "SCDOUT_internal_online": ["can_bus", "s1_activation", "c_activation", "d_activation"],
    "CDE_ood": ["c_activation", "d_activation"],
    "CDE_ood_with_S0S1": ["can_bus", "s1_activation", "c_activation", "d_activation"],
}

CARDINALITY_DEFAULT = "1:6,2:3,3:1"
BIT_OP_DEFAULT = "flip:1,stuck0:1,stuck1:1"

# can_bus grouping:
# [:3] translation, [3:7] rotation, [-2] patch angle rad, [-1] patch angle deg
CANBUS_GROUPS = {
    "translation_xyz": [0, 1, 2],
    "rotation_quat": [3, 4, 5, 6],
    "small_signed_state": [7, 8, 10, 11, 12],
    "lp_9": [9],
    "lp_13": [13],
    "near_zero_slots": [14, 15],
    "patch_angle_pair": [16, 17],
}
CANBUS_GROUP_ORDER = list(CANBUS_GROUPS.keys())
CANBUS_GROUP_TARGET_WEIGHTS = {
    "translation_xyz": 5,
    "rotation_quat": 4,
    "small_signed_state": 2,
    "lp_9": 4,
    "lp_13": 4,
    "near_zero_slots": 1,
    "patch_angle_pair": 5,
}
CANBUS_DIRECT_IDXS = [0, 1, 16, 17]
CANBUS_HIGH_PRIORITY_GROUPS = ["translation_xyz", "rotation_quat", "patch_angle_pair", "lp_9", "lp_13"]
CANBUS_WEAK_GROUPS = ["small_signed_state", "near_zero_slots"]

# v19 inherits the small-signed-state structure logic that fixed the v9 harmful false negatives.
# The detector below does not tune general thresholds blindly; it adds structure-aware
# checks for exponent collapse, per-index log-magnitude outliers, and context mismatch.
SMALL_SIGNED_STATE_IDXS = CANBUS_GROUPS["small_signed_state"]
SMALL_SIGNED_STRUCTURE_FEATURES = [
    "min_log_abs_margin",
    "max_log_abs_margin",
    "collapse_hit_count",
    "sign_rare_count",
    "context_mahalanobis",
]

TRANSLATION_SHAPE_FEATURES = [
    "xy_log_ratio",
    "xy_low_margin",
    "xy_high_margin",
    "xy_shape_alarm",
]

DEFAULT_HARM_L2_THRESHOLD = 0.5
DEFAULT_MILD_HARM_L2_THRESHOLD = 0.1
DEFAULT_HARM_COLLISION_THRESHOLD = 0.1
DEFAULT_STRICT_COLLISION_THRESHOLD = 1.0 / 3.0

MIN_EFFECTIVE_REL = 0.10
MAX_INJECTION_ATTEMPTS = 128

CANBUS_GROUP_FEATURE_MAP = {
    "translation_xyz": ["finite_ratio", "max_dim_robust_z", "norm_ratio", "abs_max_ratio", "top1_over_mean"],
    "rotation_quat": ["finite_ratio", "max_dim_robust_z", "norm_ratio", "abs_max_ratio", "top1_over_mean", "quat_norm_error"],
    "small_signed_state": ["finite_ratio", "max_dim_robust_z", "norm_ratio", "abs_max_ratio", "top1_over_mean"],
    "lp_9": ["finite_ratio", "max_dim_robust_z", "norm_ratio", "abs_max_ratio", "top1_over_mean"],
    "lp_13": ["finite_ratio", "max_dim_robust_z", "norm_ratio", "abs_max_ratio", "top1_over_mean"],
    "near_zero_slots": ["finite_ratio", "max_dim_robust_z", "norm_ratio", "abs_max_ratio", "top1_over_mean"],
    "patch_angle_pair": ["finite_ratio", "max_dim_robust_z", "norm_ratio", "abs_max_ratio", "top1_over_mean", "angle_consistency_error"],
}
CANBUS_DIRECT_FEATURES = ["finite_ratio", "direct_max_dim_robust_z", "direct_norm_ratio", "xy_max_dim_robust_z", "patch_max_dim_robust_z", "angle_consistency_error"]

DEFAULT_CAN_BUS_MLP_HOOK_SUFFIX = "vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.can_bus_mlp"


# v32 S1 activation fault targets. These are suffix-matched against model.named_modules().
# Preflight verified all 14 default targets register and execute on full-dataset inference.
DEFAULT_V32_S1_TARGET_SPECS = (
    "primary_l4_conv3=img_backbone.layer4.0.conv3,"
    "l1_conv1=img_backbone.layer1.0.conv1,"
    "l1_conv2=img_backbone.layer1.0.conv2,"
    "l1_downsample=img_backbone.layer1.0.downsample,"
    "l2_conv1=img_backbone.layer2.0.conv1,"
    "l2_conv2=img_backbone.layer2.0.conv2,"
    "l4_conv1=img_backbone.layer4.0.conv1,"
    "neck_lateral0=img_neck.lateral_convs.0.conv,"
    "neck_lateral1=img_neck.lateral_convs.1.conv,"
    "neck_lateral2=img_neck.lateral_convs.2.conv,"
    "neck_fpn0=img_neck.fpn_convs.0.conv,"
    "neck_fpn1=img_neck.fpn_convs.1.conv,"
    "neck_fpn2=img_neck.fpn_convs.2.conv,"
    "neck_fpn3=img_neck.fpn_convs.3.conv"
)
DEFAULT_V32_S1_PRIMARY_TARGET = "primary_l4_conv3"
DEFAULT_V32_S1_BROAD_TARGET_WEIGHTS = (
    "l1_conv1:1,l1_conv2:1,l1_downsample:1,l2_conv1:1,l2_conv2:1,"
    "l4_conv1:1,neck_lateral0:1,neck_lateral1:1,neck_lateral2:1,"
    "neck_fpn0:1,neck_fpn1:1,neck_fpn2:1,neck_fpn3:1"
)

# v44 C/D/E OOD using v35.1 model single-fault target defaults.  These are deliberately
# coarse and suffix-based so that readiness scan can verify the exact module
# names in the user's current checkout before a formal run.
DEFAULT_V44_C_TARGET_SPECS = (
    "c_mm_projector_scene=mm_projector_scene,"
    "c_mm_projector_scene_layer0=mm_projector_scene.0,"
    "c_mm_projector_scene_layer1=mm_projector_scene.1,"
    "c_mm_projector_scene_layer2=mm_projector_scene.2"
)
DEFAULT_V44_C_TARGET_WEIGHTS = "c_mm_projector_scene:3,c_mm_projector_scene_layer0:1,c_mm_projector_scene_layer1:1,c_mm_projector_scene_layer2:1"

DEFAULT_V44_D_TARGET_SPECS = (
    "d_llm_layer0=model.layers.0,"
    "d_llm_layer6=model.layers.6,"
    "d_llm_layer12=model.layers.12,"
    "d_llm_layer18=model.layers.18,"
    "d_llm_layer23=model.layers.23"
)
DEFAULT_V44_D_TARGET_WEIGHTS = "d_llm_layer0:1,d_llm_layer6:1,d_llm_layer12:1,d_llm_layer18:1,d_llm_layer23:1"

# E is not a torch module.  These targets perturb the generated planning output
# before pred_trajs_dict/per-sample metric construction.  They do not touch GT or
# metric computation.
DEFAULT_V44_E_TARGET_WEIGHTS = "e_traj_zero:1,e_traj_y_shift:1,e_traj_scale:1,e_parse_break:1"
CANBUS_MLP_FEATURES = [
    "finite_ratio",
    "max_dim_robust_z",
    "norm_ratio",
    "abs_max_ratio",
    "top1_over_mean",
    "zero_ratio",
    "pos_ratio",
    "neg_ratio",
    "mean",
    "std",
]


# v19 stage-summary hooks. These are suffix-based and non-strict by default.
# Missing hooks are logged and saved, not fatal, because different OpenDriveVLA
# checkouts may wrap the same submodules under slightly different prefixes.
DEFAULT_STAGE_HOOK_SPECS = ",".join([
    # S0 / early can_bus propagation
    "s0_can_bus_mlp=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.can_bus_mlp",
    "s_pts_bbox_head=vision_tower.vision_tower.vision_model.pts_bbox_head",

    # S-stage BEV / transformer path. Some suffixes may be missing in different checkouts; missing hooks are non-fatal.
    "s_bev_transformer=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer",
    "s_bev_encoder=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.encoder",
    "s_bev_encoder_layer0=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.encoder.layers.0",
    "s_bev_encoder_layer1=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.encoder.layers.1",
    "s_bev_encoder_layer2=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.encoder.layers.2",
    "s_bev_encoder_layer3=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.encoder.layers.3",
    "s_bev_encoder_layer4=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.encoder.layers.4",
    "s_bev_encoder_layer5=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.encoder.layers.5",

    # C-stage projection path. Token-level summaries are approximated with tensor-level summaries here.
    "c_mm_projector_scene=mm_projector_scene",
    "c_mm_projector_scene_layer0=mm_projector_scene.0",
    "c_mm_projector_scene_layer1=mm_projector_scene.1",
    "c_mm_projector_scene_layer2=mm_projector_scene.2",

    # v26 online head / projector feature logging. These are online-safe summaries
    # from the current forward pass only. Missing hooks are non-fatal.
    "v26_vision_model=vision_tower.vision_tower.vision_model",
    "v26_track_decoder=vision_tower.vision_tower.vision_model.pts_bbox_head.transformer.decoder",
    "v26_track_cls_branch0=vision_tower.vision_tower.vision_model.pts_bbox_head.cls_branches.0",
    "v26_track_cls_branch5=vision_tower.vision_tower.vision_model.pts_bbox_head.cls_branches.5",
    "v26_track_reg_branch0=vision_tower.vision_tower.vision_model.pts_bbox_head.reg_branches.0",
    "v26_track_reg_branch5=vision_tower.vision_tower.vision_model.pts_bbox_head.reg_branches.5",
    "v26_track_past_traj_branch0=vision_tower.vision_tower.vision_model.pts_bbox_head.past_traj_reg_branches.0",
    "v26_track_past_traj_branch5=vision_tower.vision_tower.vision_model.pts_bbox_head.past_traj_reg_branches.5",
    "v26_seg_head=vision_tower.vision_tower.vision_model.seg_head",
    "v26_mm_projector_track=mm_projector_track",
    "v26_mm_projector_map=mm_projector_map",

    # D-stage reasoning path. Dense but still lightweight layer sampling.
    "d_llm_layer0=model.layers.0",
    "d_llm_layer6=model.layers.6",
    "d_llm_layer12=model.layers.12",
    "d_llm_layer18=model.layers.18",
    "d_llm_layer23=model.layers.23",
    # D_out boundary monitors. These are still internal neural float tensors; they are not E output-interface perturbations.
    "dout_model_norm=model.norm",
    "dout_lm_head=lm_head",
])
STAGE_SUMMARY_NUMERIC_KEYS = [
    "finite_ratio", "mean", "std", "l2_norm", "abs_max", "mean_abs", "zero_ratio",
    "pos_ratio", "neg_ratio", "top1_over_mean", "p95_abs", "p99_abs",
]


def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"[{VERSION}] wrote {path}")


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{VERSION}] wrote {path}")

def unwrap_batch_string(x):
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def normalize_sample_token(x: str) -> str:
    x = str(x)
    if x.endswith("_trajectory"):
        return x[:-11]
    return x


def sample_seed(base: int, sample_id: str, event_id: int) -> int:
    h = hashlib.md5(f"{sample_id}::{event_id}".encode("utf-8")).hexdigest()
    return int(base) ^ int(h[:8], 16)


def parse_weighted_map(spec: str, cast_key=str, cast_val=int) -> Dict[Any, int]:
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":", 1)
        out[cast_key(k.strip())] = cast_val(v.strip())
    if not out:
        raise ValueError(f"Failed to parse weighted map from: {spec}")
    return out


def weighted_choice(rng: random.Random, weight_map: Dict[Any, int]):
    items = list(weight_map.items())
    total = sum(max(0, int(w)) for _, w in items)
    if total <= 0:
        raise ValueError("weight_map must have positive total weight")
    r = rng.uniform(0, total)
    c = 0.0
    for k, w in items:
        c += max(0, int(w))
        if r <= c:
            return k
    return items[-1][0]


def median_abs_deviation(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ------------------------------
# Data accessors
# ------------------------------

def get_batch_meta(data) -> Dict[str, Any]:
    return data["uniad_data"]["img_metas"][0][0]


def get_img_tensor(data) -> torch.Tensor:
    img = data["uniad_data"]["img"]
    if isinstance(img, list):
        img = img[0]
    if not torch.is_tensor(img):
        raise RuntimeError(f"img tensor not found, got {type(img)}")
    return img


def ensure_batch_size_one(data):
    img = get_img_tensor(data)
    if img.ndim != 5:
        raise RuntimeError(f"expected image tensor ndim=5, got {img.shape}")
    if img.shape[0] != 1:
        raise RuntimeError(f"Full_S_S0_v6 currently requires batch_size=1, got {img.shape[0]}")


def get_can_bus_numpy(meta: Dict[str, Any]) -> np.ndarray:
    can_bus = np.asarray(meta.get("can_bus", np.zeros(18, dtype=np.float64)), dtype=np.float64)
    return can_bus.copy()


# -----------------------------------------------------------------------------
# v26 online driving-context feature helpers
# -----------------------------------------------------------------------------

def _to_numpy_scalar_or_array(x: Any) -> Optional[np.ndarray]:
    """Best-effort conversion for tensors/lists/scalars used in online metadata."""
    try:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    except Exception:
        return None


def get_command_id_from_data(data: Dict[str, Any]) -> Optional[int]:
    """Return UniAD high-level driving command id if available.

    Confirmed mapping from projects/mmdet3d_plugin/datasets/data_utils/trajectory_api.py:
      0 = RIGHT, 1 = LEFT, 2 = FORWARD.
    The field is online input to UniAD planning_head, not a paired clean/future label.
    """
    try:
        cmd = data.get("uniad_data", {}).get("command", None)
        if isinstance(cmd, (list, tuple)) and cmd:
            cmd = cmd[0]
        arr = _to_numpy_scalar_or_array(cmd)
        if arr is None or arr.size == 0:
            return None
        return int(np.asarray(arr).reshape(-1)[0])
    except Exception:
        return None


def compute_v26_command_context_features(data: Dict[str, Any]) -> Dict[str, float]:
    cmd = get_command_id_from_data(data)
    out: Dict[str, float] = {
        "v26__command_available": float(int(cmd is not None)),
        "v26__command_id": float(cmd if cmd is not None else -1),
        "v26__command_right": float(int(cmd == 0)),
        "v26__command_left": float(int(cmd == 1)),
        "v26__command_forward": float(int(cmd == 2)),
        "v26__command_turn": float(int(cmd in (0, 1))),
        "v26__command_unknown": float(int(cmd not in (0, 1, 2))),
    }
    return out


def _quat_yaw_wxyz(q: Any) -> float:
    """Yaw from a quaternion stored as [w, x, y, z]."""
    try:
        w, x, y, z = [float(v) for v in list(q)[:4]]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))
    except Exception:
        return 0.0


def _rotate_global_delta_to_ego(dx: float, dy: float, ego_yaw: float) -> Tuple[float, float]:
    """Rotate global delta into ego frame with x forward and y left proxy."""
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    return float(c * dx - s * dy), float(s * dx + c * dy)


_NUSC_CACHE: Dict[str, Any] = {}

def _get_nuscenes_for_v26() -> Optional[Any]:
    """Lazy NuScenes loader for online-like annotation/ego-pose context.

    This uses current-frame sample annotations and ego pose only. It does not use
    paired clean/fault labels, future frames, or GT planning trajectories.
    """
    global _NUSC_CACHE
    if "nusc" in _NUSC_CACHE:
        return _NUSC_CACHE.get("nusc")
    try:
        from nuscenes.nuscenes import NuScenes
        root = os.environ.get("NUSCENES_DATAROOT", os.path.join(PROJECT_ROOT, "data/nuscenes"))
        nusc = NuScenes(version=os.environ.get("NUSCENES_VERSION", "v1.0-trainval"), dataroot=root, verbose=False)
        _NUSC_CACHE["nusc"] = nusc
        return nusc
    except Exception as e:
        _NUSC_CACHE["error"] = repr(e)
        _NUSC_CACHE["nusc"] = None
        return None


def _extract_pred_traj_xy(pred: Any) -> np.ndarray:
    try:
        arr = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
        arr = arr[np.isfinite(arr).all(axis=1)]
        return arr
    except Exception:
        return np.zeros((0, 2), dtype=np.float64)


def compute_v26_agent_context_features(sample_token: str, pred: Any) -> Dict[str, float]:
    """Current-frame agent / clearance proxies from nuScenes annotations.

    The predicted trajectory is assumed to be in ego frame. nuScenes annotations are
    transformed from global frame to ego frame using the current sample's ego pose.
    This remains online-compatible for the current frame and does not use future GT.
    """
    out: Dict[str, float] = {
        "v26__agent_context_available": 0.0,
        "v26__agent_count": 0.0,
        "v26__agent_nearest_center_dist": math.inf,
        "v26__agent_front_nearest_dist": math.inf,
        "v26__agent_num_within_10m": 0.0,
        "v26__agent_num_within_20m": 0.0,
        "v26__agent_num_front_30m": 0.0,
        "v26__traj_agent_min_center_dist": math.inf,
        "v26__traj_agent_min_clearance_approx": math.inf,
        "v26__traj_agent_collision_risk_proxy": 0.0,
    }
    nusc = _get_nuscenes_for_v26()
    if nusc is None:
        out["v26__agent_context_error"] = 1.0
        return out
    token = normalize_sample_token(sample_token)
    try:
        sample = nusc.get("sample", token)
        lidar_token = sample.get("data", {}).get("LIDAR_TOP", "")
        if not lidar_token:
            return out
        sd = nusc.get("sample_data", lidar_token)
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        ego_t = ego.get("translation", [0.0, 0.0, 0.0])
        ego_yaw = _quat_yaw_wxyz(ego.get("rotation", [1.0, 0.0, 0.0, 0.0]))
        traj = _extract_pred_traj_xy(pred)
        agent_centers: List[Tuple[float, float, float, float]] = []
        for ann_tok in sample.get("anns", []) or []:
            ann = nusc.get("sample_annotation", ann_tok)
            cat = str(ann.get("category_name", ""))
            # Use dynamic objects and barriers/cones as obstacles; exclude clearly non-object categories.
            if not (cat.startswith("vehicle") or cat.startswith("human") or cat.startswith("pedestrian") or cat in {"movable_object.barrier", "movable_object.trafficcone", "static_object.bicycle_rack"}):
                continue
            tr = ann.get("translation", None)
            size = ann.get("size", [1.0, 1.0, 1.0])
            if tr is None:
                continue
            ex, ey = _rotate_global_delta_to_ego(float(tr[0]) - float(ego_t[0]), float(tr[1]) - float(ego_t[1]), ego_yaw)
            # Conservative approximate radius in BEV.
            sx = float(size[0]) if len(size) > 0 else 1.0
            sy = float(size[1]) if len(size) > 1 else 1.0
            radius = 0.5 * math.sqrt(max(sx, 0.0) ** 2 + max(sy, 0.0) ** 2)
            agent_centers.append((ex, ey, radius, 1.0))
        out["v26__agent_context_available"] = 1.0
        out["v26__agent_count"] = float(len(agent_centers))
        if not agent_centers:
            return out
        dists = [math.sqrt(x*x + y*y) for x, y, _, _ in agent_centers]
        out["v26__agent_nearest_center_dist"] = float(min(dists))
        out["v26__agent_num_within_10m"] = float(sum(1 for d in dists if d <= 10.0))
        out["v26__agent_num_within_20m"] = float(sum(1 for d in dists if d <= 20.0))
        front = [x for x, y, _, _ in agent_centers if x > 0.0 and abs(y) <= 8.0 and x <= 30.0]
        out["v26__agent_num_front_30m"] = float(len(front))
        out["v26__agent_front_nearest_dist"] = float(min(front)) if front else math.inf
        if traj.size:
            min_center = math.inf
            min_clear = math.inf
            for px, py in traj:
                for ax, ay, radius, _ in agent_centers:
                    d = math.sqrt((float(px) - ax) ** 2 + (float(py) - ay) ** 2)
                    min_center = min(min_center, d)
                    min_clear = min(min_clear, d - radius - 1.0)  # approximate ego half-width margin
            out["v26__traj_agent_min_center_dist"] = float(min_center)
            out["v26__traj_agent_min_clearance_approx"] = float(min_clear)
            out["v26__traj_agent_collision_risk_proxy"] = float(int(min_clear <= 0.5))
            out["v26__traj_agent_low_clearance_2m"] = float(int(min_clear <= 2.0))
            out["v26__traj_agent_low_clearance_5m"] = float(int(min_clear <= 5.0))
        return out
    except Exception:
        out["v26__agent_context_error"] = 1.0
        return out


def _angle_diff_v26(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return float(d)


def compute_v26_ego_pose_context_features(sample_token: str) -> Dict[str, float]:
    """Independent online ego-motion proxy from current and previous ego pose.

    Uses only the current sample and previous sample in the same scene. It does
    not use future frames, paired clean outputs, or harmful labels.
    """
    out: Dict[str, float] = {
        "v26__ego_pose_available": 0.0,
        "v26__ego_pose_prev_available": 0.0,
        "v26__ego_pose_speed_prev": math.nan,
        "v26__ego_pose_yaw_rate_prev": math.nan,
        "v26__ego_pose_delta_xy_prev": math.nan,
        "v26__ego_pose_delta_yaw_prev": math.nan,
        "v26__ego_pose_dt_prev": math.nan,
        "v26__ego_pose_yaw": math.nan,
        "v26__ego_pose_query_error": 1.0,
    }
    nusc = _get_nuscenes_for_v26()
    if nusc is None:
        return out
    token = normalize_sample_token(sample_token)
    try:
        sample = nusc.get("sample", token)
        lidar_token = sample.get("data", {}).get("LIDAR_TOP", "")
        if not lidar_token:
            return out
        sd = nusc.get("sample_data", lidar_token)
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        trans = ego.get("translation", [])
        yaw = _quat_yaw_wxyz(ego.get("rotation", [1.0, 0.0, 0.0, 0.0]))
        out.update({
            "v26__ego_pose_available": 1.0,
            "v26__ego_pose_query_error": 0.0,
            "v26__ego_pose_yaw": float(yaw),
        })
        prev_token = str(sample.get("prev", "") or "")
        if prev_token:
            prev_sample = nusc.get("sample", prev_token)
            prev_lidar = prev_sample.get("data", {}).get("LIDAR_TOP", "")
            if prev_lidar:
                psd = nusc.get("sample_data", prev_lidar)
                pego = nusc.get("ego_pose", psd["ego_pose_token"])
                ptrans = pego.get("translation", [])
                pyaw = _quat_yaw_wxyz(pego.get("rotation", [1.0, 0.0, 0.0, 0.0]))
                if len(trans) >= 2 and len(ptrans) >= 2:
                    dx = float(trans[0]) - float(ptrans[0])
                    dy = float(trans[1]) - float(ptrans[1])
                    dxy = math.sqrt(dx * dx + dy * dy)
                    dt = (float(sample.get("timestamp", 0.0)) - float(prev_sample.get("timestamp", 0.0))) / 1e6
                    dyaw = _angle_diff_v26(yaw, pyaw)
                    out.update({
                        "v26__ego_pose_prev_available": 1.0,
                        "v26__ego_pose_delta_xy_prev": float(dxy),
                        "v26__ego_pose_delta_yaw_prev": float(dyaw),
                        "v26__ego_pose_dt_prev": float(dt),
                        "v26__ego_pose_speed_prev": float(dxy / dt) if dt and dt > 0 else math.nan,
                        "v26__ego_pose_yaw_rate_prev": float(dyaw / dt) if dt and dt > 0 else math.nan,
                    })
    except Exception:
        out["v26__ego_pose_query_error"] = 1.0
    return out


def set_can_bus_numpy(meta: Dict[str, Any], arr: np.ndarray):
    meta["can_bus"] = np.asarray(arr, dtype=np.float64).copy()


def clone_for_sample_isolation(obj: Any) -> Any:
    """Recursively clone mutable per-sample objects before fault injection.

    This prevents an injected sample from mutating objects that may be reused by
    the dataloader, dataset cache, hooks, or later no-fault samples in a mixed run.
    """
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: clone_for_sample_isolation(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_for_sample_isolation(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_for_sample_isolation(v) for v in obj)
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj


def snapshot_can_bus_numpy(data: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        return get_can_bus_numpy(get_batch_meta(data)).copy()
    except Exception:
        return None


def restore_can_bus_snapshot(data: Dict[str, Any], snapshot: Optional[np.ndarray]) -> int:
    if snapshot is None:
        return 0
    try:
        set_can_bus_numpy(get_batch_meta(data), snapshot)
        return 1
    except Exception:
        return 0


def max_abs_can_bus_delta(data: Dict[str, Any], snapshot: Optional[np.ndarray]) -> float:
    if snapshot is None:
        return math.nan
    try:
        cur = get_can_bus_numpy(get_batch_meta(data))
        if cur.shape != snapshot.shape:
            return math.inf
        return float(np.max(np.abs(cur.astype(np.float64) - snapshot.astype(np.float64))))
    except Exception:
        return math.nan


def reset_runtime_capture_state(can_bus_capture: Any, stage_captures: Dict[str, Any]) -> Dict[str, int]:
    status = {"can_bus_capture_reset": 0, "stage_captures_reset": 0}
    try:
        can_bus_capture.reset()
        status["can_bus_capture_reset"] = 1
    except Exception:
        pass
    try:
        reset_stage_captures(stage_captures)
        status["stage_captures_reset"] = 1
    except Exception:
        pass
    return status


def clear_cuda_between_samples(enabled: bool) -> None:
    if not enabled:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


TEMPORAL_STATE_NONE_ATTRS = [
    "prev_bev",
    "prev_bev_list",
    "prev_scene_token",
    "scene_token",
    "last_scene_token",
    "cached_bev",
    "bev_cache",
    "bev_embed_cache",
    "past_key_values",
]

TEMPORAL_STATE_ZERO_ATTRS = [
    "prev_pos",
    "prev_angle",
]


def reset_model_temporal_runtime_state(model_engine) -> Dict[str, Any]:
    """Best-effort reset of model-internal temporal/cache state.

    v22 promotes this from a diagnostic switch to the default evaluation mechanism.
    It prevents injected samples from contaminating later no-fault samples through
    BEVFormer/UniAD temporal runtime state such as prev_bev or prev_frame_info.
    """
    status: Dict[str, Any] = {
        "enabled": 1,
        "modules_seen": 0,
        "prev_frame_info_reset": 0,
        "attr_reset": 0,
        "reset_errors": 0,
        "reset_attr_names": [],
    }

    try:
        model = unwrap_model_for_modules(model_engine)
    except Exception:
        model = getattr(model_engine, "module", model_engine)

    try:
        modules = list(model.modules()) if hasattr(model, "modules") else [model]
    except Exception:
        modules = [model]

    for module in modules:
        status["modules_seen"] += 1

        try:
            if hasattr(module, "prev_frame_info"):
                pfi = getattr(module, "prev_frame_info")
                if isinstance(pfi, dict):
                    if "prev_bev" in pfi:
                        pfi["prev_bev"] = None
                    if "scene_token" in pfi:
                        pfi["scene_token"] = None
                    if "prev_pos" in pfi:
                        pfi["prev_pos"] = 0
                    if "prev_angle" in pfi:
                        pfi["prev_angle"] = 0
                    status["prev_frame_info_reset"] += 1
        except Exception:
            status["reset_errors"] += 1

        for name in TEMPORAL_STATE_NONE_ATTRS:
            try:
                if hasattr(module, name):
                    val = getattr(module, name)
                    if not isinstance(val, (torch.nn.Module, torch.nn.Parameter)):
                        setattr(module, name, None)
                        status["attr_reset"] += 1
                        status["reset_attr_names"].append(name)
            except Exception:
                status["reset_errors"] += 1

        for name in TEMPORAL_STATE_ZERO_ATTRS:
            try:
                if hasattr(module, name):
                    val = getattr(module, name)
                    if not isinstance(val, (torch.nn.Module, torch.nn.Parameter)):
                        setattr(module, name, 0)
                        status["attr_reset"] += 1
                        status["reset_attr_names"].append(name)
            except Exception:
                status["reset_errors"] += 1

    status["reset_attr_names"] = sorted(set(status["reset_attr_names"]))
    return status


def disabled_model_temporal_runtime_state_status() -> Dict[str, Any]:
    return {
        "enabled": 0,
        "modules_seen": 0,
        "prev_frame_info_reset": 0,
        "attr_reset": 0,
        "reset_errors": 0,
        "reset_attr_names": [],
    }


def temporal_state_reset_enabled(args) -> bool:
    return bool(getattr(args, "reset_model_state_between_samples", True))


def dtype_exp_info_numpy(dtype: np.dtype) -> Tuple[int, int]:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.float16):
        return (10, 5)
    if dt == np.dtype(np.float32):
        return (23, 8)
    if dt == np.dtype(np.float64):
        return (52, 11)
    raise TypeError(f"Unsupported numpy dtype for bit fault: {dt}")


def dtype_exp_info_torch(dtype: torch.dtype) -> Tuple[int, int]:
    """Return (mantissa_bits, exponent_bits) for floating torch dtypes.

    This function is used by S1 activation bit mutation.  The initial v32B
    mixed run missed this helper, so every S1 planned fault failed with
    ``NameError: dtype_exp_info_torch is not defined`` while S0 can_bus faults
    still worked.  Keeping the torch-specific helper separate from the numpy
    helper avoids dtype comparison ambiguity and supports fp16/bf16/fp32/fp64.
    """
    if dtype == torch.float16:
        return (10, 5)
    if dtype == torch.bfloat16:
        return (7, 8)
    if dtype == torch.float32:
        return (23, 8)
    if dtype == torch.float64:
        return (52, 11)
    raise TypeError(f"Unsupported torch dtype for bit fault: {dtype}")


def choose_exponent_span(lsb_start: int, exp_bits: int, width: int, rng: random.Random) -> List[int]:
    width = int(width)
    if width < 1 or width > exp_bits:
        raise ValueError(f"Invalid exponent fault width={width} for exp_bits={exp_bits}")
    start = rng.randint(lsb_start, lsb_start + exp_bits - width)
    return list(range(start, start + width))


def apply_uint_mask_signed(x: int, bits_total: int) -> int:
    sign_limit = 1 << (bits_total - 1)
    full = 1 << bits_total
    return x - full if x >= sign_limit else x


def enumerate_exponent_spans(lsb_start: int, exp_bits: int, width: int) -> List[List[int]]:
    return [list(range(start, start + width)) for start in range(lsb_start, lsb_start + exp_bits - width + 1)]


def valid_spans_for_uint(uint: int, bits_total: int, mantissa_bits: int, exp_bits: int, width: int, bit_op: str) -> List[List[int]]:
    spans = enumerate_exponent_spans(mantissa_bits, exp_bits, width)
    valid = []
    for span in spans:
        changed = False
        for bit in span:
            cur = (uint >> bit) & 1
            if bit_op == "flip":
                changed = True
                break
            if bit_op == "stuck0" and cur == 1:
                changed = True
                break
            if bit_op == "stuck1" and cur == 0:
                changed = True
                break
        if changed:
            valid.append(span)
    return valid


def scalar_uint_info_torch(flat_tensor: torch.Tensor, flat_idx: int) -> Tuple[int, int, int, int]:
    if flat_tensor.dtype == torch.float16:
        bits_total = 16
        intval = int(flat_tensor.view(torch.int16)[flat_idx].item())
    elif flat_tensor.dtype == torch.bfloat16:
        bits_total = 16
        intval = int(flat_tensor.view(torch.int16)[flat_idx].item())
    elif flat_tensor.dtype == torch.float32:
        bits_total = 32
        intval = int(flat_tensor.view(torch.int32)[flat_idx].item())
    elif flat_tensor.dtype == torch.float64:
        bits_total = 64
        intval = int(flat_tensor.view(torch.int64)[flat_idx].item())
    else:
        raise TypeError(f"Unsupported dtype {flat_tensor.dtype}")
    mantissa_bits, exp_bits = dtype_exp_info_torch(flat_tensor.dtype)
    uint = intval & ((1 << bits_total) - 1)
    return uint, bits_total, mantissa_bits, exp_bits


def scalar_uint_info_numpy(arr: np.ndarray, flat_idx: int) -> Tuple[int, int, int, int]:
    flat = arr.reshape(-1)
    dt = flat.dtype
    if dt == np.float16:
        bits_total = 16
        uint = int(flat.view(np.uint16)[flat_idx])
    elif dt == np.float32:
        bits_total = 32
        uint = int(flat.view(np.uint32)[flat_idx])
    elif dt == np.float64:
        bits_total = 64
        uint = int(flat.view(np.uint64)[flat_idx])
    else:
        raise TypeError(f"Unsupported numpy dtype {dt}")
    mantissa_bits, exp_bits = dtype_exp_info_numpy(dt)
    return uint, bits_total, mantissa_bits, exp_bits


def mutate_one_scalar_tensor_exponent_bits_(flat_tensor: torch.Tensor, flat_idx: int, targets: List[int], bit_op: str):
    if flat_tensor.ndim != 1:
        raise ValueError("flat_tensor must be 1-D")
    if not flat_tensor.dtype.is_floating_point:
        raise TypeError(f"expected floating tensor, got {flat_tensor.dtype}")
    orig = flat_tensor[flat_idx].item()
    uint, bits_total, _, _ = scalar_uint_info_torch(flat_tensor, flat_idx)
    for bit in targets:
        mask = 1 << bit
        if bit_op == "flip":
            uint ^= mask
        elif bit_op == "stuck0":
            uint &= ~mask
        elif bit_op == "stuck1":
            uint |= mask
        else:
            raise ValueError(f"Unsupported bit_op {bit_op}")
    new_int = apply_uint_mask_signed(uint, bits_total)
    if bits_total == 16:
        flat_tensor.view(torch.int16)[flat_idx] = new_int
    elif bits_total == 32:
        flat_tensor.view(torch.int32)[flat_idx] = new_int
    else:
        flat_tensor.view(torch.int64)[flat_idx] = new_int
    new_val = flat_tensor[flat_idx].item()
    return {"flat_idx": int(flat_idx), "bit_positions_lsb": [int(b) for b in targets], "before": safe_float(orig), "after": safe_float(new_val)}


def mutate_one_scalar_numpy_exponent_bits_(arr: np.ndarray, flat_idx: int, targets: List[int], bit_op: str):
    flat = arr.reshape(-1)
    dt = flat.dtype
    orig = flat[flat_idx].item()
    uint, bits_total, _, _ = scalar_uint_info_numpy(arr, flat_idx)
    if dt == np.float16:
        raw = flat.view(np.uint16)
    elif dt == np.float32:
        raw = flat.view(np.uint32)
    elif dt == np.float64:
        raw = flat.view(np.uint64)
    else:
        raise TypeError(f"Unsupported numpy dtype {dt}")
    for bit in targets:
        mask = 1 << bit
        if bit_op == "flip":
            uint ^= mask
        elif bit_op == "stuck0":
            uint &= ~mask
        elif bit_op == "stuck1":
            uint |= mask
        else:
            raise ValueError(f"Unsupported bit_op {bit_op}")
    if bits_total == 16:
        raw[flat_idx] = np.uint16(uint)
    elif bits_total == 32:
        raw[flat_idx] = np.uint32(uint)
    else:
        raw[flat_idx] = np.uint64(uint)
    new_val = flat[flat_idx].item()
    return {"flat_idx": int(flat_idx), "bit_positions_lsb": [int(b) for b in targets], "before": safe_float(orig), "after": safe_float(new_val)}


def rel_change(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(a), 1e-8)
    out = np.abs(b - a) / denom
    finite = out[np.isfinite(out)]
    return float(finite.max()) if finite.size else math.inf


def canbus_group_name(flat_idx: int) -> str:
    for gname, idxs in CANBUS_GROUPS.items():
        if int(flat_idx) in idxs:
            return gname
    return "unknown"


def weighted_canbus_candidate_indices(arr: np.ndarray, rng: random.Random) -> List[int]:
    candidates = [int(i) for i in np.flatnonzero(np.isfinite(arr.reshape(-1)))]
    keys = []
    for idx in candidates:
        g = canbus_group_name(idx)
        w = float(CANBUS_GROUP_TARGET_WEIGHTS.get(g, 1))
        # smaller random key means higher priority; high-weight groups are sampled earlier on average.
        keys.append((rng.random() / max(w, 1e-6), idx))
    keys.sort(key=lambda x: x[0])
    return [idx for _, idx in keys]


def canbus_angle_consistency_error(vec: np.ndarray) -> float:
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    if x.size < 18:
        return math.inf
    rad = x[16]
    deg = x[17]
    if not np.isfinite(rad) or not np.isfinite(deg):
        return math.inf
    return float(abs(deg - rad * 180.0 / math.pi))


def canbus_quat_norm_error(vec: np.ndarray) -> float:
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    if x.size < 7:
        return math.inf
    q = x[3:7]
    if not np.isfinite(q).all():
        return math.inf
    return float(abs(np.linalg.norm(q) - 1.0))


def impactful_can_bus(before: np.ndarray, after: np.ndarray, flat_idx: int) -> Tuple[bool, Dict[str, Any]]:
    before = np.asarray(before, dtype=np.float64).reshape(-1)
    after = np.asarray(after, dtype=np.float64).reshape(-1)
    gname = canbus_group_name(flat_idx)
    idxs = np.asarray(CANBUS_GROUPS.get(gname, [flat_idx]), dtype=np.int64)
    b = before[idxs]
    a = after[idxs]
    abs_diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    rel = rel_change(b, a)
    info = {"group": gname, "group_abs_diff_max": abs_diff, "group_rel_change_max": rel}
    if not np.isfinite(after).all() or not np.isfinite(a).all():
        info["reason"] = "nonfinite_after"
        return True, info
    if gname == "rotation_quat":
        bq = canbus_quat_norm_error(before)
        aq = canbus_quat_norm_error(after)
        info["quat_norm_error_before"] = bq
        info["quat_norm_error_after"] = aq
        if aq > max(bq + 0.05, 0.15) or rel >= 0.08:
            info["reason"] = "quat_structure_shift"
            return True, info
    if gname == "patch_angle_pair":
        be = canbus_angle_consistency_error(before)
        ae = canbus_angle_consistency_error(after)
        info["angle_consistency_before"] = be
        info["angle_consistency_after"] = ae
        if ae > max(be + 1e-3, 1e-3) or abs_diff >= 0.1 or rel >= 0.05:
            info["reason"] = "patch_angle_consistency_shift"
            return True, info
    if gname == "translation_xyz":
        if abs_diff >= 1.0 or rel >= 0.05:
            info["reason"] = "translation_shift"
            return True, info
    if gname in {"lp_9", "lp_13"}:
        if abs_diff >= 0.5 and rel >= 0.08:
            info["reason"] = "large_positive_shift"
            return True, info
    if gname == "small_signed_state":
        if abs_diff >= 0.03 and rel >= 0.35:
            info["reason"] = "small_signed_shift"
            return True, info
    if gname == "near_zero_slots":
        if abs_diff >= 0.10 and rel >= 1.0:
            info["reason"] = "near_zero_shift"
            return True, info
    if abs_diff > 1e-6 and rel >= MIN_EFFECTIVE_REL:
        info["reason"] = "group_relative_shift"
        return True, info
    info["reason"] = "too_small"
    return False, info

def effective_value_changed(before: float, after: float) -> bool:
    if not (np.isfinite(before) and np.isfinite(after)):
        return True
    return abs(float(after) - float(before)) > (1e-8 + 1e-5 * abs(float(before)))


@dataclass
class FaultEventSpec:
    target: str
    bit_count: int
    bit_op: str
    event_id: int


def parse_s1_target_specs(raw: str) -> List[Tuple[str, str]]:
    text = str(raw or "").strip()
    if not text or text == "__DEFAULT__":
        text = DEFAULT_V32_S1_TARGET_SPECS
    out: List[Tuple[str, str]] = []
    for part in [x.strip() for x in text.split(",") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"Invalid S1 target spec '{part}', expected name=suffix")
        name, suffix = part.split("=", 1)
        name = name.strip(); suffix = suffix.strip()
        if not name or not suffix:
            raise ValueError(f"Invalid S1 target spec '{part}', empty name/suffix")
        out.append((name, suffix))
    return out


def build_s1_target_weights(
    rng: random.Random,
    policy: str,
    primary_name: str,
    primary_weight: int,
    broad_weights_spec: str,
) -> Dict[str, int]:
    broad = parse_weighted_map(broad_weights_spec, cast_key=str, cast_val=int)
    if not broad:
        broad = parse_weighted_map(DEFAULT_V32_S1_BROAD_TARGET_WEIGHTS, cast_key=str, cast_val=int)
    policy = str(policy or "weighted").lower()
    if policy == "primary_only":
        return {primary_name: 1}
    if policy == "uniform_all":
        out = dict(broad)
        out[primary_name] = out.get(primary_name, 1)
        return {k: 1 for k in out.keys()}
    if policy == "weighted":
        out = {primary_name: max(1, int(primary_weight))}
        for k, v in broad.items():
            if k == primary_name:
                continue
            out[k] = max(1, int(v))
        return out
    raise ValueError(f"Unknown --s1-target-policy={policy}")


def parse_v44_named_weights(raw: str, default_raw: str) -> Dict[str, int]:
    text = str(raw or "").strip()
    if not text or text == "__DEFAULT__":
        text = default_raw
    out = parse_weighted_map(text, cast_key=str, cast_val=int)
    return {str(k): max(1, int(v)) for k, v in out.items() if str(k).strip()}


def v44_allowed_strict_scopes(allowed_targets: List[str]) -> List[str]:
    scopes: List[str] = []
    if "can_bus" in allowed_targets:
        scopes.append("S0_only")
    if "s1_activation" in allowed_targets:
        scopes.append("S1_only")
    if "c_activation" in allowed_targets:
        scopes.append("C_only")
    if "d_activation" in allowed_targets:
        scopes.append("D_only")
    if "e_output" in allowed_targets:
        scopes.append("E_only")
    return scopes


class BlindFaultPlan:
    """v32 blind fault plan.

    Backward compatible with v31 S0-only mode. When allowed_targets contains
    s1_activation or --target-mode A1_can_bus_s1_mixed is used, the plan can sample S0_only, S1_only, or S0S1_mixed depending on
    --mixed-fault-scope-weights. In the corrected v35 single-fault runner,
    this is restricted to S0_only/S1_only and --fault-event-count-weights=1:1,
    so each injected sample contains exactly one fault event. Bit-count and
    bit-op remain sampled per event.
    """
    def __init__(
        self,
        subset_ids: List[str],
        allowed_targets: List[str],
        experiment_seed: int,
        positive_min: int,
        positive_max: int,
        cardinality_weights: Dict[int, int],
        bit_op_weights: Dict[str, int],
        scope_weights: Optional[Dict[str, int]] = None,
        event_count_weights: Optional[Dict[int, int]] = None,
        s1_target_weights: Optional[Dict[str, int]] = None,
        c_target_weights: Optional[Dict[str, int]] = None,
        d_target_weights: Optional[Dict[str, int]] = None,
        e_target_weights: Optional[Dict[str, int]] = None,
        mixed_enabled: bool = False,
        strict_single_fault: bool = False,
    ):
        self.subset_ids = [normalize_sample_token(x) for x in subset_ids]
        self.allowed_targets = list(allowed_targets)
        self.rng = random.Random(int(experiment_seed))
        self.cardinality_weights = cardinality_weights
        self.bit_op_weights = bit_op_weights
        self.strict_single_fault = bool(strict_single_fault)
        self.scope_weights = scope_weights or {"S0_only": 1}
        self.event_count_weights = event_count_weights or {1: 1}
        self.s1_target_weights = s1_target_weights or {DEFAULT_V32_S1_PRIMARY_TARGET: 1}
        self.c_target_weights = c_target_weights or {"c_mm_projector_scene": 1}
        self.d_target_weights = d_target_weights or {"d_llm_layer12": 1}
        self.e_target_weights = e_target_weights or {"e_traj_y_shift": 1}
        self.mixed_enabled = bool(mixed_enabled or any(t in self.allowed_targets for t in ["s1_activation", "c_activation", "d_activation", "e_output"]))

        # v35 corrected single-fault mode:
        # one selected sample receives exactly one fault event, and the event is
        # located either in S0 or in S1. S0S1_mixed and multi-event samples are
        # forbidden regardless of the legacy CLI defaults. Bit-count still
        # follows cardinality_weights, so one event may be 1/2/3-bit.
        if self.strict_single_fault:
            allowed_scopes = set(v44_allowed_strict_scopes(self.allowed_targets))
            cleaned_scope_weights = {
                str(k): int(v) for k, v in self.scope_weights.items()
                if str(k) in allowed_scopes and int(v) > 0
            }
            if not cleaned_scope_weights:
                cleaned_scope_weights = {scope: 1 for scope in sorted(allowed_scopes)}
            if not cleaned_scope_weights:
                raise ValueError(f"No legal strict single-fault scopes for allowed_targets={self.allowed_targets}")
            self.scope_weights = cleaned_scope_weights
            self.event_count_weights = {1: 1}
            self.mixed_enabled = True
        n_total = len(self.subset_ids)
        target_count = min(n_total, self.rng.randint(int(positive_min), int(positive_max)))
        self.positive_ids = set(self.rng.sample(self.subset_ids, target_count))
        self.sample_specs: Dict[str, List[FaultEventSpec]] = {}
        self.sample_scopes: Dict[str, str] = {}
        event_id = 0
        for sid in self.subset_ids:
            if sid not in self.positive_ids:
                self.sample_specs[sid] = []
                self.sample_scopes[sid] = "no_fault"
                continue
            if self.mixed_enabled:
                scope = str(weighted_choice(self.rng, self.scope_weights))
                if scope not in {"S0_only", "S1_only", "S0S1_mixed", "C_only", "D_only", "E_only"}:
                    raise ValueError(f"Invalid mixed fault scope '{scope}'")
            else:
                scope = "S0_only"
            specs: List[FaultEventSpec] = []
            if self.strict_single_fault:
                if scope == "S0_only":
                    bit_count = int(weighted_choice(self.rng, self.cardinality_weights))
                    bit_op = str(weighted_choice(self.rng, self.bit_op_weights))
                    specs.append(FaultEventSpec(target="can_bus", bit_count=bit_count, bit_op=bit_op, event_id=event_id))
                    event_id += 1
                elif scope == "S1_only":
                    tgt = str(weighted_choice(self.rng, self.s1_target_weights))
                    bit_count = int(weighted_choice(self.rng, self.cardinality_weights))
                    bit_op = str(weighted_choice(self.rng, self.bit_op_weights))
                    specs.append(FaultEventSpec(target=f"s1:{tgt}", bit_count=bit_count, bit_op=bit_op, event_id=event_id))
                    event_id += 1
                elif scope == "C_only":
                    tgt = str(weighted_choice(self.rng, self.c_target_weights))
                    bit_count = int(weighted_choice(self.rng, self.cardinality_weights))
                    bit_op = str(weighted_choice(self.rng, self.bit_op_weights))
                    specs.append(FaultEventSpec(target=f"c:{tgt}", bit_count=bit_count, bit_op=bit_op, event_id=event_id))
                    event_id += 1
                elif scope == "D_only":
                    tgt = str(weighted_choice(self.rng, self.d_target_weights))
                    bit_count = int(weighted_choice(self.rng, self.cardinality_weights))
                    bit_op = str(weighted_choice(self.rng, self.bit_op_weights))
                    specs.append(FaultEventSpec(target=f"d:{tgt}", bit_count=bit_count, bit_op=bit_op, event_id=event_id))
                    event_id += 1
                elif scope == "E_only":
                    tgt = str(weighted_choice(self.rng, self.e_target_weights))
                    bit_count = int(weighted_choice(self.rng, self.cardinality_weights))
                    bit_op = str(weighted_choice(self.rng, self.bit_op_weights))
                    specs.append(FaultEventSpec(target=f"e:{tgt}", bit_count=bit_count, bit_op=bit_op, event_id=event_id))
                    event_id += 1
                else:
                    raise ValueError(f"strict_single_fault forbids fault scope '{scope}'")
            else:
                if scope in {"S0_only", "S0S1_mixed"}:
                    n_s0 = int(weighted_choice(self.rng, self.event_count_weights))
                    for _ in range(max(1, n_s0)):
                        bit_count = int(weighted_choice(self.rng, self.cardinality_weights))
                        bit_op = str(weighted_choice(self.rng, self.bit_op_weights))
                        specs.append(FaultEventSpec(target="can_bus", bit_count=bit_count, bit_op=bit_op, event_id=event_id))
                        event_id += 1
                if scope in {"S1_only", "S0S1_mixed"}:
                    n_s1 = int(weighted_choice(self.rng, self.event_count_weights))
                    for _ in range(max(1, n_s1)):
                        tgt = str(weighted_choice(self.rng, self.s1_target_weights))
                        bit_count = int(weighted_choice(self.rng, self.cardinality_weights))
                        bit_op = str(weighted_choice(self.rng, self.bit_op_weights))
                        specs.append(FaultEventSpec(target=f"s1:{tgt}", bit_count=bit_count, bit_op=bit_op, event_id=event_id))
                        event_id += 1
            self.sample_specs[sid] = specs
            self.sample_scopes[sid] = scope

    def get_specs(self, sample_id: str) -> Tuple[int, List[FaultEventSpec]]:
        sid = normalize_sample_token(sample_id)
        specs = self.sample_specs.get(sid, [])
        return int(len(specs) > 0), list(specs)

    def get_scope(self, sample_id: str) -> str:
        sid = normalize_sample_token(sample_id)
        return str(self.sample_scopes.get(sid, "no_fault"))

    def summary(self) -> Dict[str, Any]:
        scope_counts: Dict[str, int] = {}
        spec_counts: Dict[str, int] = {}
        for sid, scope in self.sample_scopes.items():
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
            spec_counts[str(len(self.sample_specs.get(sid, [])))] = spec_counts.get(str(len(self.sample_specs.get(sid, []))), 0) + 1
        return {
            "version": VERSION,
            "positive_count": int(len(self.positive_ids)),
            "positive_sample_ids": sorted(self.positive_ids),
            "allowed_targets": list(self.allowed_targets),
            "mixed_enabled": int(self.mixed_enabled),
            "strict_single_fault": int(self.strict_single_fault),
            "scope_weights": {str(k): int(v) for k, v in self.scope_weights.items()},
            "scope_counts": scope_counts,
            "spec_count_distribution": spec_counts,
            "cardinality_weights": {str(k): int(v) for k, v in self.cardinality_weights.items()},
            "bit_op_weights": {str(k): int(v) for k, v in self.bit_op_weights.items()},
            "event_count_weights": {str(k): int(v) for k, v in self.event_count_weights.items()},
            "s1_target_weights": {str(k): int(v) for k, v in self.s1_target_weights.items()},
            "c_target_weights": {str(k): int(v) for k, v in self.c_target_weights.items()},
            "d_target_weights": {str(k): int(v) for k, v in self.d_target_weights.items()},
            "e_target_weights": {str(k): int(v) for k, v in self.e_target_weights.items()},
            "sample_specs": {
                sid: [vars(spec) for spec in specs]
                for sid, specs in self.sample_specs.items()
                if specs
            },
            "sample_scopes": {sid: scope for sid, scope in self.sample_scopes.items() if scope != "no_fault"},
        }




# ------------------------------
# Feature extraction
# ------------------------------

def object_can_bus(data) -> np.ndarray:
    meta = get_batch_meta(data)
    return get_can_bus_numpy(meta)


def finite_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 1.0
    return float(np.isfinite(arr).mean())


def robust_stats_matrix(vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(vectors, dtype=np.float64)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med[None, :]), axis=0)
    mad = np.where(mad < 1e-12, 1e-12, mad)
    return med, mad


def robust_z(vec: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    x = np.asarray(vec, dtype=np.float64)
    return np.abs(x - med) / mad


def vector_basic(vec: np.ndarray) -> Dict[str, float]:
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    fr = finite_ratio(x)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "finite_ratio": fr,
            "nonfinite_ratio": 1.0 - fr,
            "abs_max": math.inf,
            "mean_abs": math.inf,
            "norm2": math.inf,
            "top1_over_mean": math.inf,
        }
    absx = np.abs(finite)
    mean_abs = float(absx.mean())
    abs_max = float(absx.max())
    norm2 = float(np.linalg.norm(finite))
    top1_over_mean = float(abs_max / (mean_abs + 1e-12))
    return {
        "finite_ratio": fr,
        "nonfinite_ratio": 1.0 - fr,
        "abs_max": abs_max,
        "mean_abs": mean_abs,
        "norm2": norm2,
        "top1_over_mean": top1_over_mean,
    }


def canbus_group_features(cb: np.ndarray, dim_stats: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    cb = np.asarray(cb, dtype=np.float64).reshape(-1)
    out = {}
    group_stats = (dim_stats or {}).get("can_bus_groups", {})
    for gname, idxs in CANBUS_GROUPS.items():
        gvec = cb[np.asarray(idxs, dtype=np.int64)]
        basic = vector_basic(gvec)
        med = np.asarray(group_stats.get(gname, {}).get("median", np.zeros_like(gvec)), dtype=np.float64)
        mad = np.asarray(group_stats.get(gname, {}).get("mad", np.ones_like(gvec)), dtype=np.float64)
        z = robust_z(gvec, med, mad)
        finite_z = z[np.isfinite(z)]
        max_z = float(finite_z.max()) if finite_z.size else math.inf
        median_norm = float(group_stats.get(gname, {}).get("median_norm", 1.0))
        median_absmax = float(group_stats.get(gname, {}).get("median_absmax", 1.0))
        feats = {
            **basic,
            "max_dim_robust_z": max_z,
            "norm_ratio": float(basic["norm2"] / max(median_norm, 1e-12)) if np.isfinite(basic["norm2"]) else math.inf,
            "abs_max_ratio": float(basic["abs_max"] / max(median_absmax, 1e-12)) if np.isfinite(basic["abs_max"]) else math.inf,
        }
        if gname == "rotation_quat":
            feats["quat_norm_error"] = canbus_quat_norm_error(cb)
        if gname == "patch_angle_pair":
            feats["angle_consistency_error"] = canbus_angle_consistency_error(cb)
        out[gname] = feats
    return out


def canbus_direct_use_features(cb: np.ndarray, dim_stats: Optional[Dict[str, Any]]) -> Dict[str, float]:
    cb = np.asarray(cb, dtype=np.float64).reshape(-1)
    direct = cb[np.asarray(CANBUS_DIRECT_IDXS, dtype=np.int64)]
    basic = vector_basic(direct)
    st = (dim_stats or {}).get("can_bus_direct", {})
    med = np.asarray(st.get("median", np.zeros_like(direct)), dtype=np.float64)
    mad = np.asarray(st.get("mad", np.ones_like(direct)), dtype=np.float64)
    z = robust_z(direct, med, mad)
    finite_z = z[np.isfinite(z)]
    max_z = float(finite_z.max()) if finite_z.size else math.inf
    median_norm = float(st.get("median_norm", 1.0))
    xy = cb[[0,1]] if cb.size >= 2 else np.zeros(2, dtype=np.float64)
    patch = cb[[16,17]] if cb.size >= 18 else np.zeros(2, dtype=np.float64)
    xy_st = (dim_stats or {}).get("can_bus_xy", {})
    patch_st = (dim_stats or {}).get("can_bus_patch", {})
    xy_z = robust_z(xy, np.asarray(xy_st.get("median", np.zeros_like(xy)), dtype=np.float64), np.asarray(xy_st.get("mad", np.ones_like(xy)), dtype=np.float64))
    patch_z = robust_z(patch, np.asarray(patch_st.get("median", np.zeros_like(patch)), dtype=np.float64), np.asarray(patch_st.get("mad", np.ones_like(patch)), dtype=np.float64))
    return {
        "finite_ratio": basic["finite_ratio"],
        "direct_max_dim_robust_z": max_z,
        "direct_norm_ratio": float(basic["norm2"] / max(median_norm, 1e-12)) if np.isfinite(basic["norm2"]) else math.inf,
        "xy_max_dim_robust_z": float(np.nanmax(xy_z)) if xy_z.size else math.inf,
        "patch_max_dim_robust_z": float(np.nanmax(patch_z)) if patch_z.size else math.inf,
        "angle_consistency_error": canbus_angle_consistency_error(cb),
    }




def apply_one_fault(data, sample_id: str, spec: FaultEventSpec, base_seed: int) -> Dict[str, Any]:
    """Apply one can_bus exponent-bit fault only."""
    if spec.target != "can_bus":
        raise ValueError(f"Full_S_S0_v26_online_heads_feature_rich only supports target=can_bus, got {spec.target}")

    rng = random.Random(sample_seed(base_seed, sample_id, spec.event_id))
    info: Dict[str, Any] = {
        "target": spec.target,
        "bit_count": int(spec.bit_count),
        "bit_op": spec.bit_op,
        "event_id": int(spec.event_id),
        "field": "exponent",
        "applied": 0,
        "effective": 0,
    }

    meta = get_batch_meta(data)
    arr = get_can_bus_numpy(meta)
    before_full = arr.copy()
    candidate_indices = weighted_canbus_candidate_indices(arr, rng)
    tries = 0
    for flat_idx in candidate_indices:
        uint, bits_total, mantissa_bits, exp_bits = scalar_uint_info_numpy(arr, flat_idx)
        spans = valid_spans_for_uint(uint, bits_total, mantissa_bits, exp_bits, spec.bit_count, spec.bit_op)
        rng.shuffle(spans)
        for span in spans:
            tries += 1
            if tries > MAX_INJECTION_ATTEMPTS:
                break
            work = before_full.copy()
            fault_info = mutate_one_scalar_numpy_exponent_bits_(work, flat_idx, span, spec.bit_op)
            changed = effective_value_changed(fault_info["before"], fault_info["after"])
            impactful, impact_info = impactful_can_bus(before_full, work, flat_idx)
            if changed and impactful:
                set_can_bus_numpy(meta, work)
                info.update({
                    "applied": 1,
                    "effective": 1,
                    "effective_value_changed": 1,
                    **fault_info,
                    "impact": impact_info,
                })
                return info
        if tries > MAX_INJECTION_ATTEMPTS:
            break
    info["reason"] = "no_impactful_candidate"
    info["tries"] = tries
    return info

def canbus_group_feature_list(gname: str) -> List[str]:
    return list(CANBUS_GROUP_FEATURE_MAP[gname])


def compute_feature_threshold(
    values: List[float],
    *,
    kind: str = "range",
    mad_scale: float = 6.0,
    strong_mad_scale: float = 10.0,
    p_low: float = 0.005,
    p_high: float = 0.995,
    p_vlow: float = 0.001,
    p_vhigh: float = 0.999,
) -> Optional[Dict[str, Any]]:
    vals = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None
    med = float(np.median(vals))
    mad = median_abs_deviation(vals)
    p001 = float(np.quantile(vals, p_vlow))
    p005 = float(np.quantile(vals, p_low))
    p995 = float(np.quantile(vals, p_high))
    p999 = float(np.quantile(vals, p_vhigh))
    if kind == "lower_only":
        lower = min(p005, med - mad_scale * mad)
        strong_lower = min(p001, med - strong_mad_scale * mad)
        return {"kind": "lower_only", "lower": lower, "strong_lower": strong_lower, "median": med, "mad": mad}
    if kind == "upper_only":
        upper = max(p995, med + mad_scale * mad)
        strong_upper = max(p999, med + strong_mad_scale * mad)
        return {"kind": "upper_only", "upper": upper, "strong_upper": strong_upper, "median": med, "mad": mad}
    lower = min(p005, med - mad_scale * mad)
    upper = max(p995, med + mad_scale * mad)
    strong_lower = min(p001, med - strong_mad_scale * mad)
    strong_upper = max(p999, med + strong_mad_scale * mad)
    return {
        "kind": "range",
        "lower": lower,
        "upper": upper,
        "strong_lower": strong_lower,
        "strong_upper": strong_upper,
        "median": med,
        "mad": mad,
    }


def detect_value(value: float, thr: Dict[str, Any]) -> Dict[str, Any]:
    value = float(value)
    if thr["kind"] == "lower_only":
        alarm = int(value < float(thr["lower"]))
        strong = int(value < float(thr["strong_lower"]))
        margin = max(0.0, float(thr["lower"]) - value) / (abs(float(thr["mad"])) + 1e-6)
        return {"alarm": alarm, "strong_alarm": strong, "margin": margin}
    if thr["kind"] == "upper_only":
        alarm = int(value > float(thr["upper"]))
        strong = int(value > float(thr["strong_upper"]))
        margin = max(0.0, value - float(thr["upper"])) / (abs(float(thr["mad"])) + 1e-6)
        return {"alarm": alarm, "strong_alarm": strong, "margin": margin}
    lower = float(thr["lower"]); upper = float(thr["upper"])
    s_lower = float(thr["strong_lower"]); s_upper = float(thr["strong_upper"])
    alarm = int((value < lower) or (value > upper))
    strong = int((value < s_lower) or (value > s_upper))
    if value < lower:
        margin = (lower - value) / (abs(float(thr["mad"])) + 1e-6)
    elif value > upper:
        margin = (value - upper) / (abs(float(thr["mad"])) + 1e-6)
    else:
        margin = 0.0
    return {"alarm": alarm, "strong_alarm": strong, "margin": margin}


def detect_translation_shape(feats: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    st = thresholds.get("dim_stats", {}).get("translation_shape", {})
    xlr = float(feats.get("xy_log_ratio", 0.0))
    lower = float(st.get("xy_log_ratio_lower", -math.inf))
    upper = float(st.get("xy_log_ratio_upper", math.inf))
    strong_lower = float(st.get("xy_log_ratio_strong_lower", lower))
    strong_upper = float(st.get("xy_log_ratio_strong_upper", upper))
    alarm = int(xlr < lower or xlr > upper)
    strong = int(xlr < strong_lower or xlr > strong_upper)
    margin = max(0.0, lower - xlr, xlr - upper)
    return {
        "pred_label": int(alarm),
        "strong_count": int(strong),
        "soft_score": float(margin),
        "details": {
            "xy_log_ratio": {
                "value": xlr,
                "lower": lower,
                "upper": upper,
                "strong_lower": strong_lower,
                "strong_upper": strong_upper,
                "alarm": alarm,
                "strong_alarm": strong,
                "margin": float(margin),
            },
            "x": float(feats.get("x", 0.0)),
            "y": float(feats.get("y", 0.0)),
        },
    }


def detect_object_can_bus_value(group_feats: Dict[str, Dict[str, float]], direct_feats: Dict[str, float], small_struct_feats: Dict[str, Any], translation_shape_feats: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    group_reports = {}
    small_struct_report = detect_small_signed_structure(small_struct_feats, thresholds)
    translation_shape_report = detect_translation_shape(translation_shape_feats, thresholds)
    max_group_soft = 0.0
    any_group_pred = 0
    high_priority_hits = 0
    for gname in CANBUS_GROUP_ORDER:
        details = {}
        hard = 0
        strong = 0
        soft = 0.0
        feats = group_feats[gname]
        for feat in canbus_group_feature_list(gname):
            thr = thresholds["feature_thresholds"]["can_bus_groups"][gname].get(feat)
            if thr is None:
                continue
            outcome = detect_value(float(feats.get(feat, 0.0)), thr)
            details[feat] = {"value": float(feats.get(feat, 0.0)), **outcome}
            hard += int(outcome["alarm"])
            strong += int(outcome["strong_alarm"])
            if feat in {"finite_ratio", "max_dim_robust_z", "quat_norm_error", "angle_consistency_error"}:
                w = 1.0
            else:
                w = 0.6
            soft += w * float(outcome["margin"])
        if gname in CANBUS_WEAK_GROUPS:
            pred = int(strong >= 1 or soft > thresholds["soft_thresholds"]["can_bus"]["p999"])
        elif gname in {"lp_9", "lp_13"}:
            pred = int(strong >= 1 or hard >= 1 or soft > thresholds["soft_thresholds"]["can_bus"]["p995"])
        else:
            pred = int(strong >= 1 or hard >= 1 or soft > thresholds["soft_thresholds"]["can_bus"]["p995"])
        if gname == "small_signed_state":
            pred = int(pred == 1 or small_struct_report.get("pred_label", 0) == 1)
            soft = max(soft, float(small_struct_report.get("soft_score", 0.0)))
        if gname == "translation_xyz":
            pred = int(pred == 1 or translation_shape_report.get("pred_label", 0) == 1)
            soft = max(soft, float(translation_shape_report.get("soft_score", 0.0)))
        if gname in CANBUS_HIGH_PRIORITY_GROUPS and pred:
            high_priority_hits += 1
        group_report = {
            "pred_label": pred,
            "alarm_count": hard,
            "strong_count": strong,
            "soft_score": soft,
            "details": details,
        }
        if gname == "small_signed_state":
            group_report["structure_detector"] = small_struct_report
        if gname == "translation_xyz":
            group_report["translation_shape_detector"] = translation_shape_report
        group_reports[gname] = group_report
        any_group_pred = max(any_group_pred, pred)
        max_group_soft = max(max_group_soft, soft)

    direct_details = {}
    direct_hard = 0
    direct_strong = 0
    direct_soft = 0.0
    for feat in CANBUS_DIRECT_FEATURES:
        thr = thresholds["feature_thresholds"]["can_bus_direct"].get(feat)
        if thr is None:
            continue
        outcome = detect_value(float(direct_feats.get(feat, 0.0)), thr)
        direct_details[feat] = {"value": float(direct_feats.get(feat, 0.0)), **outcome}
        direct_hard += int(outcome["alarm"])
        direct_strong += int(outcome["strong_alarm"])
        direct_soft += 1.0 * float(outcome["margin"])
    direct_pred = int(direct_strong >= 1 or direct_hard >= 1 or direct_soft > thresholds["soft_thresholds"]["can_bus"]["p995"])

    obj_pred = int(
        direct_pred == 1
        or high_priority_hits >= 1
        or any_group_pred == 1
        or max(max_group_soft, direct_soft) > thresholds["soft_thresholds"]["can_bus"]["p995"]
    )
    return {
        "pred_label": obj_pred,
        "alarm_group_count": int(sum(int(group_reports[g]["pred_label"]) for g in group_reports)),
        "high_priority_hits": int(high_priority_hits),
        "direct_use": {
            "pred_label": direct_pred,
            "alarm_count": direct_hard,
            "strong_count": direct_strong,
            "soft_score": float(direct_soft),
            "details": direct_details,
        },
        "soft_score": float(max(max_group_soft, direct_soft)),
        "groups": group_reports,
    }

def classification_metrics_impl(records: List[Dict], key_path: str = "combined") -> Dict[str, Any]:
    y_true = []
    y_pred = []
    for r in records:
        obj = r["detector_report"]
        pred = obj[key_path]["pred_label"]
        if pred not in (0, 1):
            continue
        y_true.append(int(r.get("is_positive_effective", r["is_positive"])))
        y_pred.append(int(pred))
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    def sd(a, b): return float(a) / float(b) if b else 0.0

    prec = sd(tp, tp + fp)
    rec = sd(tp, tp + fn)
    return {
        "num_samples": len(y_true),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": sd(tp + tn, len(y_true)),
        "precision": prec,
        "recall": rec,
        "f1": sd(2 * prec * rec, prec + rec),
        "false_alarm_rate": sd(fp, fp + tn),
    }


def load_model_with_deepspeed(args, device):
    disable_torch_init()
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": args.attn_implementation,
        "overwrite_config": {"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    }
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_base=None, model_name="llava_qwen", device_map=device, **llava_model_args
    )
    ds_config = {
        "fp16": {"enabled": args.fp16},
        "bf16": {"enabled": args.bf16},
        "zero_optimization": {"stage": 0},
        "train_micro_batch_size_per_gpu": args.batch_size,
        "wall_clock_breakdown": False,
        "inference_mode": True,
    }
    model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=[])
    return tokenizer, model_engine, image_processor, context_len


def load_subset_ids(path: str):
    with open(path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    if not isinstance(ids, list):
        raise ValueError(f"subset ids file must contain a JSON list, got {type(ids)}")
    return set(ids), ids


def _collect_entry_candidate_ids(obj, fallback_idx=None, max_depth=3):
    keys = {
        "id",
        "qa_id",
        "sample_id",
        "sample_token",
        "token",
        "scene_id",
        "scene_token",
    }
    out = []

    def visit(x, depth):
        if depth < 0:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keys and v is not None:
                    out.append(str(v))
            for v in x.values():
                if isinstance(v, (dict, list, tuple)):
                    visit(v, depth - 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                if isinstance(v, (dict, list, tuple)):
                    visit(v, depth - 1)

    visit(obj, max_depth)
    if fallback_idx is not None:
        out.append(str(fallback_idx))

    dedup = []
    seen = set()
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def get_entry_id(entry, fallback_idx=None):
    candidates = _collect_entry_candidate_ids(entry, fallback_idx=fallback_idx, max_depth=2)
    if candidates:
        return candidates[0]
    return fallback_idx


def apply_subset_inplace(dataset, subset_ids_path, rank=0):
    subset_ids_set, subset_ids_list = load_subset_ids(subset_ids_path)
    subset_ids_set = set(str(x) for x in subset_ids_set)

    original_list_data_dict = dataset.list_data_dict
    original_data_infos = dataset.data_infos
    keep_indices, matched_ids = [], set()

    if rank == 0:
        print(f"[subset] loading subset ids from: {subset_ids_path}")
        print(f"[subset] requested ids: {len(subset_ids_list)}")
        print(f"[subset] original dataset size: {len(dataset.list_data_dict)}")

    for idx, entry in enumerate(original_list_data_dict):
        info = original_data_infos[idx] if idx < len(original_data_infos) else None

        candidate_ids = []
        candidate_ids.extend(_collect_entry_candidate_ids(entry, fallback_idx=idx, max_depth=3))
        if info is not None:
            candidate_ids.extend(_collect_entry_candidate_ids(info, fallback_idx=None, max_depth=3))

        matched_id = None
        for cid in candidate_ids:
            if str(cid) in subset_ids_set:
                matched_id = str(cid)
                break

        if matched_id is not None:
            keep_indices.append(idx)
            matched_ids.add(matched_id)

    if not keep_indices:
        raise RuntimeError(f"No dataset items matched subset ids from {subset_ids_path}.")

    dataset.list_data_dict = [original_list_data_dict[i] for i in keep_indices]
    dataset.data_infos = [original_data_infos[i] for i in keep_indices]

    if rank == 0:
        print(f"[subset] kept indices: {len(keep_indices)}")
        print(f"[subset] matched unique ids: {len(matched_ids)} / {len(subset_ids_list)}")
        print(f"[subset] filtered dataset size: {len(dataset.list_data_dict)}")

    return dataset, subset_ids_list


def inference_data(data, model_engine, tokenizer, args):
    sample_id = data["id"]
    question = data["question"]
    input_ids = data["input_ids"]
    uniad_data = data.get("uniad_data", None)
    uniad_pth = data.get("uniad_pth", None)
    qa_instance_ind = data.get("qa_instance_ind", None)
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            cont = model_engine.generate(
                input_ids,
                uniad_data=uniad_data,
                uniad_pth=uniad_pth,
                qa_instance_ind=qa_instance_ind,
                do_sample=False,
                temperature=0,
                max_new_tokens=int(getattr(args, "max_new_tokens", 192)),
                num_beams=1,
            )
    answer = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return {"id": sample_id, "question": question, "answer": answer}



# -----------------------------------------------------------------------------
# v32 S1 activation-output fault hooks
# -----------------------------------------------------------------------------

def v32_first_tensor(obj: Any):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            t = v32_first_tensor(item)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for item in obj.values():
            t = v32_first_tensor(item)
            if t is not None:
                return t
    return None


def v32_set_first_tensor(obj: Any, new_tensor: torch.Tensor) -> Tuple[Any, bool]:
    if torch.is_tensor(obj):
        return new_tensor, True
    if isinstance(obj, tuple):
        items = list(obj)
        for i, item in enumerate(items):
            new_item, ok = v32_set_first_tensor(item, new_tensor)
            if ok:
                items[i] = new_item
                return tuple(items), True
        return obj, False
    if isinstance(obj, list):
        items = list(obj)
        for i, item in enumerate(items):
            new_item, ok = v32_set_first_tensor(item, new_tensor)
            if ok:
                items[i] = new_item
                return items, True
        return obj, False
    if isinstance(obj, dict):
        d = dict(obj)
        for k, item in obj.items():
            new_item, ok = v32_set_first_tensor(item, new_tensor)
            if ok:
                d[k] = new_item
                return d, True
        return obj, False
    return obj, False


def v32_tensor_basic_summary(t: Any) -> Dict[str, Any]:
    if t is None or not torch.is_tensor(t):
        return {"available": 0}
    with torch.no_grad():
        x = t.detach()
        xf = x.float()
        finite = torch.isfinite(xf)
        finite_ratio = float(finite.float().mean().item()) if xf.numel() else 1.0
        if xf.numel() == 0 or not bool(finite.any().item()):
            return {"available": 1, "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device), "numel": int(x.numel()), "finite_ratio": finite_ratio, "abs_max": None, "mean_abs": None}
        vals = xf[finite]
        return {"available": 1, "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device), "numel": int(x.numel()), "finite_ratio": finite_ratio, "abs_max": float(vals.abs().max().item()), "mean_abs": float(vals.abs().mean().item())}


def v32_mutate_activation_clone(t: torch.Tensor, spec: FaultEventSpec, rng: random.Random, max_attempts: int = 200) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Mutate one S1 activation scalar without materializing all tensor indices.

    v35 speedfix rationale:
    early S1 activations can contain tens to hundreds of millions of scalar
    elements. The original implementation created ``list(range(numel))`` and
    shuffled it before trying only a small number of candidates. That preserved
    the intended random flat-index sampling semantics but became a severe
    single-threaded CPU bottleneck.

    This version samples random flat indices directly, checks finiteness only
    for the sampled scalar, and tries at most ``max_attempts`` unique indices.
    The fault model remains random activation flat-index exponent-bit mutation
    with the same bit_count / bit_op constraints.
    """
    info: Dict[str, Any] = {"target": spec.target, "bit_count": int(spec.bit_count), "bit_op": spec.bit_op, "event_id": int(spec.event_id), "field": "activation_exponent", "applied": 0, "effective": 0}
    if t is None or not torch.is_tensor(t):
        info["reason"] = "no_tensor_output"; return t, info
    if not torch.is_floating_point(t):
        info["reason"] = f"not_floating_dtype:{t.dtype}"; return t, info
    if t.numel() <= 0:
        info["reason"] = "empty_tensor"; return t, info

    out = t.clone()
    flat = out.reshape(-1)
    orig_flat = t.reshape(-1)
    numel = int(flat.numel())

    tried_indices = set()
    tries = 0
    while tries < max_attempts and len(tried_indices) < numel:
        flat_idx = int(rng.randrange(numel))
        if flat_idx in tried_indices:
            continue
        tried_indices.add(flat_idx)
        tries += 1

        try:
            val = flat[flat_idx]
            if not bool(torch.isfinite(val.detach().float()).item()):
                continue
            uint, bits_total, mantissa_bits, exp_bits = scalar_uint_info_torch(flat, flat_idx)
            spans = valid_spans_for_uint(uint, bits_total, mantissa_bits, exp_bits, int(spec.bit_count), str(spec.bit_op))
        except Exception as e:
            info.update({"reason": f"span_error:{type(e).__name__}:{e}", "attempts": tries}); return t, info

        rng.shuffle(spans)
        for span in spans:
            before = safe_float(flat[flat_idx].detach().float().cpu().item())
            try:
                fi = mutate_one_scalar_tensor_exponent_bits_(flat, flat_idx, span, str(spec.bit_op))
            except Exception:
                continue
            after = safe_float(flat[flat_idx].detach().float().cpu().item())
            effective = effective_value_changed(before, after)
            if effective:
                info.update(fi)
                info.update({"applied": 1, "effective": 1, "reason": "ok", "attempts": tries, "dtype": str(t.dtype), "shape": list(t.shape), "replace_output_ok": 1})
                return out, info
            flat[flat_idx] = orig_flat[flat_idx]

    info.update({"reason": "no_effective_span_found", "attempts": tries, "dtype": str(t.dtype), "shape": list(t.shape)})
    return t, info


class V32S1FaultRuntime:
    def __init__(self, target_specs: List[Tuple[str, str]], base_seed: int = 0):
        self.target_specs = list(target_specs)
        self.base_seed = int(base_seed)
        self.name_to_module: Dict[str, str] = {}
        self.name_to_suffix: Dict[str, str] = {name: suffix for name, suffix in self.target_specs}
        self.current_sample_id = ""
        self.pending_by_name: Dict[str, List[FaultEventSpec]] = {}
        self.applied_faults: List[Dict[str, Any]] = []
        self.capture_summaries: Dict[str, Dict[str, Any]] = {}
        # v50.3 strictfix:
        # The S target registry covers the whole S branch, including backbone, BEV,
        # track and segmentation heads. Some of these modules are executed more
        # than once for a single sample. Strict-single-fault semantics require the
        # planned event to be consumed on its first matching runtime hit; otherwise
        # spec=1 rows can become applied/effective=2 or 3, exactly as the v50.1
        # smoke found for several S/S4 targets.
        self.consumed_event_ids: set = set()

    def reset(self):
        self.current_sample_id = ""
        self.pending_by_name = {}
        self.applied_faults = []
        self.capture_summaries = {}
        self.consumed_event_ids = set()

    def begin_sample(self, sample_id: str, specs: List[FaultEventSpec], base_seed: int):
        self.reset(); self.current_sample_id = normalize_sample_token(sample_id); self.base_seed = int(base_seed)
        for spec in specs:
            if str(spec.target).startswith("s1:"):
                logical_name = str(spec.target).split(":", 1)[1]
                self.pending_by_name.setdefault(logical_name, []).append(spec)

    def hook_for(self, logical_name: str, module_name: str):
        self.name_to_module[logical_name] = module_name
        def _hook(module, inputs, output):
            specs_all = self.pending_by_name.get(logical_name, [])
            # v50.3 strictfix: most registered S hooks are not selected for the
            # current sample. For unselected targets the hook must be a true
            # no-op; otherwise every module execution performs finite/max/mean
            # tensor reductions and forces GPU/CPU synchronization.
            if not specs_all:
                return output

            specs = [spec for spec in specs_all if int(spec.event_id) not in self.consumed_event_ids]
            if not specs:
                previous = self.capture_summaries.get(logical_name)
                if previous is not None:
                    previous["call_count_after_consumed"] = int(previous.get("call_count_after_consumed", 0) or 0) + 1
                    previous["already_consumed_fault_count"] = int(len(specs_all))
                return output

            before_t = v32_first_tensor(output)
            previous = self.capture_summaries.get(logical_name, {}) or {}
            summary: Dict[str, Any] = {
                "logical_name": logical_name, "module_name": module_name,
                "call_count": int(previous.get("call_count", 0) or 0) + 1,
                "input_summary": v32_tensor_basic_summary(v32_first_tensor(inputs)),
                "output_summary_before": v32_tensor_basic_summary(before_t),
                "requested_fault_count": len(specs_all),
                "already_consumed_fault_count": int(len(specs_all) - len(specs)),
                "applied_fault_count": 0, "effective_fault_count": 0, "faults": [],
            }
            if before_t is not None and specs:
                new_t = before_t
                for spec in specs:
                    # Consume the event before mutation, matching the D-side
                    # guard. This prevents repeated runtime hits from injecting
                    # the same planned event multiple times in one sample.
                    self.consumed_event_ids.add(int(spec.event_id))
                    rng = random.Random(sample_seed(self.base_seed, self.current_sample_id, int(spec.event_id)))
                    new_t, info = v32_mutate_activation_clone(new_t, spec, rng)
                    info.update({
                        "target": spec.target,
                        "s1_logical_name": logical_name,
                        "s1_module_name": module_name,
                        "s1_suffix": self.name_to_suffix.get(logical_name, ""),
                        "one_shot_consumed": 1,
                    })
                    summary["faults"].append(info); self.applied_faults.append(info)
                replaced, ok = v32_set_first_tensor(output, new_t)
                summary["replace_output_ok"] = int(ok)
                if ok: output = replaced
                else:
                    for info in summary["faults"]: info["replace_output_ok"] = 0
                summary["applied_fault_count"] = int(sum(int(f.get("applied", 0) or 0) for f in summary["faults"]))
                summary["effective_fault_count"] = int(sum(int(f.get("effective", 0) or 0) for f in summary["faults"]))
                summary["output_summary_after"] = v32_tensor_basic_summary(new_t)
            self.capture_summaries[logical_name] = summary
            return output
        return _hook

    def finish_sample(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        applied = list(self.applied_faults)
        summary = {"pending_target_count": int(len(self.pending_by_name)), "pending_fault_count": int(sum(len(v) for v in self.pending_by_name.values())), "applied_fault_count": int(sum(int(f.get("applied", 0) or 0) for f in applied)), "effective_fault_count": int(sum(int(f.get("effective", 0) or 0) for f in applied)), "captures": self.capture_summaries}
        return applied, summary

def register_v32_s1_fault_hooks(model_engine, target_specs_raw: str) -> Tuple[V32S1FaultRuntime, List[Any], Dict[str, Any]]:
    specs = parse_s1_target_specs(target_specs_raw)
    runtime = V32S1FaultRuntime(specs)
    handles: List[Any] = []
    status: Dict[str, Any] = {"registered": {}, "missing": {}, "target_specs": [{"logical_name": n, "suffix": s} for n, s in specs]}
    for logical_name, suffix in specs:
        found, err = find_module_by_suffix_optional(model_engine, suffix)
        if found is None:
            status["missing"][logical_name] = {"suffix": suffix, "error": err}; continue
        module_name, module = found
        handles.append(module.register_forward_hook(runtime.hook_for(logical_name, module_name)))
        status["registered"][logical_name] = {"suffix": suffix, "module_name": module_name, "module_class": module.__class__.__name__}
        print(f"[{VERSION}] registered v32 S1 hook: {logical_name} -> {module_name} :: {module.__class__.__name__}")
    status["registered_count"] = int(len(status["registered"])); status["missing_count"] = int(len(status["missing"]))
    return runtime, handles, status


# -----------------------------------------------------------------------------
# v44 generic C/D activation hooks and E output perturbation
# -----------------------------------------------------------------------------

class V44ActivationFaultRuntime:
    def __init__(self, target_specs: List[Tuple[str, str]], prefix: str, base_seed: int = 0):
        self.target_specs = list(target_specs)
        self.prefix = str(prefix).strip().lower()
        self.base_seed = int(base_seed)
        self.name_to_module: Dict[str, str] = {}
        self.name_to_suffix: Dict[str, str] = {name: suffix for name, suffix in self.target_specs}
        self.current_sample_id = ""
        self.pending_by_name: Dict[str, List[FaultEventSpec]] = {}
        self.applied_faults: List[Dict[str, Any]] = []
        self.capture_summaries: Dict[str, Dict[str, Any]] = {}
        # v44 v2 one-shot guard:
        # LLM layers are called many times during generation: once for prefill and
        # repeatedly for autoregressive token decoding.  The v1 generic hook kept
        # applying the same planned D fault at every runtime hit, producing
        # spec=1 but effective=68/74/512 rows.  A strict single-fault experiment
        # must consume each planned event after the first matching hook call, even
        # if that first mutation is ineffective.
        self.consumed_event_ids: set = set()

    def reset(self):
        self.current_sample_id = ""; self.pending_by_name = {}; self.applied_faults = []; self.capture_summaries = {}; self.consumed_event_ids = set()

    def begin_sample(self, sample_id: str, specs: List[FaultEventSpec], base_seed: int):
        self.reset(); self.current_sample_id = normalize_sample_token(sample_id); self.base_seed = int(base_seed)
        expected = self.prefix + ":"
        for spec in specs:
            if str(spec.target).startswith(expected):
                logical_name = str(spec.target).split(":", 1)[1]
                self.pending_by_name.setdefault(logical_name, []).append(spec)

    def hook_for(self, logical_name: str, module_name: str):
        self.name_to_module[logical_name] = module_name
        def _hook(module, inputs, output):
            specs_all = self.pending_by_name.get(logical_name, [])
            # v50.3 strictfix: C/D registration can include hundreds of modules, but
            # strict-single-fault selects only one target for a sample.  Unselected
            # hooks must return immediately.  The previous lite package still ran
            # v32_tensor_basic_summary() for every D hook on every generation step,
            # which dominated runtime and made CPU appear single-core bound.
            if not specs_all:
                return output

            specs = [spec for spec in specs_all if int(spec.event_id) not in self.consumed_event_ids]
            if not specs:
                previous = self.capture_summaries.get(logical_name)
                if previous is not None:
                    previous["call_count_after_consumed"] = int(previous.get("call_count_after_consumed", 0) or 0) + 1
                    previous["already_consumed_fault_count"] = int(len(specs_all))
                return output

            before_t = v32_first_tensor(output)
            previous = self.capture_summaries.get(logical_name, {}) or {}
            summary: Dict[str, Any] = {
                "prefix": self.prefix, "logical_name": logical_name, "module_name": module_name,
                "call_count": int(previous.get("call_count", 0) or 0) + 1,
                "input_summary": v32_tensor_basic_summary(v32_first_tensor(inputs)),
                "output_summary_before": v32_tensor_basic_summary(before_t),
                "requested_fault_count": len(specs_all),
                "applied_fault_count": 0, "effective_fault_count": 0, "faults": [],
            }
            summary["already_consumed_fault_count"] = int(len(specs_all) - len(specs))
            if before_t is not None and specs:
                new_t = before_t
                for spec in specs:
                    # Strict single-fault guard: consume the planned event on its
                    # first matching module execution. This prevents LLM decode
                    # loops from applying the same event to every generated token.
                    self.consumed_event_ids.add(int(spec.event_id))
                    rng = random.Random(sample_seed(self.base_seed, self.current_sample_id, int(spec.event_id)))
                    new_t, info = v32_mutate_activation_clone(new_t, spec, rng)
                    info.update({
                        "target": spec.target,
                        "module_prefix": self.prefix,
                        "logical_name": logical_name,
                        "module_name": module_name,
                        "suffix": self.name_to_suffix.get(logical_name, ""),
                        "one_shot_consumed": 1,
                    })
                    summary["faults"].append(info); self.applied_faults.append(info)
                replaced, ok = v32_set_first_tensor(output, new_t)
                summary["replace_output_ok"] = int(ok)
                if ok:
                    output = replaced
                else:
                    for info in summary["faults"]:
                        info["replace_output_ok"] = 0
                summary["applied_fault_count"] = int(sum(int(f.get("applied", 0) or 0) for f in summary["faults"]))
                summary["effective_fault_count"] = int(sum(int(f.get("effective", 0) or 0) for f in summary["faults"]))
                summary["output_summary_after"] = v32_tensor_basic_summary(new_t)
            self.capture_summaries[logical_name] = summary
            return output
        return _hook

    def finish_sample(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        applied = list(self.applied_faults)
        summary = {
            "prefix": self.prefix,
            "pending_target_count": int(len(self.pending_by_name)),
            "pending_fault_count": int(sum(len(v) for v in self.pending_by_name.values())),
            "applied_fault_count": int(sum(int(f.get("applied", 0) or 0) for f in applied)),
            "effective_fault_count": int(sum(int(f.get("effective", 0) or 0) for f in applied)),
            "captures": self.capture_summaries,
        }
        return applied, summary


def register_v44_activation_fault_hooks(model_engine, target_specs_raw: str, prefix: str) -> Tuple[V44ActivationFaultRuntime, List[Any], Dict[str, Any]]:
    specs = parse_s1_target_specs(target_specs_raw)
    runtime = V44ActivationFaultRuntime(specs, prefix=prefix)
    handles: List[Any] = []
    status: Dict[str, Any] = {"prefix": prefix, "registered": {}, "missing": {}, "target_specs": [{"logical_name": n, "suffix": s} for n, s in specs]}
    for logical_name, suffix in specs:
        found, err = find_module_by_suffix_optional(model_engine, suffix)
        if found is None:
            status["missing"][logical_name] = {"suffix": suffix, "error": err}
            continue
        module_name, module = found
        handles.append(module.register_forward_hook(runtime.hook_for(logical_name, module_name)))
        status["registered"][logical_name] = {"suffix": suffix, "module_name": module_name, "module_class": module.__class__.__name__}
        print(f"[{VERSION}] registered v44 {prefix.upper()} hook: {logical_name} -> {module_name} :: {module.__class__.__name__}")
    status["registered_count"] = int(len(status["registered"])); status["missing_count"] = int(len(status["missing"]))
    return runtime, handles, status


def v44_format_traj_as_answer(traj: List[Tuple[float, float]]) -> str:
    return "[" + ", ".join(f"({float(x):.4f}, {float(y):.4f})" for x, y in traj[:6]) + "]"


def apply_v44_e_output_fault(result: Dict[str, Any], sample_id: str, spec: FaultEventSpec, base_seed: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "target": spec.target, "bit_count": int(spec.bit_count), "bit_op": spec.bit_op,
        "event_id": int(spec.event_id), "field": "e_output_prediction",
        "applied": 0, "effective": 0,
    }
    logical = str(spec.target).split(":", 1)[1] if ":" in str(spec.target) else str(spec.target)
    answer_before = result.get("answer", "")
    traj_before, meta_before = robust_retrieve_traj_with_meta(answer_before)
    rng = random.Random(sample_seed(int(base_seed), normalize_sample_token(sample_id), int(spec.event_id)))
    try:
        if logical == "e_traj_zero":
            traj_after = [(0.0, 0.0)] * 6
            answer_after = v44_format_traj_as_answer(traj_after)
        elif logical == "e_traj_y_shift":
            # deterministic lateral output perturbation; magnitude follows bit_count
            mag = {1: 0.75, 2: 1.50, 3: 3.00}.get(int(spec.bit_count), 1.50)
            if str(spec.bit_op) == "stuck0":
                mag *= -1.0
            elif str(spec.bit_op) == "stuck1":
                mag *= 1.5
            traj_after = [(float(x), float(y) + mag) for x, y in traj_before]
            answer_after = v44_format_traj_as_answer(traj_after)
        elif logical == "e_traj_scale":
            scale = {1: 1.5, 2: -1.0, 3: 3.0}.get(int(spec.bit_count), 2.0)
            if str(spec.bit_op) == "stuck0":
                scale = 0.0
            elif str(spec.bit_op) == "stuck1":
                scale = abs(scale) + 1.0
            traj_after = [(float(x) * scale, float(y) * scale) for x, y in traj_before]
            answer_after = v44_format_traj_as_answer(traj_after)
        elif logical == "e_parse_break":
            token = rng.randint(1000, 9999)
            answer_after = f"FAULTED_TRAJECTORY_PARSE_BREAK_{token} no valid coordinate pairs"
            traj_after, _ = robust_retrieve_traj_with_meta(answer_after)
        else:
            info["reason"] = f"unknown_e_target:{logical}"
            return info
        # Keep answer type consistent with tokenizer.batch_decode output.
        result["answer"] = [answer_after]
        traj_check, meta_after = robust_retrieve_traj_with_meta(result.get("answer", ""))
        before_arr = np.asarray(traj_before, dtype=np.float64).reshape(-1, 2)
        after_arr = np.asarray(traj_check, dtype=np.float64).reshape(-1, 2)
        diff = float(np.max(np.abs(np.nan_to_num(after_arr - before_arr, nan=0.0)))) if before_arr.shape == after_arr.shape else math.inf
        info.update({
            "applied": 1,
            "effective": int(diff > 1e-9),
            "reason": "ok" if diff > 1e-9 else "no_output_change",
            "e_logical_name": logical,
            "max_abs_traj_delta": diff,
            "parse_before": meta_before,
            "parse_after": meta_after,
        })
    except Exception as e:
        info.update({"reason": f"e_output_fault_error:{type(e).__name__}:{e}"})
    return info


def v48_normalize_trajectory_object(obj: Any) -> Optional[List[Tuple[float, float]]]:
    """Convert a trajectory-like object into a 6x2 Python-float trajectory.

    This helper is deliberately narrow: E1 faults are applied to the trajectory
    object returned by the retrieval/parser path, before evaluator/pred-file
    construction.  It does not touch GT, metric code, or final saved evaluator
    outputs.
    """
    try:
        arr = np.asarray(obj, dtype=np.float64)
        if arr.size < 2:
            return None
        arr = arr.reshape(-1, 2)
        if arr.shape[0] < 1:
            return None
        arr = arr[:6]
        if arr.shape[0] < 6:
            last = arr[-1:].copy()
            arr = np.concatenate([arr] + [last] * (6 - arr.shape[0]), axis=0)
        out: List[Tuple[float, float]] = []
        for x, y in arr[:6]:
            out.append((float(x), float(y)))
        return out
    except Exception:
        return None


def v48_mutate_trajectory_float_expbits(
    traj: List[Tuple[float, float]],
    spec: FaultEventSpec,
    rng: random.Random,
    dtype: str = "float32",
    max_attempts: int = 128,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """Mutate one coordinate in a Python-float trajectory object.

    E1 is not a torch.nn.Module.  The runtime trace showed that E1 appears as a
    Python trajectory object returned by robust_retrieve_traj_with_meta / parsed
    before evaluator.  To keep the main fault model bit-level and comparable to
    activation exp-bit faults, this function converts coordinates to float32,
    mutates exponent bits of one scalar, and writes the mutated numeric value
    back into the trajectory object.
    """
    info: Dict[str, Any] = {
        "field": "e1_trajectory_float_exponent",
        "e1_numeric_dtype": str(dtype),
        "applied": 0,
        "effective": 0,
    }
    base = v48_normalize_trajectory_object(traj)
    if not base:
        info["reason"] = "no_trajectory_object"
        return traj, info
    np_dtype = np.float32 if str(dtype).lower() in {"float32", "fp32", "single"} else np.float64
    arr = np.asarray(base, dtype=np_dtype).reshape(-1, 2).copy()
    orig_arr = arr.copy()
    flat = arr.reshape(-1)
    n = int(flat.size)
    if n <= 0:
        info["reason"] = "empty_trajectory"
        return base, info
    # Prefer non-zero finite coordinates because stuck0/stuck1 may be no-op on
    # already-zero exponent fields.  Fallback to any finite coordinate.
    candidates = [i for i in range(n) if np.isfinite(float(flat[i])) and abs(float(flat[i])) > 1e-9]
    if not candidates:
        candidates = [i for i in range(n) if np.isfinite(float(flat[i]))]
    rng.shuffle(candidates)
    tries = 0
    for flat_idx in candidates[:max_attempts]:
        tries += 1
        try:
            uint, bits_total, mantissa_bits, exp_bits = scalar_uint_info_numpy(arr, int(flat_idx))
            spans = valid_spans_for_uint(uint, bits_total, mantissa_bits, exp_bits, int(spec.bit_count), str(spec.bit_op))
        except Exception as e:
            info.update({"reason": f"span_error:{type(e).__name__}:{e}", "attempts": tries})
            return base, info
        rng.shuffle(spans)
        for span in spans:
            before = float(flat[int(flat_idx)])
            try:
                fi = mutate_one_scalar_numpy_exponent_bits_(arr, int(flat_idx), span, str(spec.bit_op))
            except Exception:
                flat[int(flat_idx)] = orig_arr.reshape(-1)[int(flat_idx)]
                continue
            after = float(flat[int(flat_idx)])
            if np.isfinite(after) and effective_value_changed(before, after):
                out_arr = arr.astype(np.float64).reshape(-1, 2)
                out = [(float(x), float(y)) for x, y in out_arr[:6]]
                row = int(flat_idx) // 2
                col = int(flat_idx) % 2
                info.update(fi)
                info.update({
                    "applied": 1,
                    "effective": 1,
                    "reason": "ok",
                    "attempts": tries,
                    "trajectory_shape": list(arr.shape),
                    "coord_row": int(row),
                    "coord_col": int(col),
                    "coord_name": "x" if col == 0 else "y",
                    "max_abs_traj_delta": float(np.max(np.abs(out_arr - np.asarray(base, dtype=np.float64).reshape(-1, 2)))),
                })
                return out, info
            flat[int(flat_idx)] = orig_arr.reshape(-1)[int(flat_idx)]
    info.update({"reason": "no_effective_trajectory_span_found", "attempts": tries, "trajectory_shape": list(arr.shape)})
    return base, info


def apply_v48_e1_internal_fault(result: Dict[str, Any], sample_id: str, spec: FaultEventSpec, base_seed: int) -> Dict[str, Any]:
    """Apply E1 function-level trajectory-float bit mutation.

    Unlike v44 E2 output stress tests, this function does not corrupt decoded
    text and does not modify evaluator code.  It parses the current generated
    answer into the same trajectory object used by the retrieve/parser path,
    mutates one coordinate's exponent bits, stores the forced trajectory in the
    plan-conv row, and lets build_pred_files_from_plan_conv consume that object
    before evaluator/metric computation.
    """
    info: Dict[str, Any] = {
        "target": spec.target,
        "bit_count": int(spec.bit_count),
        "bit_op": spec.bit_op,
        "event_id": int(spec.event_id),
        "field": "e1_trajectory_float_exponent",
        "applied": 0,
        "effective": 0,
    }
    logical = str(spec.target).split(":", 1)[1] if ":" in str(spec.target) else str(spec.target)
    if logical not in {"e1_retrieve_traj_float", "e1_parsed_traj_float"}:
        # Backward-compatible fallback for old E2 stress-test targets.  v48 smoke
        # does not use these targets, but keeping the fallback avoids surprising
        # failures if a caller accidentally reuses the script.
        return apply_v44_e_output_fault(result, sample_id, spec, base_seed)
    answer_before = result.get("answer", "")
    traj_before, meta_before = robust_retrieve_traj_with_meta(answer_before)
    rng = random.Random(sample_seed(int(base_seed), normalize_sample_token(sample_id), int(spec.event_id)))
    try:
        traj_after, mut_info = v48_mutate_trajectory_float_expbits(traj_before, spec, rng, dtype="float32")
        info.update(mut_info)
        info.update({
            "e_logical_name": logical,
            "parse_before": meta_before,
            "e1_hook_position": "robust_retrieve_traj_with_meta_return_before_evaluator" if logical == "e1_retrieve_traj_float" else "build_pred_files_parsed_traj_before_evaluator",
        })
        if int(info.get("applied", 0) or 0) == 1:
            # Do not change decoded answer text.  The forced trajectory represents
            # the E1 retrieved/parsed trajectory object that is passed to pred-file
            # generation and evaluator.
            result["v48_e1_forced_traj"] = [[float(x), float(y)] for x, y in traj_after]
            result["v48_e1_fault_meta"] = {
                "sample_id": normalize_sample_token(sample_id),
                "target": str(spec.target),
                "event_id": int(spec.event_id),
                "bit_count": int(spec.bit_count),
                "bit_op": str(spec.bit_op),
                "logical": logical,
                "mutation": {k: v for k, v in info.items() if k not in {"parse_before"}},
            }
            # Confirm that the object differs from the unmutated retrieval output.
            before_arr = np.asarray(traj_before, dtype=np.float64).reshape(-1, 2)
            after_arr = np.asarray(traj_after, dtype=np.float64).reshape(-1, 2)
            diff = float(np.max(np.abs(np.nan_to_num(after_arr - before_arr, nan=0.0)))) if before_arr.shape == after_arr.shape else math.inf
            info["effective"] = int(diff > 1e-9)
            info["reason"] = "ok" if diff > 1e-9 else "no_e1_object_change"
            info["max_abs_traj_delta"] = diff
    except Exception as e:
        info.update({"reason": f"e1_fault_error:{type(e).__name__}:{e}", "applied": 0, "effective": 0})
    return info



# Runtime can_bus_mlp hook
# -----------------------------------------------------------------------------

class CanBusMlpCapture:
    def __init__(self):
        self.reset()

    def reset(self):
        self.call_count = 0
        self.last_input_vector = None
        self.last_output_vector = None
        self.last_input_shape = None
        self.last_output_shape = None
        self.last_output_dtype = None

    @staticmethod
    def _first_tensor(obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                t = CanBusMlpCapture._first_tensor(item)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for item in obj.values():
                t = CanBusMlpCapture._first_tensor(item)
                if t is not None:
                    return t
        return None

    @staticmethod
    def _to_numpy_vector(t: torch.Tensor) -> np.ndarray:
        if t is None:
            return np.zeros(0, dtype=np.float64)
        x = t.detach().float().cpu().reshape(-1).numpy().astype(np.float64)
        return x

    def hook(self, module, inputs, output):
        self.call_count += 1
        inp = self._first_tensor(inputs)
        out = self._first_tensor(output)
        if inp is not None:
            self.last_input_shape = list(inp.shape)
            self.last_input_vector = self._to_numpy_vector(inp)
        if out is not None:
            self.last_output_shape = list(out.shape)
            self.last_output_dtype = str(out.dtype)
            self.last_output_vector = self._to_numpy_vector(out)

    def output_vector(self, require: bool = True) -> np.ndarray:
        if self.last_output_vector is None:
            if require:
                raise RuntimeError(
                    "can_bus_mlp hook was registered, but no output was captured during generate(). "
                    "This usually means the hook target is wrong or the current inference route did not execute can_bus_mlp."
                )
            return np.zeros(0, dtype=np.float64)
        return self.last_output_vector.copy()

    def summary(self) -> Dict[str, Any]:
        return {
            "call_count": int(self.call_count),
            "last_input_shape": self.last_input_shape,
            "last_output_shape": self.last_output_shape,
            "last_output_dtype": self.last_output_dtype,
        }


def unwrap_model_for_modules(model_engine):
    m = getattr(model_engine, "module", model_engine)
    # Some wrappers nest the actual module one more level.
    if hasattr(m, "module") and hasattr(m.module, "named_modules"):
        return m.module
    return m


def find_module_by_suffix(model_engine, suffix: str):
    model = unwrap_model_for_modules(model_engine)
    exact = None
    suffix_matches = []
    for name, module in model.named_modules():
        if name == suffix:
            exact = (name, module)
            break
        if name.endswith(suffix):
            suffix_matches.append((name, module))
    if exact is not None:
        return exact
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        names = [x[0] for x in suffix_matches[:20]]
        raise RuntimeError(f"Ambiguous can_bus_mlp hook suffix={suffix}; matches={names}")
    available = [n for n, _ in model.named_modules() if "can_bus" in n or "canbus" in n.lower()]
    raise RuntimeError(
        f"Could not locate can_bus_mlp hook target by suffix={suffix}. "
        f"Available can_bus-like modules: {available[:50]}"
    )


def register_can_bus_mlp_hook(model_engine, suffix: str) -> Tuple[CanBusMlpCapture, Any, str]:
    name, module = find_module_by_suffix(model_engine, suffix)
    capture = CanBusMlpCapture()
    handle = module.register_forward_hook(capture.hook)
    print(f"[{VERSION}] registered can_bus_mlp hook: {name} :: {module.__class__.__name__}")
    return capture, handle, name


# -----------------------------------------------------------------------------
# v19 generic stage-summary hooks
# -----------------------------------------------------------------------------

class StageSummaryCapture:
    """Capture only lightweight scalar summaries from a module input/output.

    This is for analysis and later lightweight harm-predictor training. It never saves
    full tensors, so storage and overhead stay bounded even for BEV/LLM tensors.
    """

    def __init__(self, stage_name: str, max_calls_saved: int = 3):
        self.stage_name = str(stage_name)
        self.max_calls_saved = int(max_calls_saved)
        # v50.3 strictfix: D/model.norm hooks are executed repeatedly during
        # autoregressive generation.  Summarizing every call performs multiple
        # reductions and synchronizations.  Default to summarizing only the first
        # runtime hit; S/C/BEV hooks are normally single-hit, so their features are
        # preserved.  Set V50_STAGE_SUMMARY_MAX_SUMMARY_CALLS=-1 to restore all-call
        # v53 legacy-full mode keeps the historical stage-summary semantics by
        # default.  Use V50_STAGE_SUMMARY_MAX_SUMMARY_CALLS only for debugging/smoke.
        try:
            self.max_summary_calls = int(os.getenv("V50_STAGE_SUMMARY_MAX_SUMMARY_CALLS", "-1"))
        except Exception:
            self.max_summary_calls = -1
        self.reset()

    def reset(self):
        self.call_count = 0
        self.summarized_call_count = 0
        self.input_last: Dict[str, Any] = {}
        self.output_last: Dict[str, Any] = {}
        self.input_sum: Dict[str, float] = {}
        self.output_sum: Dict[str, float] = {}
        self.saved_calls: List[Dict[str, Any]] = []

    @staticmethod
    def _first_tensor(obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                t = StageSummaryCapture._first_tensor(item)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for item in obj.values():
                t = StageSummaryCapture._first_tensor(item)
                if t is not None:
                    return t
        return None

    @staticmethod
    def _safe_float(x: torch.Tensor) -> float:
        try:
            return float(x.detach().float().cpu().item())
        except Exception:
            return math.inf

    @classmethod
    def tensor_summary(cls, t: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Historical full scalar stage summary used by legacy v35.1 feature matrices.

        v53 restores the v35/v26-compatible summary content: finite ratio, mean,
        std, l2 norm, abs max, mean abs, sign/zero ratios, top1-over-mean, and
        p95/p99 absolute quantiles.  This is intentionally fuller than the v50.3
        fast diagnostic summary because the frozen v35.1 detectors require these
        exact feature families.
        """
        if t is None or not torch.is_tensor(t):
            return {"available": 0}
        with torch.no_grad():
            x = t.detach()
            shape = list(x.shape)
            dtype = str(x.dtype)
            xf = x.float().reshape(-1)
            n = int(xf.numel())
            if n == 0:
                return {"available": 1, "shape": shape, "dtype": dtype, "numel": 0}
            finite = torch.isfinite(xf)
            finite_count = int(finite.sum().detach().cpu().item())
            finite_ratio = float(finite_count) / float(n)
            if finite_count == 0:
                return {
                    "available": 1, "shape": shape, "dtype": dtype, "numel": n,
                    "finite_ratio": finite_ratio, "mean": math.inf, "std": math.inf,
                    "l2_norm": math.inf, "abs_max": math.inf, "mean_abs": math.inf,
                    "zero_ratio": 0.0, "pos_ratio": 0.0, "neg_ratio": 0.0,
                    "top1_over_mean": math.inf, "p95_abs": math.inf, "p99_abs": math.inf,
                }
            y = xf[finite]
            ay = torch.abs(y)
            mean = cls._safe_float(y.mean())
            std = cls._safe_float(y.std(unbiased=False)) if int(y.numel()) > 1 else 0.0
            mean_abs = cls._safe_float(ay.mean())
            abs_max = cls._safe_float(ay.max())
            l2_norm = cls._safe_float(torch.linalg.vector_norm(y))
            zero_ratio = float((y == 0).sum().detach().cpu().item()) / float(finite_count)
            pos_ratio = float((y > 0).sum().detach().cpu().item()) / float(finite_count)
            neg_ratio = float((y < 0).sum().detach().cpu().item()) / float(finite_count)
            if int(ay.numel()) >= 2:
                try:
                    p95_abs = cls._safe_float(torch.quantile(ay, 0.95))
                    p99_abs = cls._safe_float(torch.quantile(ay, 0.99))
                except Exception:
                    # Fallback for environments where quantile is slow/unsupported on a dtype/device.
                    vals = ay.detach().float().cpu().numpy()
                    p95_abs = float(np.quantile(vals, 0.95))
                    p99_abs = float(np.quantile(vals, 0.99))
            else:
                p95_abs = abs_max
                p99_abs = abs_max
            return {
                "available": 1,
                "shape": shape,
                "dtype": dtype,
                "numel": n,
                "finite_ratio": finite_ratio,
                "mean": mean,
                "std": std,
                "l2_norm": l2_norm,
                "abs_max": abs_max,
                "mean_abs": mean_abs,
                "zero_ratio": zero_ratio,
                "pos_ratio": pos_ratio,
                "neg_ratio": neg_ratio,
                "top1_over_mean": float(abs_max / (mean_abs + 1e-12)) if np.isfinite(mean_abs) else math.inf,
                "p95_abs": p95_abs,
                "p99_abs": p99_abs,
            }

    @staticmethod
    def _add_to_sum(dst: Dict[str, float], src: Dict[str, Any]):
        for k in STAGE_SUMMARY_NUMERIC_KEYS:
            v = src.get(k)
            if v is not None and np.isfinite(float(v)):
                dst[k] = float(dst.get(k, 0.0)) + float(v)

    @staticmethod
    def _avg_from_sum(sum_dict: Dict[str, float], denom: int) -> Dict[str, float]:
        if denom <= 0:
            return {}
        return {f"avg_{k}": float(v) / float(denom) for k, v in sum_dict.items()}

    def hook(self, module, inputs, output):
        self.call_count += 1
        if self.max_summary_calls >= 0 and self.summarized_call_count >= self.max_summary_calls:
            return
        self.summarized_call_count += 1
        inp = self._first_tensor(inputs)
        out = self._first_tensor(output)
        in_sum = self.tensor_summary(inp)
        out_sum = self.tensor_summary(out)
        self.input_last = in_sum
        self.output_last = out_sum
        self._add_to_sum(self.input_sum, in_sum)
        self._add_to_sum(self.output_sum, out_sum)
        if len(self.saved_calls) < self.max_calls_saved:
            self.saved_calls.append({
                "call_index": int(self.call_count),
                "input": in_sum,
                "output": out_sum,
            })

    def summary(self) -> Dict[str, Any]:
        denom = int(self.summarized_call_count)
        return {
            "call_count": int(self.call_count),
            "summarized_call_count": denom,
            "summary_policy": {"max_summary_calls": int(self.max_summary_calls)},
            "input_last": self.input_last,
            "output_last": self.output_last,
            "input_avg": self._avg_from_sum(self.input_sum, denom),
            "output_avg": self._avg_from_sum(self.output_sum, denom),
            "saved_calls": self.saved_calls,
        }


def parse_stage_hook_specs(spec: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for part in str(spec or "").split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f"Invalid stage hook spec '{part}'. Expected name=suffix.")
        name, suffix = part.split('=', 1)
        name = name.strip()
        suffix = suffix.strip()
        if not name or not suffix:
            raise ValueError(f"Invalid stage hook spec '{part}'. Expected non-empty name and suffix.")
        out.append((name, suffix))
    return out


def find_module_by_suffix_optional(model_engine, suffix: str):
    model = unwrap_model_for_modules(model_engine)
    exact = None
    suffix_matches = []
    for name, module in model.named_modules():
        if name == suffix:
            exact = (name, module)
            break
        if name.endswith(suffix):
            suffix_matches.append((name, module))
    if exact is not None:
        return exact, None
    if len(suffix_matches) == 1:
        return suffix_matches[0], None
    if len(suffix_matches) > 1:
        return None, f"ambiguous matches={ [x[0] for x in suffix_matches[:20]] }"
    return None, "not_found"


def register_stage_summary_hooks(model_engine, spec: str, strict: bool = False):
    captures: Dict[str, StageSummaryCapture] = {}
    handles: List[Any] = []
    status: Dict[str, Dict[str, Any]] = {}
    for stage_name, suffix in parse_stage_hook_specs(spec):
        found, err = find_module_by_suffix_optional(model_engine, suffix)
        if found is None:
            status[stage_name] = {"registered": 0, "suffix": suffix, "error": err}
            msg = f"[{VERSION}] stage hook missing: {stage_name} suffix={suffix} error={err}"
            if strict:
                raise RuntimeError(msg)
            print("[WARN]", msg)
            continue
        module_name, module = found
        cap = StageSummaryCapture(stage_name)
        handle = module.register_forward_hook(cap.hook)
        captures[stage_name] = cap
        handles.append(handle)
        status[stage_name] = {
            "registered": 1,
            "suffix": suffix,
            "module_name": module_name,
            "module_class": module.__class__.__name__,
        }
        print(f"[{VERSION}] registered stage hook: {stage_name} -> {module_name} :: {module.__class__.__name__}")
    return captures, handles, status


def reset_stage_captures(stage_captures: Dict[str, StageSummaryCapture]):
    for cap in stage_captures.values():
        cap.reset()


def collect_stage_summaries(stage_captures: Dict[str, StageSummaryCapture]) -> Dict[str, Any]:
    return {name: cap.summary() for name, cap in stage_captures.items()}


def _v19_get_stage_value(flat: Dict[str, float], stage: str, side: str, key: str) -> Optional[float]:
    v = flat.get(f"stage__{stage}__{side}__{key}")
    if v is None:
        return None
    try:
        vf = float(v)
        return vf if np.isfinite(vf) else None
    except Exception:
        return None


def compute_v19_propagation_features(stage_flat: Dict[str, float]) -> Dict[str, float]:
    """Derived cross-stage propagation features for offline harm learning/rule distillation.

    These features are online-computable because they only depend on current forward-pass
    stage summaries. They do not use clean-reference outputs, GT, or harmful labels.
    """
    out: Dict[str, float] = {}
    metrics = ["finite_ratio", "l2_norm", "abs_max", "mean_abs", "p99_abs", "zero_ratio", "pos_ratio", "neg_ratio"]
    pairs = [
        ("s_bev_encoder", "d_llm_layer0"),
        ("s_bev_encoder_layer0", "s_bev_encoder_layer5"),
        ("d_llm_layer0", "d_llm_layer6"),
        ("d_llm_layer6", "d_llm_layer12"),
        ("d_llm_layer12", "d_llm_layer18"),
        ("d_llm_layer18", "d_llm_layer23"),
        ("d_llm_layer0", "d_llm_layer23"),
    ]
    for a, b in pairs:
        for m in metrics:
            va = _v19_get_stage_value(stage_flat, a, "output_last", m)
            vb = _v19_get_stage_value(stage_flat, b, "output_last", m)
            if va is None or vb is None:
                continue
            base = f"prop__{a}__to__{b}__{m}"
            out[f"{base}__delta"] = float(vb - va)
            out[f"{base}__ratio"] = float(vb / (abs(va) + 1e-12))
    # Co-occurrence proxy features: intentionally simple and threshold-light.
    stages = ["s_bev_encoder", "d_llm_layer0", "d_llm_layer6", "d_llm_layer12", "d_llm_layer18", "d_llm_layer23"]
    for st in stages:
        fr = _v19_get_stage_value(stage_flat, st, "output_last", "finite_ratio")
        am = _v19_get_stage_value(stage_flat, st, "output_last", "abs_max")
        p99 = _v19_get_stage_value(stage_flat, st, "output_last", "p99_abs")
        if fr is not None:
            out[f"prop__{st}__finite_drop_proxy"] = float(max(0.0, 1.0 - fr))
        if am is not None and p99 is not None:
            out[f"prop__{st}__tail_spike_proxy"] = float(am / (abs(p99) + 1e-12))
    return out


def flatten_stage_summary_for_training(stage_summaries: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for stage, summary in (stage_summaries or {}).items():
        for side in ["input_last", "output_last", "input_avg", "output_avg"]:
            sub = summary.get(side, {}) or {}
            for k, v in sub.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    flat[f"stage__{stage}__{side}__{k}"] = float(v)
        flat[f"stage__{stage}__call_count"] = float(summary.get("call_count", 0))
    flat.update(compute_v19_propagation_features(flat))
    return flat


def compute_v19_e_confidence_features(output_report: Dict[str, Any], output_features: Dict[str, Any]) -> Dict[str, float]:
    rep = output_report or {}
    details = rep.get("details", {}) or {}
    alarm_count = int(rep.get("alarm_count", 0) or 0)
    strong_count = int(rep.get("strong_count", 0) or 0)
    pred_label = int(rep.get("pred_label", 0) or 0)
    soft_score = float(rep.get("soft_score", 0.0) or 0.0)
    traj_all_zero = int(float((output_features or {}).get("traj_all_zero", 0.0) or 0.0) >= 0.5)
    out = {
        "econf__pred_label": float(pred_label),
        "econf__alarm_count": float(alarm_count),
        "econf__strong_count": float(strong_count),
        "econf__soft_score": float(soft_score),
        "econf__traj_all_zero": float(traj_all_zero),
        "econf__high_conf": float(int(pred_label == 1 or strong_count >= 1 or alarm_count >= 2 or traj_all_zero == 1)),
        "econf__low_conf": float(int(alarm_count >= 1 and not (pred_label == 1 or strong_count >= 1 or alarm_count >= 2 or traj_all_zero == 1))),
        "econf__multi_alarm": float(int(alarm_count >= 2)),
    }
    for name, detail in details.items():
        if isinstance(detail, dict):
            safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
            out[f"econf__detail__{safe}__alarm"] = float(int(detail.get("alarm", 0) or 0))
            out[f"econf__detail__{safe}__strong"] = float(int(detail.get("strong_alarm", 0) or 0))
            try:
                out[f"econf__detail__{safe}__margin"] = float(detail.get("margin", 0.0) or 0.0)
            except Exception:
                pass
    return out


def compute_v19_fault_induced_proxy_features(rr: Dict[str, Any]) -> Dict[str, float]:
    """Low-cost proxy features indicating whether an E anomaly has upstream propagation support.

    These are features for learning/rule distillation, not labels. They do not compare
    to clean reference outputs and therefore do not leak GT/harm labels.
    """
    stage = rr.get("stage_feature_flat", {}) or {}
    out_rep = rr.get("output_sanity_report", {}) or {}
    out_feat = rr.get("output_sanity_features", {}) or {}
    econf = compute_v19_e_confidence_features(out_rep, out_feat)
    def gv(k: str) -> float:
        try:
            return float(stage.get(k, 0.0) or 0.0)
        except Exception:
            return 0.0
    d_keys = [
        "stage__d_llm_layer0__output_last__finite_ratio",
        "stage__d_llm_layer6__output_last__finite_ratio",
        "stage__d_llm_layer12__output_last__finite_ratio",
        "stage__d_llm_layer18__output_last__finite_ratio",
        "stage__d_llm_layer23__output_last__finite_ratio",
    ]
    d_drop = max([max(0.0, 1.0 - gv(k)) for k in d_keys] + [0.0])
    s_drop = max(0.0, 1.0 - gv("stage__s_bev_encoder__output_last__finite_ratio"))
    e_alarm = float(int(econf.get("econf__alarm_count", 0.0) >= 1))
    return {
        "finduced__e_alarm": e_alarm,
        "finduced__d_finite_drop_max": float(d_drop),
        "finduced__sbev_finite_drop": float(s_drop),
        "finduced__e_and_d_drop": float(e_alarm * int(d_drop > 1e-6)),
        "finduced__e_and_sbev_drop": float(e_alarm * int(s_drop > 1e-6)),
        "finduced__e_high_and_d_drop": float(econf.get("econf__high_conf", 0.0) * int(d_drop > 1e-6)),
    }



def compute_v19_e_contextual_features(rr: Dict[str, Any]) -> Dict[str, float]:
    """E semantic/context features for reducing harmless all-zero alarms.

    These features are online-computable from the current generated answer,
    predicted trajectory, scene-context availability summary, and current can_bus.
    They do not compare against clean reference outputs.
    """
    out: Dict[str, float] = {}
    e_parse = rr.get('e_parse_features', {}) or {}
    e_out = rr.get('output_sanity_features', {}) or {}
    e_rep = rr.get('output_sanity_report', {}) or {}
    scene = rr.get('scene_context_summary', {}) or {}
    vec = (rr.get('object_vectors', {}) or {}).get('can_bus') or []
    for k, v in e_parse.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            try:
                vf = float(v)
                if np.isfinite(vf):
                    out[f'econtext__parse__{k}'] = vf
            except Exception:
                pass
    traj_all_zero = float((e_out or {}).get('traj_all_zero', 0.0) or 0.0)
    parse_fallback = float(e_parse.get('parse_fallback_zero', 0.0) or 0.0)
    parse_success = float(e_parse.get('parse_success', 0.0) or 0.0)
    alarm_count = float((e_rep or {}).get('alarm_count', 0.0) or 0.0)
    out['econtext__traj_all_zero_and_parse_fallback'] = float(int(traj_all_zero >= 0.5 and parse_fallback >= 0.5))
    out['econtext__traj_all_zero_but_parse_success'] = float(int(traj_all_zero >= 0.5 and parse_success >= 0.5))
    out['econtext__alarm_count'] = alarm_count
    for k in ['has_command_like', 'has_route_like', 'has_lane_like', 'has_occupancy_like', 'has_box_like', 'has_gt_like_candidate']:
        try:
            out[f'econtext__scene__{k}'] = float(scene.get(k, 0) or 0)
        except Exception:
            pass
    ego = v19_ego_motion_context_features_from_canbus_vector(vec)
    out.update(ego)
    motion = out.get('egoctx__motion_proxy_l2', 0.0)
    out['econtext__traj_all_zero_with_low_motion_proxy'] = float(int(traj_all_zero >= 0.5 and motion < 1e-6))
    out['econtext__traj_all_zero_with_nonzero_motion_proxy'] = float(int(traj_all_zero >= 0.5 and motion >= 1e-6))
    return out


def add_v19_temporal_context_features(row_records: List[Dict[str, Any]]) -> None:
    """Add previous-frame temporal context features without future leakage.

    v19 intentionally uses only records that have already appeared earlier in the
    current run order. It never uses the next frame, future frames, clean-paired
    output, GT trajectory, or offline harmful labels. If exact scene metadata is
    unavailable, it still records a conservative previous-row proxy.
    """
    prev_global: Optional[Dict[str, Any]] = None
    prev_by_scene: Dict[str, Dict[str, Any]] = {}

    def sf(x: Any, default: float = math.nan) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    def nested(rr: Dict[str, Any], root: str, key: str) -> float:
        try:
            return sf((rr.get(root, {}) or {}).get(key))
        except Exception:
            return math.nan

    def add_delta(feat: Dict[str, float], name: str, cur: float, prev: float) -> None:
        feat[f'temporal__{name}__cur'] = cur
        feat[f'temporal__{name}__prev'] = prev
        if np.isfinite(cur) and np.isfinite(prev):
            feat[f'temporal__{name}__delta_prev'] = float(cur - prev)
            feat[f'temporal__{name}__changed'] = float(int(abs(cur - prev) > 1e-9))
        else:
            feat[f'temporal__{name}__delta_prev'] = math.nan
            feat[f'temporal__{name}__changed'] = math.nan

    output_keys = [
        'traj_all_zero', 'path_length', 'final_disp', 'jerk_max',
        'selected_traj_all_zero', 'selected_traj_nonzero_waypoint_count',
        'selected_traj_valid_waypoint_count', 'candidate_diversity_proxy',
        'candidate_all_zero_count_proxy',
    ]
    parse_keys = ['parse_success', 'parse_fallback_zero', 'valid_waypoint_count', 'parsed_path_length', 'parsed_final_disp']
    stage_keys = [
        'stage__d_llm_layer0__output_last__finite_ratio',
        'stage__d_llm_layer18__output_last__finite_ratio',
        'stage__d_llm_layer23__output_last__finite_ratio',
        'stage__s_bev_encoder__output_last__finite_ratio',
    ]

    for rr in row_records:
        scene_key = str(rr.get('scene_id') or rr.get('scene_token') or rr.get('scene_name') or '')
        prev = prev_by_scene.get(scene_key) if scene_key else prev_global
        feat: Dict[str, float] = {
            'temporal__prev_available': 0.0,
            'temporal__prev_same_scene_available': 0.0,
            'temporal__uses_future_frame': 0.0,
        }
        if prev is not None:
            feat['temporal__prev_available'] = 1.0
            prev_scene = str(prev.get('scene_id') or prev.get('scene_token') or prev.get('scene_name') or '')
            feat['temporal__prev_same_scene_available'] = float(int(bool(scene_key) and prev_scene == scene_key))
            for key in output_keys:
                add_delta(feat, f'e_output__{key}', nested(rr, 'output_sanity_features', key), nested(prev, 'output_sanity_features', key))
            for key in parse_keys:
                add_delta(feat, f'e_parse__{key}', nested(rr, 'e_parse_features', key), nested(prev, 'e_parse_features', key))
            for key in stage_keys:
                safe = key.replace('stage__', '').replace('__', '_')
                add_delta(feat, safe, nested(rr, 'stage_feature_flat', key), nested(prev, 'stage_feature_flat', key))
            cur_rep = rr.get('output_sanity_report', {}) or {}
            prev_rep = prev.get('output_sanity_report', {}) or {}
            add_delta(feat, 'e_report__alarm_count', sf(cur_rep.get('alarm_count')), sf(prev_rep.get('alarm_count')))
            add_delta(feat, 'e_report__strong_count', sf(cur_rep.get('strong_count')), sf(prev_rep.get('strong_count')))

        rr['temporal_context_features'] = feat
        prev_global = rr
        if scene_key:
            prev_by_scene[scene_key] = rr


def _summarize_light(obj: Any, depth: int = 0, max_depth: int = 4, max_items: int = 30) -> Any:
    if depth > max_depth:
        return {"type": type(obj).__name__, "truncated": True}
    if torch.is_tensor(obj):
        return {"type": "torch.Tensor", "shape": list(obj.shape), "dtype": str(obj.dtype), "device": str(obj.device)}
    if isinstance(obj, np.ndarray):
        return {"type": "np.ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        out = {"type": "dict", "keys": list(map(str, list(obj.keys())[:max_items])), "children": {}}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                break
            out["children"][str(k)] = _summarize_light(v, depth + 1, max_depth, max_items)
        return out
    if isinstance(obj, (list, tuple)):
        out = {"type": type(obj).__name__, "len": len(obj), "children": []}
        for v in list(obj)[:min(len(obj), max_items)]:
            out["children"].append(_summarize_light(v, depth + 1, max_depth, max_items))
        return out
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return {"type": type(obj).__name__, "value_preview": str(obj)[:80]}
    return {"type": type(obj).__name__}


def summarize_scene_context_from_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Scan current batch metadata for possible E-stage scene-risk inputs.

    This does not use these fields for final alarm. It records availability, paths,
    shapes, and GT/future-likeness flags so we can later decide what is safe.
    """
    keywords = ["box", "bbox", "agent", "map", "lane", "drivable", "occ", "occup", "command", "route", "traj", "planning"]
    gt_words = ["gt", "future", "label", "target", "ann", "annotation"]
    candidates: List[Dict[str, Any]] = []
    def walk(path: str, obj: Any, depth: int = 0):
        if depth > 5 or len(candidates) >= 200:
            return
        low = path.lower()
        hit = any(k in low for k in keywords)
        if hit:
            item: Dict[str, Any] = {"path": path, "type": type(obj).__name__, "gt_like": int(any(w in low for w in gt_words))}
            if torch.is_tensor(obj):
                item.update({"shape": list(obj.shape), "dtype": str(obj.dtype)})
            elif isinstance(obj, np.ndarray):
                item.update({"shape": list(obj.shape), "dtype": str(obj.dtype)})
            elif isinstance(obj, (list, tuple)):
                item.update({"len": len(obj)})
            elif isinstance(obj, dict):
                item.update({"keys": list(map(str, list(obj.keys())[:20]))})
            else:
                item.update({"preview": str(obj)[:80]})
            candidates.append(item)
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{path}.{k}" if path else str(k), v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(list(obj)[:20]):
                walk(f"{path}[{i}]", v, depth + 1)
    try:
        walk("data", data, 0)
    except Exception as e:
        return {"available": 0, "error": repr(e), "candidates": []}
    def has_kw(*kws):
        return int(any(any(k in c.get("path", "").lower() for k in kws) for c in candidates))
    return {
        "available": 1,
        "candidate_count": len(candidates),
        "has_box_like": has_kw("box", "bbox"),
        "has_agent_like": has_kw("agent"),
        "has_map_like": has_kw("map"),
        "has_lane_like": has_kw("lane"),
        "has_drivable_like": has_kw("drivable"),
        "has_occupancy_like": has_kw("occ", "occup"),
        "has_command_like": has_kw("command"),
        "has_route_like": has_kw("route"),
        "has_gt_like_candidate": int(any(c.get("gt_like") for c in candidates)),
        "candidates": candidates[:80],
    }


def flatten_scene_context_for_training(scene_ctx: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ["available", "candidate_count", "has_box_like", "has_agent_like", "has_map_like", "has_lane_like", "has_drivable_like", "has_occupancy_like", "has_command_like", "has_route_like", "has_gt_like_candidate"]:
        try:
            out[f"scene_ctx__{k}"] = float(scene_ctx.get(k, 0) or 0)
        except Exception:
            pass
    return out


def _is_number_like(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _flatten_nested_numeric(prefix: str, obj: Any, out: Dict[str, float], max_list: int = 8):
    """Flatten a nested object into numeric features for offline training only."""
    if obj is None:
        return
    if isinstance(obj, bool):
        out[prefix] = float(int(obj))
        return
    if isinstance(obj, (int, float, np.integer, np.floating)):
        try:
            v = float(obj)
            if np.isfinite(v):
                out[prefix] = v
        except Exception:
            pass
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = re.sub(r"[^A-Za-z0-9_]+", "_", str(k)).strip("_")
            if not key:
                continue
            _flatten_nested_numeric(f"{prefix}__{key}" if prefix else key, v, out, max_list=max_list)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(list(obj)[:max_list]):
            _flatten_nested_numeric(f"{prefix}__{i}", v, out, max_list=max_list)
        return





# -----------------------------------------------------------------------------
# v26 online head/result capture helpers
# -----------------------------------------------------------------------------

V26_ONLINE_HEAD_HOOK_SUFFIX = "vision_tower.vision_tower.vision_model"


def _v26_is_finite_float(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _v26_tensor_stats(prefix: str, t: Any, out: Dict[str, float]) -> None:
    if not torch.is_tensor(t):
        out[f"{prefix}__available"] = 0.0
        return
    try:
        with torch.no_grad():
            x = t.detach()
            out[f"{prefix}__available"] = 1.0
            out[f"{prefix}__numel"] = float(x.numel())
            shp = list(x.shape)
            out[f"{prefix}__ndim"] = float(len(shp))
            for i, s in enumerate(shp[:4]):
                out[f"{prefix}__shape{i}"] = float(s)
            if x.numel() == 0:
                out[f"{prefix}__finite_ratio"] = 0.0
                return
            xf = x.float().reshape(-1)
            finite = torch.isfinite(xf)
            fcnt = int(finite.sum().detach().cpu().item())
            out[f"{prefix}__finite_ratio"] = float(fcnt) / float(max(1, xf.numel()))
            if fcnt <= 0:
                return
            vals = xf[finite]
            av = torch.abs(vals)
            out[f"{prefix}__mean"] = float(vals.mean().detach().cpu().item())
            out[f"{prefix}__std"] = float(vals.std(unbiased=False).detach().cpu().item()) if vals.numel() > 1 else 0.0
            out[f"{prefix}__absmax"] = float(av.max().detach().cpu().item())
            out[f"{prefix}__mean_abs"] = float(av.mean().detach().cpu().item())
            out[f"{prefix}__l2"] = float(torch.linalg.vector_norm(vals).detach().cpu().item())
            # Query/token norm summaries for tensors shaped [..., C].
            if x.ndim >= 2 and x.shape[-1] > 1 and x.numel() <= 20000000:
                q = x.float().reshape(-1, x.shape[-1])
                qfinite = torch.isfinite(q).all(dim=1)
                if bool(qfinite.any().detach().cpu().item()):
                    qn = torch.linalg.vector_norm(q[qfinite], dim=1)
                    out[f"{prefix}__token_count"] = float(qn.numel())
                    out[f"{prefix}__token_norm_mean"] = float(qn.mean().detach().cpu().item())
                    out[f"{prefix}__token_norm_std"] = float(qn.std(unbiased=False).detach().cpu().item()) if qn.numel() > 1 else 0.0
                    out[f"{prefix}__token_norm_max"] = float(qn.max().detach().cpu().item())
    except Exception:
        out[f"{prefix}__error"] = 1.0


def _v26_find_dicts_with_key(obj: Any, key: str, max_depth: int = 5) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    def rec(x: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(x, dict):
            if key in x:
                found.append(x)
            for v in list(x.values())[:30]:
                rec(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in list(x)[:30]:
                rec(v, depth + 1)
    rec(obj, 0)
    return found


def _v26_get_tensor_attr(obj: Any, names: List[str]) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj
    for name in names:
        try:
            v = getattr(obj, name)
            if torch.is_tensor(v):
                return v
        except Exception:
            pass
    return None


def _v26_track_boxes_tensor(boxes_obj: Any) -> Optional[torch.Tensor]:
    return _v26_get_tensor_attr(boxes_obj, ["tensor", "gravity_center", "center", "bev", "corners"])


def _v26_summarize_scores(prefix: str, x: Any, out: Dict[str, float]) -> None:
    t = _v26_get_tensor_attr(x, ["tensor"])
    if t is None:
        out[f"{prefix}__available"] = 0.0
        return
    _v26_tensor_stats(prefix, t, out)
    try:
        y = t.detach().float().reshape(-1)
        finite = torch.isfinite(y)
        if finite.any():
            vals = y[finite]
            out[f"{prefix}__count"] = float(vals.numel())
            out[f"{prefix}__gt_0_3"] = float((vals > 0.3).sum().detach().cpu().item())
            out[f"{prefix}__gt_0_5"] = float((vals > 0.5).sum().detach().cpu().item())
            out[f"{prefix}__gt_0_7"] = float((vals > 0.7).sum().detach().cpu().item())
    except Exception:
        pass


def compute_v26_online_head_features_from_output(output: Any) -> Dict[str, float]:
    """Summarize online prediction outputs from the current vision tower forward.

    Only current-forward model outputs are used. GT-like keys such as planning_gt
    are intentionally ignored and must remain audit-only.
    """
    out: Dict[str, float] = {
        "v26__online_head_capture_available": 1.0,
        "v26__result_track_available": 0.0,
        "v26__result_seg_available": 0.0,
    }
    try:
        dicts = _v26_find_dicts_with_key(output, "result_track", max_depth=5)
        if dicts:
            d = dicts[0]
            rt = d.get("result_track") or {}
            out["v26__result_track_available"] = 1.0
            if isinstance(rt, dict):
                _v26_tensor_stats("v26__track_bev_embed", rt.get("bev_embed"), out)
                _v26_tensor_stats("v26__track_query_embeddings", rt.get("track_query_embeddings"), out)
                _v26_tensor_stats("v26__track_img_feat_2d", rt.get("img_feat_2D"), out)
                _v26_summarize_scores("v26__track_scores", rt.get("track_scores"), out)
                _v26_summarize_scores("v26__scores_3d", rt.get("scores_3d"), out)
                _v26_summarize_scores("v26__sdc_scores_3d", rt.get("sdc_scores_3d"), out)
                _v26_summarize_scores("v26__sdc_track_scores", rt.get("sdc_track_scores"), out)
                boxes = rt.get("boxes_3d")
                bt = _v26_track_boxes_tensor(boxes)
                _v26_tensor_stats("v26__boxes_3d_tensor", bt, out)
                if torch.is_tensor(bt):
                    try:
                        b = bt.detach().float()
                        if b.ndim >= 2 and b.shape[-1] >= 2:
                            centers = b.reshape(-1, b.shape[-1])[:, :2]
                            finite = torch.isfinite(centers).all(dim=1)
                            if finite.any():
                                dist = torch.linalg.vector_norm(centers[finite], dim=1)
                                out["v26__track_box_count"] = float(dist.numel())
                                out["v26__track_nearest_center_dist"] = float(dist.min().detach().cpu().item())
                                out["v26__track_center_dist_mean"] = float(dist.mean().detach().cpu().item())
                                front = centers[finite][(centers[finite][:, 1].abs() <= 10.0) & (centers[finite][:, 0] >= 0.0)]
                                out["v26__track_num_front_30m"] = float(((front[:, 0] <= 30.0).sum().detach().cpu().item())) if front.numel() else 0.0
                                out["v26__track_num_within_10m"] = float((dist <= 10.0).sum().detach().cpu().item())
                                out["v26__track_num_within_20m"] = float((dist <= 20.0).sum().detach().cpu().item())
                    except Exception:
                        out["v26__track_box_geom_error"] = 1.0
        seg_dicts = _v26_find_dicts_with_key(output, "result_seg", max_depth=5)
        if seg_dicts:
            d = seg_dicts[0]
            rs = d.get("result_seg") or {}
            out["v26__result_seg_available"] = 1.0
            if isinstance(rs, dict):
                _v26_tensor_stats("v26__seg_output_query_things", rs.get("output_query_things"), out)
                _v26_tensor_stats("v26__seg_output_query_stuff", rs.get("output_query_stuff"), out)
                _v26_tensor_stats("v26__seg_chosen_output_query_things", rs.get("chosen_output_query_things"), out)
                # query count features.
                for key in ["output_query_things", "chosen_output_query_things"]:
                    val = rs.get(key)
                    if torch.is_tensor(val) and val.ndim >= 1:
                        out[f"v26__seg_{key}_count"] = float(val.shape[0])
    except Exception:
        out["v26__online_head_capture_error"] = 1.0
    return out


class V26OnlineHeadCapture:
    """Capture lightweight online prediction/result summaries from vision_model.

    The capture is reset before every sample and after every record, which is
    important for mixed fault/no-fault experiments. It stores scalar summaries
    only, never full tensors.
    """
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.call_count = 0
        self.features: Dict[str, float] = {"v26__online_head_capture_available": 0.0}

    def hook(self, module, inputs, output):
        self.call_count += 1
        try:
            self.features = compute_v26_online_head_features_from_output(output)
            self.features["v26__online_head_call_count"] = float(self.call_count)
        except Exception:
            self.features = {"v26__online_head_capture_available": 0.0, "v26__online_head_capture_error": 1.0}

    def summary(self) -> Dict[str, float]:
        out = dict(self.features or {})
        out["v26__online_head_call_count"] = float(self.call_count)
        return out


def register_v26_online_head_capture(model_engine):
    found, err = find_module_by_suffix_optional(model_engine, V26_ONLINE_HEAD_HOOK_SUFFIX)
    cap = V26OnlineHeadCapture()
    if found is None:
        print(f"[WARN] [{VERSION}] v26 online head hook missing suffix={V26_ONLINE_HEAD_HOOK_SUFFIX} error={err}")
        return cap, None, {"registered": 0, "suffix": V26_ONLINE_HEAD_HOOK_SUFFIX, "error": err}
    module_name, module = found
    handle = module.register_forward_hook(cap.hook)
    status = {"registered": 1, "suffix": V26_ONLINE_HEAD_HOOK_SUFFIX, "module_name": module_name, "module_class": module.__class__.__name__}
    print(f"[{VERSION}] registered v26 online head hook: {module_name} :: {module.__class__.__name__}")
    return cap, handle, status


def compute_v26_branch_projector_features(rr: Dict[str, Any]) -> Dict[str, float]:
    """Derived features from v26 branch/projector stage summaries."""
    stage = rr.get("stage_feature_flat", {}) or {}
    out: Dict[str, float] = {}
    def gv(name: str, key: str, default: float = math.nan) -> float:
        v = stage.get(f"stage__{name}__output_last__{key}")
        try:
            vf = float(v)
            return vf if np.isfinite(vf) else default
        except Exception:
            return default
    groups = {
        "cls0": "v26_track_cls_branch0",
        "cls5": "v26_track_cls_branch5",
        "reg0": "v26_track_reg_branch0",
        "reg5": "v26_track_reg_branch5",
        "past0": "v26_track_past_traj_branch0",
        "past5": "v26_track_past_traj_branch5",
        "seg": "v26_seg_head",
        "proj_track": "v26_mm_projector_track",
        "proj_map": "v26_mm_projector_map",
        "decoder": "v26_track_decoder",
    }
    for short, st in groups.items():
        for k in ["finite_ratio", "l2_norm", "abs_max", "mean_abs", "std", "p99_abs", "numel"]:
            val = gv(st, k)
            if np.isfinite(val):
                out[f"v26__branch_{short}_{k}"] = float(val)
        cc = stage.get(f"stage__{st}__call_count")
        try:
            out[f"v26__branch_{short}_call_count"] = float(cc)
        except Exception:
            pass
    # Layer deltas: last-vs-first branch outputs.
    for base, a, b in [("cls", "v26_track_cls_branch0", "v26_track_cls_branch5"), ("reg", "v26_track_reg_branch0", "v26_track_reg_branch5"), ("past", "v26_track_past_traj_branch0", "v26_track_past_traj_branch5")]:
        for k in ["l2_norm", "abs_max", "mean_abs", "p99_abs", "std"]:
            va, vb = gv(a, k), gv(b, k)
            if np.isfinite(va) and np.isfinite(vb):
                out[f"v26__branch_{base}_{k}_delta_5_0"] = float(vb - va)
                out[f"v26__branch_{base}_{k}_ratio_5_0"] = float(vb / (abs(va) + 1e-12))
    return out

# -----------------------------------------------------------------------------
# v26 online-risk feature helpers
# -----------------------------------------------------------------------------

def compute_v26_canbus_group_features_from_vector(vec: Any) -> Dict[str, float]:
    """Online-safe can_bus group summaries.

    These are after-fault/current-input anomaly descriptors, not independent
    ego-motion truth. They are intended for non-collapse rescue and ablation.
    """
    out: Dict[str, float] = {}
    try:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.float64)
    n = int(arr.size)
    finite = np.isfinite(arr)
    out["v26__canbus_dim"] = float(n)
    out["v26__canbus_finite_ratio"] = float(np.mean(finite)) if n else 0.0
    if n and finite.any():
        vals = arr[finite]
        out["v26__canbus_abs_max"] = float(np.max(np.abs(vals)))
        out["v26__canbus_l2"] = float(np.linalg.norm(vals))
        out["v26__canbus_abs_sum"] = float(np.sum(np.abs(vals)))
    else:
        out["v26__canbus_abs_max"] = 0.0
        out["v26__canbus_l2"] = 0.0
        out["v26__canbus_abs_sum"] = 0.0

    groups = {
        "translation_xyz": [0, 1, 2],
        "rotation_quat": [3, 4, 5, 6],
        "small_signed_state": [7, 8, 10, 11, 12, 14, 15],
        "lp_9": [9],
        "lp_13": [13],
        "patch_angle_pair": [16, 17],
        "near_zero_slots": [7, 8, 9, 10, 11, 12, 13, 14, 15],
    }
    dom_name = "none"
    dom_score = -1.0
    alarms = 0
    for name, idxs in groups.items():
        vals = [float(arr[i]) for i in idxs if 0 <= i < n and math.isfinite(float(arr[i]))]
        if vals:
            a = np.asarray(vals, dtype=np.float64)
            l2 = float(np.linalg.norm(a))
            absmax = float(np.max(np.abs(a)))
            abssum = float(np.sum(np.abs(a)))
            finite_ratio = float(len(vals) / max(1, len([i for i in idxs if 0 <= i < n])))
        else:
            l2 = absmax = abssum = 0.0
            finite_ratio = 0.0
        out[f"v26__canbus_group_{name}_l2"] = l2
        out[f"v26__canbus_group_{name}_absmax"] = absmax
        out[f"v26__canbus_group_{name}_abssum"] = abssum
        out[f"v26__canbus_group_{name}_finite_ratio"] = finite_ratio
        if absmax > dom_score:
            dom_name, dom_score = name, absmax
        alarms += int(absmax > 1e-6)
    out["v26__canbus_group_alarm_count_proxy"] = float(alarms)
    out["v26__canbus_group_dominant_score"] = float(max(0.0, dom_score))
    # Numeric dominant id for model use; mapping documented in README.
    names = list(groups.keys())
    out["v26__canbus_group_dominant_id"] = float(names.index(dom_name)) if dom_name in names else -1.0
    return out


def compute_v26_risk_availability_features(scene_ctx: Dict[str, Any]) -> Dict[str, float]:
    """Availability summary for online-risk feature sources discovered in data.

    gt/future/label/target/ann/annotation/sdc_planning candidates are counted
    separately and are NOT used as detector evidence. This function helps v26
    decide whether online predicted track/motion/occupancy/lane/planning signals
    are available in the current runtime data structure.
    """
    candidates = list((scene_ctx or {}).get("candidates", []) or [])
    out: Dict[str, float] = {
        "v26__risk_candidate_count": float(len(candidates)),
        "v26__risk_online_candidate_count": 0.0,
        "v26__risk_gt_like_candidate_count": 0.0,
    }
    groups = {
        "pred_track_box": ["track", "box", "bbox", "agent"],
        "pred_motion": ["motion", "traj", "trajectory"],
        "pred_occupancy": ["occ", "occup"],
        "pred_lane_map": ["lane", "map", "drivable"],
        "planning_candidate": ["planning", "candidate", "score", "confidence"],
    }
    for g in groups:
        out[f"v26__risk_{g}_online_available"] = 0.0
        out[f"v26__risk_{g}_online_count"] = 0.0
        out[f"v26__risk_{g}_gt_like_count"] = 0.0
    for c in candidates:
        path = str(c.get("path", "")).lower()
        gt_like = int(c.get("gt_like", 0) or 0)
        if gt_like:
            out["v26__risk_gt_like_candidate_count"] += 1.0
        else:
            out["v26__risk_online_candidate_count"] += 1.0
        for g, kws in groups.items():
            if any(k in path for k in kws):
                if gt_like:
                    out[f"v26__risk_{g}_gt_like_count"] += 1.0
                else:
                    out[f"v26__risk_{g}_online_count"] += 1.0
                    out[f"v26__risk_{g}_online_available"] = 1.0
    return out


def compute_v26_planning_candidate_proxy_features(rr: Dict[str, Any]) -> Dict[str, float]:
    """Online-safe planning candidate proxies available from output sanity.

    These do not use sdc_planning or paired clean labels. They summarize whether
    the generated/selected planning candidates look diverse or collapsed when
    candidate-level proxies are available in the current output sanity dict.
    """
    eout = rr.get("output_sanity_features", {}) or {}
    def sf(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default
    return {
        "v26__planning_candidate_count_proxy": sf(eout.get("candidate_count_proxy", 0.0)),
        "v26__planning_candidate_all_zero_count_proxy": sf(eout.get("candidate_all_zero_count_proxy", 0.0)),
        "v26__planning_candidate_nonzero_count_proxy": sf(eout.get("candidate_nonzero_count_proxy", 0.0)),
        "v26__planning_candidate_diversity_proxy": sf(eout.get("candidate_diversity_proxy", 0.0)),
        "v26__planning_selected_all_zero": sf(eout.get("selected_traj_all_zero", 0.0)),
        "v26__planning_selected_nonzero_waypoint_count": sf(eout.get("selected_traj_nonzero_waypoint_count", 0.0)),
    }


# -----------------------------------------------------------------------------
# v35.1 exact behavior feature helpers
# -----------------------------------------------------------------------------
# These helpers intentionally reproduce tools_scan/append_v35_output_behavior_features.py
# from the v35/v38 zero-shot pipeline.  They are exact feature construction, not
# fuzzy feature mapping: v35.1 exported detectors require feat__behout_* and
# feat__behctx_* columns with these names.
V35_BEHAVIOR_PAIR_RE = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
V35_BEHAVIOR_FLOAT_RE = r"(-?\d+(?:\.\d+)?)"


def v35_behavior_sample_clean(x: Any) -> str:
    s = str(x or "")
    if s.endswith("_trajectory"):
        s = s[:-len("_trajectory")]
    return s


def v35_behavior_parse_pairs(text: Any) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in V35_BEHAVIOR_PAIR_RE.findall(str(text or "")):
        try:
            out.append((float(a), float(b)))
        except Exception:
            pass
    return out


def v35_behavior_parse_answer_traj(ans: Any) -> List[Tuple[float, float]]:
    return v35_behavior_parse_pairs(str(ans or ""))


def v35_behavior_parse_named_pair(text: str, label: str) -> Optional[Tuple[float, float]]:
    pat = re.compile(label + r"\s*:\s*\(" + V35_BEHAVIOR_FLOAT_RE + r"\s*,\s*" + V35_BEHAVIOR_FLOAT_RE + r"\)", re.I)
    m = pat.search(str(text or ""))
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))


def v35_behavior_parse_named_scalar(text: str, label: str) -> Optional[float]:
    pat = re.compile(label + r"\s*:\s*\(" + V35_BEHAVIOR_FLOAT_RE + r"\)", re.I)
    m = pat.search(str(text or ""))
    return float(m.group(1)) if m else None


def v35_behavior_traj_features(points: List[Tuple[float, float]], prefix: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    n = len(points)
    d[f"{prefix}parse_ok"] = 1.0 if n >= 2 else 0.0
    d[f"{prefix}point_count"] = float(n)
    if n == 0:
        for k in ["final_x","final_y","final_disp","path_length","directness","step_mean","step_std","step_min","step_max","step_cv","near_zero_flag","zero_step_ratio","heading_change","heading_abs_change_sum","heading_abs_change_max","curvature_mean","curvature_max","x_range","y_range","x_monotonicity","y_monotonicity"]:
            d[f"{prefix}{k}"] = math.nan
        return d
    arr = np.asarray(points, dtype=float)
    d[f"{prefix}final_x"] = float(arr[-1,0])
    d[f"{prefix}final_y"] = float(arr[-1,1])
    final_disp = float(np.linalg.norm(arr[-1] - arr[0])) if n >= 2 else float(np.linalg.norm(arr[-1]))
    d[f"{prefix}final_disp"] = final_disp
    d[f"{prefix}x_range"] = float(np.nanmax(arr[:,0]) - np.nanmin(arr[:,0]))
    d[f"{prefix}y_range"] = float(np.nanmax(arr[:,1]) - np.nanmin(arr[:,1]))
    if n >= 2:
        diff = np.diff(arr, axis=0)
        steps = np.linalg.norm(diff, axis=1)
        path_len = float(np.nansum(steps))
        d[f"{prefix}path_length"] = path_len
        d[f"{prefix}directness"] = float(final_disp / path_len) if path_len > 1e-9 else 0.0
        d[f"{prefix}step_mean"] = float(np.nanmean(steps))
        d[f"{prefix}step_std"] = float(np.nanstd(steps))
        d[f"{prefix}step_min"] = float(np.nanmin(steps))
        d[f"{prefix}step_max"] = float(np.nanmax(steps))
        d[f"{prefix}step_cv"] = float(np.nanstd(steps) / (np.nanmean(steps) + 1e-9))
        d[f"{prefix}near_zero_flag"] = 1.0 if path_len < 0.5 else 0.0
        d[f"{prefix}zero_step_ratio"] = float(np.mean(steps < 1e-6))
        headings = np.arctan2(diff[:,1], diff[:,0])
        if len(headings) >= 2:
            dh = np.diff(np.unwrap(headings))
            d[f"{prefix}heading_change"] = float(np.unwrap(headings)[-1] - np.unwrap(headings)[0])
            d[f"{prefix}heading_abs_change_sum"] = float(np.nansum(np.abs(dh)))
            d[f"{prefix}heading_abs_change_max"] = float(np.nanmax(np.abs(dh)))
            curv = np.abs(dh) / (steps[1:] + 1e-9)
            d[f"{prefix}curvature_mean"] = float(np.nanmean(curv)) if len(curv) else 0.0
            d[f"{prefix}curvature_max"] = float(np.nanmax(curv)) if len(curv) else 0.0
        else:
            d[f"{prefix}heading_change"] = 0.0
            d[f"{prefix}heading_abs_change_sum"] = 0.0
            d[f"{prefix}heading_abs_change_max"] = 0.0
            d[f"{prefix}curvature_mean"] = 0.0
            d[f"{prefix}curvature_max"] = 0.0
        dx = np.diff(arr[:,0]); dy = np.diff(arr[:,1])
        d[f"{prefix}x_monotonicity"] = float(abs(np.nansum(np.sign(dx))) / max(1, len(dx)))
        d[f"{prefix}y_monotonicity"] = float(abs(np.nansum(np.sign(dy))) / max(1, len(dy)))
    else:
        d[f"{prefix}path_length"] = 0.0
        d[f"{prefix}directness"] = 0.0
        for k in ["step_mean","step_std","step_min","step_max","step_cv","heading_change","heading_abs_change_sum","heading_abs_change_max","curvature_mean","curvature_max","x_monotonicity","y_monotonicity"]:
            d[f"{prefix}{k}"] = 0.0
        d[f"{prefix}near_zero_flag"] = 1.0
        d[f"{prefix}zero_step_ratio"] = 1.0
    return d


def v35_behavior_question_context_features(question: str) -> Dict[str, float]:
    q = str(question or "")
    d: Dict[str, float] = {}
    vel = v35_behavior_parse_named_pair(q, r"Velocity \(vx,vy\)")
    acc = v35_behavior_parse_named_pair(q, r"Acceleration \(ax,ay\)")
    cb = v35_behavior_parse_named_pair(q, r"Can Bus")
    for name, pair in [("ego_velocity", vel), ("ego_accel", acc), ("ego_canbus", cb)]:
        if pair is None:
            d[f"feat__behctx_{name}_x"] = math.nan
            d[f"feat__behctx_{name}_y"] = math.nan
            d[f"feat__behctx_{name}_norm"] = math.nan
        else:
            d[f"feat__behctx_{name}_x"] = pair[0]
            d[f"feat__behctx_{name}_y"] = pair[1]
            d[f"feat__behctx_{name}_norm"] = float(math.sqrt(pair[0]*pair[0] + pair[1]*pair[1]))
    for label, key in [(r"Heading Angular Velocity \(v_yaw\)", "ego_yaw_rate"), (r"Heading Speed", "ego_heading_speed"), (r"Steering", "ego_steering")]:
        v = v35_behavior_parse_named_scalar(q, label)
        d[f"feat__behctx_{key}"] = math.nan if v is None else float(v)
    pairs = v35_behavior_parse_pairs(q)
    hist = pairs[-6:] if len(pairs) >= 2 else []
    d.update(v35_behavior_traj_features(hist, "feat__behctx_hist_"))
    return d


def v35_behavior_features_from_plan_row(plan_row: Optional[Dict[str, Any]]) -> Dict[str, float]:
    feat: Dict[str, float] = {"feat__behout_feature_found": 0.0}
    if not isinstance(plan_row, dict):
        return feat
    ans_pts = v35_behavior_parse_answer_traj(plan_row.get("answer", ""))
    feat.update(v35_behavior_traj_features(ans_pts, "feat__behout_"))
    feat.update(v35_behavior_question_context_features(str(plan_row.get("question", ""))))
    feat["feat__behout_answer_pair_count"] = float(len(ans_pts))
    feat["feat__behout_answer_available"] = 1.0 if len(ans_pts) >= 2 else 0.0
    feat["feat__behout_feature_found"] = 1.0
    return feat


def v35_load_plan_behavior_lookup(result_dir: Path) -> Dict[str, Dict[str, float]]:
    plan_path = result_dir / "plan_conv.json"
    lookup: Dict[str, Dict[str, float]] = {}
    if not plan_path.exists():
        return lookup
    for row in load_jsonl(plan_path):
        sid = v35_behavior_sample_clean(row.get("id") or row.get("sample_id") or "")
        if sid:
            lookup[sid] = v35_behavior_features_from_plan_row(row)
    return lookup

def compute_v26_component_features(rr: Dict[str, Any]) -> Dict[str, float]:
    """v22 component-comparison features.

    These are online-computable features designed for the third component in v22:
    fine-grained trajectory risk / zero-plan plausibility / non-collapse rescue.
    They do not use clean-paired planning metrics, GT, or harmful labels.
    """
    out: Dict[str, float] = {}
    eout = rr.get("output_sanity_features", {}) or {}
    eparse = rr.get("e_parse_features", {}) or {}
    stage = rr.get("stage_feature_flat", {}) or {}
    vec = (rr.get("object_vectors", {}) or {}).get("can_bus") or []

    def sf(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    parse_success = sf(eparse.get("parse_success", 1.0), 1.0)
    answer_is_maxlen = sf(eparse.get("answer_is_maxlen_like", eout.get("answer_is_maxlen_like", 0.0)), 0.0)
    parse_fallback = sf(eparse.get("parse_fallback_zero", 0.0), 0.0)
    no_numeric = sf(eparse.get("parse_fail_no_numeric_tokens", 0.0), 0.0)
    coord_pairs = sf(eparse.get("raw_coord_pair_count", 1.0), 1.0)
    parse_collapse = float(int(answer_is_maxlen >= 0.5 or parse_success < 0.5 or parse_fallback >= 0.5 or no_numeric >= 0.5 or coord_pairs <= 0.0))
    out["v26__collapse_rule_alarm"] = parse_collapse

    final_disp = sf(eout.get("final_disp", eparse.get("parsed_final_disp", 0.0)), 0.0)
    path_length = sf(eout.get("path_length", eparse.get("parsed_path_length", 0.0)), 0.0)
    step_mean = sf(eout.get("step_mean", 0.0), 0.0)
    step_max = sf(eout.get("step_max", 0.0), 0.0)
    endpoint_abs_y = sf(eout.get("endpoint_abs_y_proxy", 0.0), 0.0)
    heading_change = sf(eout.get("heading_change_proxy", 0.0), 0.0)
    turn_sum = sf(eout.get("turn_angle_abs_sum_proxy", 0.0), 0.0)
    straightness = sf(eout.get("straightness_proxy", 0.0), 0.0)
    traj_zero = sf(eout.get("traj_all_zero", 0.0), 0.0)

    out.update({
        "v26__planned_final_disp": final_disp,
        "v26__planned_path_length": path_length,
        "v26__planned_step_mean": step_mean,
        "v26__planned_step_max": step_max,
        "v26__endpoint_abs_y_proxy": endpoint_abs_y,
        "v26__heading_change_proxy": heading_change,
        "v26__turn_angle_abs_sum_proxy": turn_sum,
        "v26__straightness_proxy": straightness,
        "v26__near_zero_plan": float(int(final_disp <= 0.2 or path_length <= 0.2 or traj_zero >= 0.5)),
        "v26__valid_output_noncollapse": float(int(parse_collapse < 0.5 and parse_success >= 0.5 and coord_pairs > 0.0)),
    })

    ego = v19_ego_motion_context_features_from_canbus_vector(vec)
    motion_l2 = sf(ego.get("egoctx__motion_proxy_l2", 0.0), 0.0)
    motion_abs_sum = sf(ego.get("egoctx__motion_proxy_abs_sum", 0.0), 0.0)
    translation_l2 = sf(ego.get("egoctx__translation_xyz__l2", 0.0), 0.0)
    patch_angle_l2 = sf(ego.get("egoctx__patch_angle_pair__l2", 0.0), 0.0)
    near_zero = float(int(final_disp <= 0.2 or path_length <= 0.2 or traj_zero >= 0.5))
    online_ctx = rr.get("online_context_features", {}) or {}
    agent = rr.get("v26_agent_features", {}) or {}
    cmd_right = sf(online_ctx.get("v26__command_right", 0.0), 0.0)
    cmd_left = sf(online_ctx.get("v26__command_left", 0.0), 0.0)
    cmd_forward = sf(online_ctx.get("v26__command_forward", 0.0), 0.0)
    cmd_turn = sf(online_ctx.get("v26__command_turn", 0.0), 0.0)
    lateral_range = sf(eout.get("lateral_range_proxy", 0.0), 0.0)
    long_range = sf(eout.get("longitudinal_range_proxy", 0.0), 0.0)
    curvature = sf(eout.get("curvature_proxy", 0.0), 0.0)
    smoothness = sf(eout.get("smoothness_proxy", 0.0), 0.0)
    min_clear = sf(agent.get("v26__traj_agent_min_clearance_approx", math.inf), math.inf)
    front_dist = sf(agent.get("v26__agent_front_nearest_dist", math.inf), math.inf)
    nearest = sf(agent.get("v26__agent_nearest_center_dist", math.inf), math.inf)
    collision_proxy = sf(agent.get("v26__traj_agent_collision_risk_proxy", 0.0), 0.0)
    ego_pose = rr.get("v26_ego_pose_features", {}) or {}
    ego_speed_prev = sf(ego_pose.get("v26__ego_pose_speed_prev", math.nan), math.nan)
    ego_yaw_rate_prev = sf(ego_pose.get("v26__ego_pose_yaw_rate_prev", math.nan), math.nan)
    low_speed_zero_veto = float(int(parse_collapse >= 0.5 and near_zero >= 0.5 and math.isfinite(ego_speed_prev) and ego_speed_prev <= 0.01))
    out.update({
        "v26__ego_motion_proxy_l2": motion_l2,
        "v26__ego_motion_proxy_abs_sum": motion_abs_sum,
        "v26__ego_translation_l2": translation_l2,
        "v26__ego_patch_angle_l2": patch_angle_l2,
        "v26__zero_plan_with_nonzero_motion_proxy": float(int(near_zero >= 0.5 and motion_l2 > 1e-6)),
        "v26__zero_plan_with_input_alarm": float(int(near_zero >= 0.5 and int(rr.get("raw_input_alarm", 0) or 0) == 1)),
        "v26__noncollapse_input_alarm": float(int(parse_collapse < 0.5 and int(rr.get("raw_input_alarm", 0) or 0) == 1)),
        "v26__cmd_right": cmd_right,
        "v26__cmd_left": cmd_left,
        "v26__cmd_forward": cmd_forward,
        "v26__cmd_turn": cmd_turn,
        "v26__cmd_forward_zero_conflict": float(int(cmd_forward >= 0.5 and near_zero >= 0.5)),
        "v26__cmd_turn_zero_conflict": float(int(cmd_turn >= 0.5 and near_zero >= 0.5)),
        "v26__cmd_forward_lateral_conflict": float(int(cmd_forward >= 0.5 and lateral_range > max(1.0, long_range * 0.75))),
        "v26__cmd_turn_low_heading_change": float(int(cmd_turn >= 0.5 and heading_change < 0.05 and path_length > 1.0)),
        "v26__traj_lateral_range": lateral_range,
        "v26__traj_longitudinal_range": long_range,
        "v26__traj_curvature_proxy": curvature,
        "v26__traj_smoothness_proxy": smoothness,
        "v26__agent_nearest_center_dist": nearest,
        "v26__agent_front_nearest_dist": front_dist,
        "v26__traj_agent_min_clearance": min_clear,
        "v26__traj_agent_collision_risk_proxy": collision_proxy,
        "v26__ego_pose_speed_prev": ego_speed_prev,
        "v26__ego_pose_yaw_rate_prev": ego_yaw_rate_prev,
        "v26__speed_veto_low_speed_zero_collapse": low_speed_zero_veto,
        "v26__speed_veto_collapse_alarm": float(int(parse_collapse >= 0.5 and low_speed_zero_veto < 0.5)),
        "v26__noncollapse_collision_proxy": float(int(parse_collapse < 0.5 and collision_proxy >= 0.5)),
        "v26__noncollapse_low_clearance": float(int(parse_collapse < 0.5 and min_clear <= 2.0)),
    })

    d_stages = [
        "stage__d_llm_layer0__output_last__finite_ratio",
        "stage__d_llm_layer6__output_last__finite_ratio",
        "stage__d_llm_layer12__output_last__finite_ratio",
        "stage__d_llm_layer18__output_last__finite_ratio",
        "stage__d_llm_layer23__output_last__finite_ratio",
    ]
    d_drop = max([max(0.0, 1.0 - sf(stage.get(k), 1.0)) for k in d_stages] + [0.0])
    s_drop = max(0.0, 1.0 - sf(stage.get("stage__s_bev_encoder__output_last__finite_ratio"), 1.0))
    out.update({
        "v26__d_finite_drop_max": float(d_drop),
        "v26__sbev_finite_drop": float(s_drop),
        "v26__noncollapse_d_drop": float(int(parse_collapse < 0.5 and d_drop > 1e-6)),
        "v26__noncollapse_sbev_drop": float(int(parse_collapse < 0.5 and s_drop > 1e-6)),
    })
    return out

def flatten_training_row_for_matrix(rr: Dict[str, Any]) -> Dict[str, Any]:
    """Create one flat feature row for lightweight harmful-classifier training.

    Labels/deltas are included for offline training/evaluation only and must never be
    used as online detector inputs.
    """
    row: Dict[str, Any] = {
        "row_index": rr.get("row_index"),
        "sample_id": rr.get("sample_id"),
        "row_source": rr.get("row_source", ""),
        "is_experiment_row": int(rr.get("is_experiment_row", 0) or 0),
        "is_fault_selected": int(rr.get("is_fault_selected", 0) or 0),
        "fault_spec_count": int(rr.get("fault_spec_count", 0) or 0),
        "fault_attempted_count": int(rr.get("fault_attempted_count", 0) or 0),
        "fault_applied_count": int(rr.get("fault_applied_count", 0) or 0),
        "fault_effective_count": int(rr.get("fault_effective_count", 0) or 0),
        "scene_id": rr.get("scene_id", ""),
        "scene_token": rr.get("scene_token", ""),
        "frame_index": rr.get("frame_index", ""),
        "timestamp": rr.get("timestamp", ""),
        "is_clean": int(rr.get("is_clean", 0) or 0),
        "is_effective_injected": int(rr.get("injected_effective", 0) or 0),
        "fault_target": rr.get("fault_target", "can_bus"),
        "fault_group": rr.get("fault_group", "unknown"),
        "fault_scope": rr.get("fault_scope", "no_fault"),
        "s0_fault_spec_count": int(rr.get("s0_fault_spec_count", 0) or 0),
        "s1_fault_spec_count": int(rr.get("s1_fault_spec_count", 0) or 0),
        "s0_fault_attempted_count": int(rr.get("s0_fault_attempted_count", 0) or 0),
        "s0_fault_effective_count": int(rr.get("s0_fault_effective_count", 0) or 0),
        "s1_fault_attempted_count": int(rr.get("s1_fault_attempted_count", 0) or 0),
        "s1_fault_applied_count": int(rr.get("s1_fault_applied_count", 0) or 0),
        "s1_fault_effective_count": int(rr.get("s1_fault_effective_count", 0) or 0),
        "injected_effective": int(rr.get("injected_effective", 0) or 0),
        "mild_harmful": int(rr.get("mild_harmful", 0) or 0),
        "main_harmful": int(rr.get("main_harmful", 0) or 0),
        "strict_harmful": int(rr.get("strict_harmful", 0) or 0),
        "delta_l2_avg": rr.get("delta_l2_avg"),
        "delta_collision_avg": rr.get("delta_collision_avg"),
        "raw_input_alarm": int(rr.get("raw_input_alarm", 0) or 0),
        "e_output_sanity_alarm": int(rr.get("e_output_sanity_alarm", 0) or 0),
        "v19_output_first_harm_alarm": int(rr.get("v19_output_first_harm_alarm", 0) or 0),
    }
    faults = rr.get("faults") or []
    if faults:
        f0 = faults[0] or {}
        row.update({
            "fault_index": f0.get("idx"),
            "fault_bit_op": f0.get("bit_op"),
            "fault_bit_count": f0.get("bit_count"),
            "fault_before": f0.get("before"),
            "fault_after": f0.get("after"),
        })
        if _is_number_like(f0.get("after")) and _is_number_like(f0.get("before")):
            before = float(f0.get("before"))
            after = float(f0.get("after"))
            row["fault_abs_delta"] = abs(after - before)
            row["fault_abs_ratio"] = abs(after) / (abs(before) + 1e-12)
    feat: Dict[str, float] = {}
    _flatten_nested_numeric("s0", rr.get("s0_feature_flat", {}), feat)
    _flatten_nested_numeric("stage", rr.get("stage_feature_flat", {}), feat)
    _flatten_nested_numeric("e_output", rr.get("output_sanity_features", {}), feat)
    _flatten_nested_numeric("e_parse", rr.get("e_parse_features", {}), feat)
    _flatten_nested_numeric("e_report", rr.get("output_sanity_report", {}), feat)
    _flatten_nested_numeric("v19_e_conf", compute_v19_e_confidence_features(rr.get("output_sanity_report", {}), rr.get("output_sanity_features", {})), feat)
    _flatten_nested_numeric("v19_e_context", compute_v19_e_contextual_features(rr), feat)
    _flatten_nested_numeric("v19_fault_induced_proxy", compute_v19_fault_induced_proxy_features(rr), feat)
    _flatten_nested_numeric("v26_component", compute_v26_component_features(rr), feat)
    _flatten_nested_numeric("v26_canbus_group", compute_v26_canbus_group_features_from_vector((rr.get("object_vectors", {}) or {}).get("can_bus", [])), feat)
    _flatten_nested_numeric("v26_risk", compute_v26_risk_availability_features(rr.get("scene_context_summary", {}) or {}), feat)
    _flatten_nested_numeric("v26_planning", compute_v26_planning_candidate_proxy_features(rr), feat)
    # v35.1 exact schema stored ego-pose online context under the v26_online_context namespace.
    # Keep the original online_context_features and add v26_ego_pose_features into that
    # namespace with exact historical column names; also preserve the explicit
    # feat__v26_ego_pose__* namespace for diagnostics.
    v26_online_context_exact = dict(rr.get("online_context_features", {}) or {})
    for _k, _v in (rr.get("v26_ego_pose_features", {}) or {}).items():
        v26_online_context_exact.setdefault(_k, _v)
    _flatten_nested_numeric("v26_online_context", v26_online_context_exact, feat)
    _flatten_nested_numeric("v26_agent", rr.get("v26_agent_features", {}), feat)
    _flatten_nested_numeric("v26_ego_pose", rr.get("v26_ego_pose_features", {}), feat)
    _flatten_nested_numeric("v26_online_heads", rr.get("v26_online_head_features", {}), feat)
    _flatten_nested_numeric("v26_branch_projector", compute_v26_branch_projector_features(rr), feat)
    _flatten_nested_numeric("temporal_context", rr.get("temporal_context_features", {}), feat)
    _flatten_nested_numeric("scene_context", flatten_scene_context_for_training(rr.get("scene_context_summary", {})), feat)
    _flatten_nested_numeric("isolation", rr.get("isolation_info", {}) or {}, feat, max_list=4)
    for k, v in feat.items():
        row[f"feat__{k}"] = v

    # v35.1 behavior-enriched exact columns are full feature names already
    # (feat__behout_* / feat__behctx_*).  They are computed from current
    # plan_conv question/answer only and do not use clean/GT/fault deltas.
    for k, v in (rr.get("v35_behavior_features", {}) or {}).items():
        if str(k).startswith("feat__"):
            row[str(k)] = v

    # The exported v35.1 artifact contains a pandas-mangled duplicate column
    # feat__v26_component__v26__collapse_rule_alarm.1 from the historical matrix.
    # Provide it exactly instead of relying on duplicate CSV headers.
    if "feat__v26_component__v26__collapse_rule_alarm" in row:
        row["feat__v26_component__v26__collapse_rule_alarm.1"] = row.get("feat__v26_component__v26__collapse_rule_alarm")
    return row


V53_ALLOWED_EXACT_ONLINE_COLUMNS = {
    "raw_input_alarm",
    "critical_input_alarm",
    "suspicious_input_alarm",
    "e_output_sanity_alarm",
    "v19_output_first_harm_alarm",
}


def v53_deployable_feature_row_from_record(rr: Dict[str, Any]) -> Dict[str, Any]:
    """Return only online-deployable detector inputs for v53.

    This row is built from process-captured feature buffers and current model
    output behavior. It deliberately drops GT, clean-reference deltas, harmful
    labels, fault metadata, target metadata, and all analysis-only fields.

    Important exact-schema note:
    v35.1 Line-A feature_columns include a small set of top-level online alarm
    columns in addition to feat__* columns.  They are current-sample detector
    proxies, not labels, and were explicitly allowed by the v35 training code.
    v53.1 therefore keeps only those exact allowlisted top-level columns plus
    feat__* columns.  No fuzzy or similarity-based column matching is used.
    """
    flat = flatten_training_row_for_matrix(rr)
    out: Dict[str, Any] = {
        "sample_id": flat.get("sample_id", rr.get("sample_id", "")),
        "row_source": flat.get("row_source", rr.get("row_source", "")),
    }
    for k, v in flat.items():
        ks = str(k)
        if ks.startswith("feat__") or ks in V53_ALLOWED_EXACT_ONLINE_COLUMNS:
            out[ks] = v
    return out


def v53_analysis_label_row_from_deltas(sample_id: str, row_source: str, injected_effective: int, deltas: Dict[str, Any]) -> Dict[str, Any]:
    """Build posthoc evaluation labels from raw deltas only.

    Labels are for offline recall/FAR computation. They are never added to
    deployable feature rows and never used by the online detector.
    """
    def sf(x: Any) -> float:
        try:
            v = float(x)
            return v if math.isfinite(v) else math.nan
        except Exception:
            return math.nan
    dl2 = sf(deltas.get("delta_l2_avg"))
    dc = sf(deltas.get("delta_collision_avg"))
    inj = int(injected_effective or 0)
    harmful = int(inj == 1 and ((math.isfinite(dl2) and dl2 > 0.5) or (math.isfinite(dc) and dc > 0.0)))
    severe = int(inj == 1 and ((math.isfinite(dl2) and dl2 > 1.0) or (math.isfinite(dc) and dc > 0.0)))
    return {
        "sample_id": sample_id,
        "row_source": row_source,
        "injected_effective": inj,
        "delta_l2_avg": dl2,
        "delta_collision_avg": dc,
        "harmful": harmful,
        "severe_harmful": severe,
        "label_policy": "posthoc_only_not_detector_feature",
    }


def write_training_feature_matrix(rows: List[Dict[str, Any]], csv_path: Path, jsonl_path: Path, max_columns: int = 9000) -> Dict[str, Any]:
    """Write a flat CSV and JSONL matrix from stage training rows."""
    flat_rows = [flatten_training_row_for_matrix(r) for r in rows]
    if not flat_rows:
        csv_path.write_text("", encoding="utf-8")
        jsonl_path.write_text("", encoding="utf-8")
        return {"rows": 0, "columns": 0, "csv": str(csv_path), "jsonl": str(jsonl_path)}
    metadata_cols = [
        "row_index", "sample_id", "scene_id", "scene_token", "frame_index", "timestamp",
        "is_clean", "is_effective_injected",
        "fault_target", "fault_group", "fault_index", "fault_bit_op", "fault_bit_count",
        "fault_before", "fault_after", "fault_abs_delta", "fault_abs_ratio", "injected_effective",
        "mild_harmful", "main_harmful", "strict_harmful", "delta_l2_avg", "delta_collision_avg",
        "raw_input_alarm", "e_output_sanity_alarm", "v19_output_first_harm_alarm",
        "fault_scope", "s0_fault_spec_count", "s1_fault_spec_count",
        "s0_fault_attempted_count", "s0_fault_effective_count",
        "s1_fault_attempted_count", "s1_fault_applied_count", "s1_fault_effective_count",
        "feat__v26_component__v26__collapse_rule_alarm",
        "feat__isolation__reset_model_state_between_samples",
        "feat__isolation__model_temporal_reset_before_forward__enabled",
        "feat__isolation__model_temporal_reset_before_forward__prev_frame_info_reset",
        "feat__isolation__model_temporal_reset_before_forward__attr_reset",
        "feat__isolation__model_temporal_reset_before_forward__reset_errors",
        "feat__isolation__model_temporal_reset_after_record__enabled",
        "feat__isolation__model_temporal_reset_after_record__prev_frame_info_reset",
        "feat__isolation__model_temporal_reset_after_record__attr_reset",
        "feat__isolation__model_temporal_reset_after_record__reset_errors",
    ]
    feature_cols = sorted({k for r in flat_rows for k in r.keys() if k.startswith("feat__")})[:max_columns]
    cols = [c for c in metadata_cols if any(c in r for r in flat_rows)] + feature_cols
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in flat_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in flat_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"rows": len(flat_rows), "columns": len(cols), "feature_columns": len(feature_cols), "csv": str(csv_path), "jsonl": str(jsonl_path)}

def summarize_scene_context_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r.get("scene_context_summary", {}) or {} for r in records]
    out: Dict[str, Any] = {"version": VERSION, "num_records": len(records)}
    keys = ["available", "candidate_count", "has_box_like", "has_agent_like", "has_map_like", "has_lane_like", "has_drivable_like", "has_occupancy_like", "has_command_like", "has_route_like", "has_gt_like_candidate"]
    for k in keys:
        vals = []
        for row in rows:
            try:
                vals.append(float(row.get(k, 0) or 0))
            except Exception:
                pass
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "min": float(np.min(vals))}
    # Union of candidate paths, capped for readability.
    path_counts: Dict[str, int] = {}
    gt_like_paths: Dict[str, int] = {}
    for row in rows:
        for c in row.get("candidates", []) or []:
            path = str(c.get("path", ""))
            if not path:
                continue
            path_counts[path] = path_counts.get(path, 0) + 1
            if int(c.get("gt_like", 0) or 0):
                gt_like_paths[path] = gt_like_paths.get(path, 0) + 1
    out["top_candidate_paths"] = sorted(path_counts.items(), key=lambda kv: kv[1], reverse=True)[:100]
    out["top_gt_like_candidate_paths"] = sorted(gt_like_paths.items(), key=lambda kv: kv[1], reverse=True)[:100]
    return out


def summarize_stage_distributions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    values: Dict[str, List[float]] = {}
    for r in records:
        flat = flatten_stage_summary_for_training(r.get("stage_summaries", {}))
        for k, v in flat.items():
            if np.isfinite(float(v)):
                values.setdefault(k, []).append(float(v))
    out: Dict[str, Any] = {"version": VERSION, "num_records": len(records), "features": {}}
    for k, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size:
            out["features"][k] = {
                "count": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "p50": float(np.quantile(arr, 0.50)),
                "p95": float(np.quantile(arr, 0.95)),
                "p99": float(np.quantile(arr, 0.99)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return out


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def vector_extended(vec: np.ndarray) -> Dict[str, float]:
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    basic = vector_basic(x)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            **basic,
            "zero_ratio": 1.0,
            "pos_ratio": 0.0,
            "neg_ratio": 0.0,
            "mean": math.inf,
            "std": math.inf,
        }
    return {
        **basic,
        "zero_ratio": float(np.mean(finite == 0.0)),
        "pos_ratio": float(np.mean(finite > 0.0)),
        "neg_ratio": float(np.mean(finite < 0.0)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def canbus_mlp_features(vec: np.ndarray, dim_stats: Optional[Dict[str, Any]]) -> Dict[str, float]:
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    basic = vector_extended(x)
    st = (dim_stats or {}).get("can_bus_mlp", {})
    med = np.asarray(st.get("median", np.zeros_like(x)), dtype=np.float64)
    mad = np.asarray(st.get("mad", np.ones_like(x)), dtype=np.float64)
    if med.shape != x.shape:
        med = np.zeros_like(x)
    if mad.shape != x.shape:
        mad = np.ones_like(x)
    z = robust_z(x, med, mad)
    finite_z = z[np.isfinite(z)]
    median_norm = float(st.get("median_norm", 1.0))
    median_absmax = float(st.get("median_absmax", 1.0))
    return {
        **basic,
        "max_dim_robust_z": float(finite_z.max()) if finite_z.size else math.inf,
        "norm_ratio": float(basic["norm2"] / max(median_norm, 1e-12)) if np.isfinite(basic["norm2"]) else math.inf,
        "abs_max_ratio": float(basic["abs_max"] / max(median_absmax, 1e-12)) if np.isfinite(basic["abs_max"]) else math.inf,
    }


def _sign_bucket(x: float) -> int:
    if not np.isfinite(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def small_signed_structure_features(cb: np.ndarray, dim_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cb = np.asarray(cb, dtype=np.float64).reshape(-1)
    idxs = np.asarray(SMALL_SIGNED_STATE_IDXS, dtype=np.int64)
    vals = cb[idxs] if cb.size > max(idxs) else np.zeros(len(idxs), dtype=np.float64)
    abs_vals = np.abs(vals)
    log_abs = np.log10(abs_vals + 1e-300)
    st = (dim_stats or {}).get("small_signed_structure", {})
    per_stats = st.get("per_index", {})
    per_index: Dict[str, Any] = {}
    min_log_abs_margin = 0.0
    max_log_abs_margin = 0.0
    collapse_hit_count = 0
    sign_rare_count = 0
    for local_i, idx in enumerate(SMALL_SIGNED_STATE_IDXS):
        key = str(idx)
        ist = per_stats.get(key, {})
        la = float(log_abs[local_i]) if np.isfinite(log_abs[local_i]) else -math.inf
        val = float(vals[local_i]) if np.isfinite(vals[local_i]) else math.inf
        sign = _sign_bucket(val)
        lower = float(ist.get("log_abs_lower", -math.inf))
        upper = float(ist.get("log_abs_upper", math.inf))
        sign_probs = ist.get("sign_prob", {})
        sign_prob = float(sign_probs.get(str(sign), 1.0))
        low_margin = max(0.0, lower - la)
        high_margin = max(0.0, la - upper)
        min_log_abs_margin = max(min_log_abs_margin, low_margin)
        max_log_abs_margin = max(max_log_abs_margin, high_margin)
        collapse = int(la < lower)
        rare_sign = int(sign_prob < float(ist.get("rare_sign_prob", 0.02)))
        collapse_hit_count += collapse
        sign_rare_count += rare_sign
        per_index[key] = {
            "value": val,
            "abs": float(abs_vals[local_i]) if np.isfinite(abs_vals[local_i]) else math.inf,
            "log_abs": la,
            "sign": int(sign),
            "sign_prob": sign_prob,
            "log_abs_lower": lower,
            "log_abs_upper": upper,
            "low_margin": low_margin,
            "high_margin": high_margin,
            "collapse": collapse,
            "rare_sign": rare_sign,
        }

    context_mahalanobis = 0.0
    ctx = st.get("context", {})
    dims = ctx.get("dims", [])
    if dims and "mean" in ctx and "inv_cov" in ctx and cb.size > max(dims):
        x = cb[np.asarray(dims, dtype=np.int64)].astype(np.float64)
        mean = np.asarray(ctx.get("mean"), dtype=np.float64)
        inv_cov = np.asarray(ctx.get("inv_cov"), dtype=np.float64)
        if x.size == mean.size and inv_cov.shape == (x.size, x.size) and np.isfinite(x).all():
            d = x - mean
            val = float(d.T @ inv_cov @ d)
            context_mahalanobis = val if np.isfinite(val) else math.inf
        else:
            context_mahalanobis = math.inf

    return {
        "per_index": per_index,
        "min_log_abs_margin": float(min_log_abs_margin),
        "max_log_abs_margin": float(max_log_abs_margin),
        "collapse_hit_count": int(collapse_hit_count),
        "sign_rare_count": int(sign_rare_count),
        "context_mahalanobis": float(context_mahalanobis),
    }


def translation_shape_features(cb: np.ndarray, dim_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cb = np.asarray(cb, dtype=np.float64).reshape(-1)
    if cb.size < 2:
        x, y = 0.0, 0.0
    else:
        x, y = float(cb[0]), float(cb[1])
    xy_log_ratio = float(np.log10((abs(x) + 1e-300) / (abs(y) + 1e-300)))
    st = (dim_stats or {}).get("translation_shape", {})
    lower = float(st.get("xy_log_ratio_lower", -math.inf))
    upper = float(st.get("xy_log_ratio_upper", math.inf))
    strong_lower = float(st.get("xy_log_ratio_strong_lower", lower))
    strong_upper = float(st.get("xy_log_ratio_strong_upper", upper))
    low_margin = max(0.0, lower - xy_log_ratio)
    high_margin = max(0.0, xy_log_ratio - upper)
    return {
        "x": x,
        "y": y,
        "xy_log_ratio": xy_log_ratio,
        "xy_log_ratio_lower": lower,
        "xy_log_ratio_upper": upper,
        "xy_log_ratio_strong_lower": strong_lower,
        "xy_log_ratio_strong_upper": strong_upper,
        "xy_low_margin": low_margin,
        "xy_high_margin": high_margin,
        "xy_shape_alarm": int(xy_log_ratio < lower or xy_log_ratio > upper),
        "xy_shape_strong_alarm": int(xy_log_ratio < strong_lower or xy_log_ratio > strong_upper),
    }


def extract_s0_canbus_features(cb: np.ndarray, can_bus_mlp_vec: np.ndarray, dim_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "can_bus_groups": canbus_group_features(cb, dim_stats),
        "can_bus_direct": canbus_direct_use_features(cb, dim_stats),
        "can_bus_mlp": canbus_mlp_features(can_bus_mlp_vec, dim_stats),
        "small_signed_structure": small_signed_structure_features(cb, dim_stats),
        "translation_shape": translation_shape_features(cb, dim_stats),
    }


def build_dim_stats(clean_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "can_bus_groups": {},
        "can_bus_direct": {},
        "can_bus_xy": {},
        "can_bus_patch": {},
        "can_bus_mlp": {},
        "small_signed_structure": {},
        "translation_shape": {},
    }
    cb_vecs = [np.asarray(r["object_vectors"]["can_bus"], dtype=np.float64).reshape(-1)
               for r in clean_records if "can_bus" in r.get("object_vectors", {})]
    if cb_vecs:
        for gname, idxs in CANBUS_GROUPS.items():
            group_vecs = [v[np.asarray(idxs, dtype=np.int64)] for v in cb_vecs]
            med, mad = robust_stats_matrix(group_vecs)
            stats["can_bus_groups"][gname] = {
                "median": med.tolist(),
                "mad": mad.tolist(),
                "median_norm": float(np.median([np.linalg.norm(v) for v in group_vecs])),
                "median_absmax": float(np.median([np.max(np.abs(v)) for v in group_vecs])),
                "indices": list(idxs),
            }
        direct_vecs = [v[np.asarray(CANBUS_DIRECT_IDXS, dtype=np.int64)] for v in cb_vecs]
        med, mad = robust_stats_matrix(direct_vecs)
        stats["can_bus_direct"] = {
            "median": med.tolist(),
            "mad": mad.tolist(),
            "median_norm": float(np.median([np.linalg.norm(v) for v in direct_vecs])),
            "indices": list(CANBUS_DIRECT_IDXS),
        }
        xy_vecs = [v[[0, 1]] for v in cb_vecs]
        med, mad = robust_stats_matrix(xy_vecs)
        stats["can_bus_xy"] = {"median": med.tolist(), "mad": mad.tolist()}
        patch_vecs = [v[[16, 17]] for v in cb_vecs]
        med, mad = robust_stats_matrix(patch_vecs)
        stats["can_bus_patch"] = {"median": med.tolist(), "mad": mad.tolist()}

        xy_log_ratios = np.asarray([
            np.log10((abs(float(v[0])) + 1e-300) / (abs(float(v[1])) + 1e-300))
            for v in cb_vecs
        ], dtype=np.float64)
        stats["translation_shape"] = {
            "xy_log_ratio_p001": float(np.quantile(xy_log_ratios, 0.001)),
            "xy_log_ratio_p999": float(np.quantile(xy_log_ratios, 0.999)),
            "xy_log_ratio_lower": float(np.quantile(xy_log_ratios, 0.001) - 0.50),
            "xy_log_ratio_upper": float(np.quantile(xy_log_ratios, 0.999) + 0.50),
            "xy_log_ratio_strong_lower": float(np.quantile(xy_log_ratios, 0.001) - 1.00),
            "xy_log_ratio_strong_upper": float(np.quantile(xy_log_ratios, 0.999) + 1.00),
        }

        # v11 structure calibration for small_signed_state. The severe v9 misses
        # were exponent collapses to extremely tiny magnitudes. We therefore store
        # per-index log-magnitude envelopes and a context covariance over small
        # state + pose/angle fields.
        per_index: Dict[str, Any] = {}
        for idx in SMALL_SIGNED_STATE_IDXS:
            vals = np.asarray([v[idx] for v in cb_vecs], dtype=np.float64)
            abs_vals = np.abs(vals)
            log_abs = np.log10(abs_vals + 1e-300)
            signs = np.asarray([_sign_bucket(float(x)) for x in vals], dtype=np.int64)
            sign_prob = {str(s): float(np.mean(signs == s)) for s in [-1, 0, 1]}
            per_index[str(idx)] = {
                "idx": int(idx),
                "value_p001": float(np.quantile(vals, 0.001)),
                "value_p999": float(np.quantile(vals, 0.999)),
                "abs_p001": float(np.quantile(abs_vals, 0.001)),
                "abs_p999": float(np.quantile(abs_vals, 0.999)),
                # Lower is intentionally far below the clean p0.1% envelope to avoid
                # increasing false alarms while still catching exponent underflow.
                "log_abs_lower": float(np.quantile(log_abs, 0.001) - 3.0),
                "log_abs_strong_lower": float(np.quantile(log_abs, 0.001) - 6.0),
                "log_abs_upper": float(np.quantile(log_abs, 0.999) + 1.0),
                "log_abs_strong_upper": float(np.quantile(log_abs, 0.999) + 2.0),
                "sign_prob": sign_prob,
                "rare_sign_prob": 0.02,
            }
        context_dims = SMALL_SIGNED_STATE_IDXS + [3, 4, 5, 6, 9, 13, 16, 17]
        X = np.asarray([v[np.asarray(context_dims, dtype=np.int64)] for v in cb_vecs], dtype=np.float64)
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]], dtype=np.float64)
        reg = float(np.trace(cov) / max(cov.shape[0], 1)) * 1e-4 + 1e-8
        inv_cov = np.linalg.pinv(cov + np.eye(cov.shape[0], dtype=np.float64) * reg)
        stats["small_signed_structure"] = {
            "per_index": per_index,
            "context": {
                "dims": [int(x) for x in context_dims],
                "mean": mean.tolist(),
                "inv_cov": inv_cov.tolist(),
                "regularization": reg,
            },
        }

    mlp_vecs = [np.asarray(r["object_vectors"].get("can_bus_mlp", []), dtype=np.float64).reshape(-1)
                for r in clean_records if "can_bus_mlp" in r.get("object_vectors", {})]
    mlp_vecs = [v for v in mlp_vecs if v.size > 0]
    if mlp_vecs:
        med, mad = robust_stats_matrix(mlp_vecs)
        stats["can_bus_mlp"] = {
            "median": med.tolist(),
            "mad": mad.tolist(),
            "median_norm": float(np.median([np.linalg.norm(v) for v in mlp_vecs])),
            "median_absmax": float(np.median([np.max(np.abs(v)) for v in mlp_vecs])),
            "dim": int(med.size),
        }
    return stats


def recompute_features_for_records(records: List[Dict[str, Any]], dim_stats: Dict[str, Any]):
    for r in records:
        cb = np.asarray(r["object_vectors"]["can_bus"], dtype=np.float64).reshape(-1)
        mlp = np.asarray(r["object_vectors"].get("can_bus_mlp", []), dtype=np.float64).reshape(-1)
        r["features"] = extract_s0_canbus_features(cb, mlp, dim_stats)


# -----------------------------------------------------------------------------
# Thresholding / detection
# -----------------------------------------------------------------------------

def canbus_group_feature_list(gname: str) -> List[str]:
    return list(CANBUS_GROUP_FEATURE_MAP[gname])


def compute_thresholds(clean_records: List[Dict[str, Any]], args) -> Dict[str, Any]:
    dim_stats = build_dim_stats(clean_records)
    recompute_features_for_records(clean_records, dim_stats)
    thresholds: Dict[str, Any] = {
        "version": VERSION,
        "dim_stats": dim_stats,
        "feature_thresholds": {
            "can_bus_groups": {},
            "can_bus_direct": {},
            "can_bus_mlp": {},
            "small_signed_structure": {},
            "translation_shape": {},
        },
        "soft_thresholds": {},
        "target_observability": {"S0_can_bus": [r["sample_id"] for r in clean_records]},
        "fault_scope": {
            "targets": list(FAULT_TARGETS),
            "field": "exponent_only",
            "cardinality_weights": args.cardinality_weights,
            "bit_op_weights": args.bit_op_weights,
        },
        "hook": {
            "enabled": True,
            "suffix": args.can_bus_mlp_hook_suffix,
        },
    }

    for gname in CANBUS_GROUP_ORDER:
        thresholds["feature_thresholds"]["can_bus_groups"][gname] = {}
        for feat in canbus_group_feature_list(gname):
            vals = [r["features"]["can_bus_groups"][gname].get(feat) for r in clean_records]
            if feat == "finite_ratio":
                kind = "lower_only"; kwargs = {}
            elif feat in {"max_dim_robust_z", "top1_over_mean", "quat_norm_error", "angle_consistency_error", "abs_max_ratio"}:
                kind = "upper_only"; kwargs = {}
            elif feat == "norm_ratio":
                kind = "range"; kwargs = {}
            else:
                kind = "range"; kwargs = {}
            # v7 already tightened small_signed_state. Keep it, but do not push it
            # further because this group was a main false-negative source and over-
            # tightening tends to raise false alarms on clean data.
            if gname == "small_signed_state" and feat in {"max_dim_robust_z", "top1_over_mean"}:
                kwargs = {"mad_scale": 4.0, "strong_mad_scale": 6.0, "p_high": 0.99, "p_vhigh": 0.995}
            thr = compute_feature_threshold(vals, kind=kind, **kwargs)
            if thr is not None:
                thresholds["feature_thresholds"]["can_bus_groups"][gname][feat] = thr

    for feat in CANBUS_DIRECT_FEATURES:
        vals = [r["features"]["can_bus_direct"].get(feat) for r in clean_records]
        if feat == "finite_ratio":
            kind = "lower_only"
        elif feat in {"direct_max_dim_robust_z", "xy_max_dim_robust_z", "patch_max_dim_robust_z", "angle_consistency_error"}:
            kind = "upper_only"
        else:
            kind = "range"
        thr = compute_feature_threshold(
            vals,
            kind=kind,
            mad_scale=4.0 if feat != "direct_norm_ratio" else 6.0,
            strong_mad_scale=6.0 if feat != "direct_norm_ratio" else 10.0,
            p_high=0.99 if feat != "direct_norm_ratio" else 0.995,
            p_vhigh=0.995 if feat != "direct_norm_ratio" else 0.999,
        )
        if thr is not None:
            thresholds["feature_thresholds"]["can_bus_direct"][feat] = thr

    for feat in CANBUS_MLP_FEATURES:
        vals = [r["features"]["can_bus_mlp"].get(feat) for r in clean_records]
        if feat == "finite_ratio":
            kind = "lower_only"; kwargs = {}
        elif feat in {"max_dim_robust_z", "abs_max_ratio", "top1_over_mean", "zero_ratio", "pos_ratio", "neg_ratio", "std"}:
            kind = "upper_only"; kwargs = {"mad_scale": 5.0, "strong_mad_scale": 8.0, "p_high": 0.995, "p_vhigh": 0.999}
        elif feat == "norm_ratio":
            kind = "range"; kwargs = {"mad_scale": 5.0, "strong_mad_scale": 8.0}
        else:
            kind = "range"; kwargs = {"mad_scale": 6.0, "strong_mad_scale": 10.0}
        thr = compute_feature_threshold(vals, kind=kind, **kwargs)
        if thr is not None:
            thresholds["feature_thresholds"]["can_bus_mlp"][feat] = thr

    # v11 small_signed_state structural thresholds. Per-index log_abs thresholds
    # are stored in dim_stats because they are not a single scalar feature. The
    # context score uses a conservative upper-only threshold and is not allowed to
    # trigger by itself unless it is very strong.
    struct_thr: Dict[str, Any] = {
        "per_index": thresholds.get("dim_stats", {}).get("small_signed_structure", {}).get("per_index", {}),
    }
    vals = [r["features"].get("small_signed_structure", {}).get("context_mahalanobis", 0.0) for r in clean_records]
    thr = compute_feature_threshold(vals, kind="upper_only", mad_scale=6.0, strong_mad_scale=10.0, p_high=0.995, p_vhigh=0.999)
    if thr is not None:
        struct_thr["context_mahalanobis"] = thr
    thresholds["feature_thresholds"]["small_signed_structure"] = struct_thr

    tmp = copy.deepcopy(thresholds)
    tmp["soft_thresholds"] = {"can_bus": {"p995": 0.0, "p999": 0.0}, "combined": {"p995": 0.0, "p999": 0.0}}
    soft_vals = {"can_bus": [], "can_bus_value": [], "can_bus_mlp": [], "combined": []}
    for r in clean_records:
        rep = detect_s0(r["features"], tmp)
        soft_vals["can_bus"].append(float(rep["objects"]["can_bus"]["soft_score"]))
        soft_vals["can_bus_value"].append(float(rep["objects"]["can_bus"].get("value_soft_score", 0.0)))
        soft_vals["can_bus_mlp"].append(float(rep["objects"]["can_bus"].get("mlp_soft_score", 0.0)))
        soft_vals["combined"].append(float(rep["combined"]["max_soft_score"]))
    for k, vals in soft_vals.items():
        arr = np.asarray(vals, dtype=np.float64)
        thresholds["soft_thresholds"][k] = {
            "p995": float(np.quantile(arr, 0.995)) if arr.size else 0.0,
            "p999": float(np.quantile(arr, 0.999)) if arr.size else 0.0,
        }
    return thresholds



def detect_small_signed_structure(feats: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    thr_root = thresholds.get("feature_thresholds", {}).get("small_signed_structure", {})
    per_thr = thr_root.get("per_index", {})
    details: Dict[str, Any] = {}
    collapse_hits = 0
    strong_hits = 0
    high_hits = 0
    rare_sign_hits = 0
    soft = 0.0
    for key, item in feats.get("per_index", {}).items():
        ist = per_thr.get(str(key), {})
        la = float(item.get("log_abs", 0.0))
        lower = float(ist.get("log_abs_lower", -math.inf))
        strong_lower = float(ist.get("log_abs_strong_lower", -math.inf))
        upper = float(ist.get("log_abs_upper", math.inf))
        strong_upper = float(ist.get("log_abs_strong_upper", math.inf))
        low_alarm = int(la < lower)
        low_strong = int(la < strong_lower)
        high_alarm = int(la > upper)
        high_strong = int(la > strong_upper)
        sign_prob = float(item.get("sign_prob", 1.0))
        rare_sign = int(sign_prob < float(ist.get("rare_sign_prob", 0.02)))
        collapse_hits += low_alarm
        strong_hits += int(low_strong or high_strong)
        high_hits += high_alarm
        rare_sign_hits += rare_sign
        low_margin = max(0.0, lower - la)
        high_margin = max(0.0, la - upper)
        soft += low_margin + 0.5 * high_margin + 0.25 * rare_sign
        details[str(key)] = {
            **item,
            "low_alarm": low_alarm,
            "low_strong": low_strong,
            "high_alarm": high_alarm,
            "high_strong": high_strong,
            "rare_sign": rare_sign,
            "low_margin": low_margin,
            "high_margin": high_margin,
        }

    ctx_thr = thr_root.get("context_mahalanobis")
    ctx_val = float(feats.get("context_mahalanobis", 0.0))
    ctx_outcome = {"alarm": 0, "strong_alarm": 0, "margin": 0.0}
    if ctx_thr is not None:
        ctx_outcome = detect_value(ctx_val, ctx_thr)
        soft += 0.5 * float(ctx_outcome.get("margin", 0.0))

    # The severe v9 misses were exponent-collapse cases; catch them directly.
    # Context is auxiliary: it can promote a weak structural anomaly, but it is
    # not used as a stand-alone hard alarm to avoid repeating the v9 MLP FP issue.
    pred = int(
        collapse_hits >= 1
        or strong_hits >= 1
        or (high_hits >= 1 and int(ctx_outcome.get("alarm", 0)) == 1)
        or (rare_sign_hits >= 1 and int(ctx_outcome.get("strong_alarm", 0)) == 1)
    )
    return {
        "pred_label": pred,
        "collapse_hit_count": int(collapse_hits),
        "strong_count": int(strong_hits),
        "high_hit_count": int(high_hits),
        "rare_sign_count": int(rare_sign_hits),
        "context_mahalanobis": {"value": ctx_val, **ctx_outcome},
        "soft_score": float(soft),
        "details": details,
    }


def detect_object_can_bus_mlp(feats: Dict[str, float], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    details = {}
    hard = 0
    strong = 0
    soft = 0.0
    for feat in CANBUS_MLP_FEATURES:
        thr = thresholds.get("feature_thresholds", {}).get("can_bus_mlp", {}).get(feat)
        if thr is None:
            continue
        val = float(feats.get(feat, 0.0))
        outcome = detect_value(val, thr)
        details[feat] = {"value": val, **outcome}
        hard += int(outcome["alarm"])
        strong += int(outcome["strong_alarm"])
        if feat in {"finite_ratio", "max_dim_robust_z", "abs_max_ratio"}:
            w = 1.0
        elif feat in {"norm_ratio", "top1_over_mean"}:
            w = 0.8
        else:
            w = 0.45
        soft += w * float(outcome["margin"])
    st = thresholds.get("soft_thresholds", {}).get("can_bus_mlp", {"p995": math.inf})
    pred = int(strong >= 1 or hard >= 2 or soft > float(st.get("p995", math.inf)))
    return {"pred_label": pred, "alarm_count": hard, "strong_count": strong, "soft_score": float(soft), "details": details}


def derive_v11_input_alarm_levels(value_rep: Dict[str, Any], mlp_rep: Dict[str, Any]) -> Dict[str, Any]:
    """Derive harm-aware input alarm levels from the value-level detector report.

    raw_input_alarm keeps the v10-style "any can_bus structural anomaly" behavior.
    critical_input_alarm is intentionally narrower and should represent defects that
    are highly likely to be harmful by themselves. Ordinary structural anomalies are
    stored as suspicious_input_alarm and are only promoted to final harmful alarms
    after output-sanity evidence is added during post-eval.
    """
    groups = value_rep.get("groups", {}) or {}
    reasons: List[str] = []
    critical = 0

    def _group(name: str) -> Dict[str, Any]:
        return groups.get(name, {}) or {}

    def _detail_alarm(g: Dict[str, Any], feat: str, key: str = "alarm") -> int:
        try:
            return int(((g.get("details", {}) or {}).get(feat, {}) or {}).get(key, 0))
        except Exception:
            return 0

    def _detail_value(g: Dict[str, Any], feat: str) -> float:
        try:
            return float(((g.get("details", {}) or {}).get(feat, {}) or {}).get("value", 0.0))
        except Exception:
            return 0.0

    for gname, greport in groups.items():
        if _detail_alarm(greport, "finite_ratio", "alarm"):
            critical = 1
            reasons.append(f"{gname}:nonfinite")
        if int(greport.get("strong_count", 0)) >= 2:
            critical = 1
            reasons.append(f"{gname}:strong_ge2")

    small = _group("small_signed_state")
    small_struct = small.get("structure_detector", {}) or {}
    if int(small_struct.get("collapse_hit_count", 0)) >= 1:
        critical = 1
        reasons.append("small_signed_state:collapse")
    if int(small_struct.get("strong_count", 0)) >= 1:
        critical = 1
        reasons.append("small_signed_state:strong_structure")

    for gname in ("lp_9", "lp_13", "near_zero_slots"):
        g = _group(gname)
        if int(g.get("strong_count", 0)) >= 1 and int(g.get("alarm_count", 0)) >= 2:
            critical = 1
            reasons.append(f"{gname}:strong_plus_multi_alarm")

    rot = _group("rotation_quat")
    quat_err = _detail_value(rot, "quat_norm_error")
    if int(rot.get("strong_count", 0)) >= 2 or quat_err > 0.75:
        critical = 1
        reasons.append("rotation_quat:extreme")

    patch = _group("patch_angle_pair")
    angle_err = _detail_value(patch, "angle_consistency_error")
    if int(patch.get("strong_count", 0)) >= 2 or angle_err > 30.0:
        critical = 1
        reasons.append("patch_angle_pair:extreme")

    trans = _group("translation_xyz")
    tshape = trans.get("translation_shape_detector", {}) or {}
    if int(trans.get("strong_count", 0)) >= 2:
        critical = 1
        reasons.append("translation_xyz:strong_ge2")
    if int(tshape.get("strong_count", 0)) >= 1 and int(trans.get("alarm_count", 0)) >= 1:
        critical = 1
        reasons.append("translation_xyz:strong_shape_plus_value")

    raw_input_alarm = int(value_rep.get("pred_label", 0) == 1)
    mlp_alarm = int(mlp_rep.get("pred_label", 0) == 1)
    suspicious = int((raw_input_alarm == 1 or mlp_alarm == 1) and critical == 0)
    return {
        "raw_input_alarm": raw_input_alarm,
        "critical_input_alarm": int(critical),
        "suspicious_input_alarm": suspicious,
        "mlp_input_alarm": mlp_alarm,
        "critical_reasons": reasons,
    }


def detect_object_can_bus(features: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    value_rep = detect_object_can_bus_value(
        features["can_bus_groups"],
        features["can_bus_direct"],
        features.get("small_signed_structure", {}),
        features.get("translation_shape", {}),
        thresholds,
    )
    mlp_rep = detect_object_can_bus_mlp(features["can_bus_mlp"], thresholds)
    value_soft = float(value_rep.get("soft_score", 0.0))
    mlp_soft = float(mlp_rep.get("soft_score", 0.0))
    levels = derive_v11_input_alarm_levels(value_rep, mlp_rep)
    obj_pred = int(levels["raw_input_alarm"])
    return {
        "pred_label": obj_pred,
        "raw_input_alarm": int(levels["raw_input_alarm"]),
        "critical_input_alarm": int(levels["critical_input_alarm"]),
        "suspicious_input_alarm": int(levels["suspicious_input_alarm"]),
        "mlp_input_alarm": int(levels["mlp_input_alarm"]),
        "critical_reasons": levels.get("critical_reasons", []),
        "soft_score": float(value_soft),
        "value_soft_score": value_soft,
        "mlp_soft_score": mlp_soft,
        "value_detector": value_rep,
        "mlp_detector": mlp_rep,
    }


def detect_s0(features: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    rep_cb = detect_object_can_bus(features, thresholds)
    max_soft = float(rep_cb["soft_score"])
    raw_pred = int(rep_cb.get("raw_input_alarm", rep_cb.get("pred_label", 0)))
    critical_pred = int(rep_cb.get("critical_input_alarm", 0))
    suspicious_pred = int(rep_cb.get("suspicious_input_alarm", 0))
    return {
        "objects": {"can_bus": rep_cb},
        "combined": {
            "pred_label": raw_pred,
            "raw_pred_label": raw_pred,
            "critical_input_alarm": critical_pred,
            "suspicious_input_alarm": suspicious_pred,
            "pre_output_harm_pred_label": critical_pred,
            "alarm_object_count": int(raw_pred),
            "max_soft_score": float(max_soft),
        },
    }


def classification_metrics(records: List[Dict], key_path: str = "combined") -> Dict[str, Any]:
    return classification_metrics_impl(records, key_path=key_path)


def summarize_feature_distribution(records: List[Dict]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"can_bus_groups": {}, "can_bus_direct": {}, "can_bus_mlp": {}}
    for gname in CANBUS_GROUP_ORDER:
        summary["can_bus_groups"][gname] = {}
        for feat in canbus_group_feature_list(gname):
            vals = [r["features"]["can_bus_groups"][gname].get(feat) for r in records
                    if np.isfinite(r["features"]["can_bus_groups"][gname].get(feat, np.nan))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                summary["can_bus_groups"][gname][feat] = {
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "p50": float(np.quantile(arr, 0.50)),
                    "p95": float(np.quantile(arr, 0.95)),
                    "p99": float(np.quantile(arr, 0.99)),
                }
    for feat in CANBUS_DIRECT_FEATURES:
        vals = [r["features"]["can_bus_direct"].get(feat) for r in records
                if np.isfinite(r["features"]["can_bus_direct"].get(feat, np.nan))]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary["can_bus_direct"][feat] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "p50": float(np.quantile(arr, 0.50)),
                "p95": float(np.quantile(arr, 0.95)),
                "p99": float(np.quantile(arr, 0.99)),
            }
    for feat in CANBUS_MLP_FEATURES:
        vals = [r["features"]["can_bus_mlp"].get(feat) for r in records
                if np.isfinite(r["features"]["can_bus_mlp"].get(feat, np.nan))]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary["can_bus_mlp"][feat] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "p50": float(np.quantile(arr, 0.50)),
                "p95": float(np.quantile(arr, 0.95)),
                "p99": float(np.quantile(arr, 0.99)),
            }
    return summary


def build_feature_analysis_debug(records: List[Dict], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "thresholds": thresholds.get("feature_thresholds", {}),
        "soft_thresholds": thresholds.get("soft_thresholds", {}),
        "runtime_distribution": summarize_feature_distribution(records),
    }


def build_record(sample_id: str, raw_sample_id: str, is_positive: int, fault_infos: List[Dict[str, Any]],
                 features: Dict[str, Any], detector_report: Dict[str, Any], object_vectors: Dict[str, Any],
                 hook_summary: Optional[Dict[str, Any]] = None, stage_summaries: Optional[Dict[str, Any]] = None,
                 scene_context_summary: Optional[Dict[str, Any]] = None,
                 row_source: str = "",
                 isolation_info: Optional[Dict[str, Any]] = None,
                 online_context_features: Optional[Dict[str, Any]] = None,
                 v26_online_head_features: Optional[Dict[str, Any]] = None,
                 fault_spec_count: int = 0,
                 fault_attempted_count: int = 0,
                 fault_applied_count: int = 0,
                 fault_effective_count: int = 0) -> Dict[str, Any]:
    effective_positive = int(any(int(fi.get("effective", 0)) == 1 for fi in fault_infos))
    return {
        "sample_id": sample_id,
        "raw_sample_id": raw_sample_id,
        "row_source": row_source or ("experiment_injected" if int(is_positive) else "calibration_or_no_fault"),
        "is_positive": int(is_positive),
        "is_fault_selected": int(is_positive),
        "is_positive_effective": effective_positive,
        "fault_spec_count": int(fault_spec_count),
        "fault_attempted_count": int(fault_attempted_count),
        "fault_applied_count": int(fault_applied_count),
        "fault_effective_count": int(fault_effective_count),
        "faults": fault_infos,
        "features": features,
        "detector_report": detector_report,
        "object_vectors": object_vectors,
        "hook_summary": hook_summary or {},
        "stage_summaries": stage_summaries or {},
        "scene_context_summary": scene_context_summary or {},
        "isolation_info": isolation_info or {},
        "online_context_features": online_context_features or {},
        "v26_online_head_features": v26_online_head_features or {},
    }


# -----------------------------------------------------------------------------
# v53 sample-inline online detector runtime
# -----------------------------------------------------------------------------

def v53_exact_layer_from_fault_target(target: str) -> str:
    target = str(target or "")
    if target == "" or target.lower() in {"none", "nan"}:
        return "no_fault"
    if target == "can_bus" or target.startswith("can_bus"):
        return "S0_canbus"
    if target.startswith("c:"):
        return "C_projector"
    if target.startswith("d:"):
        logical = target.split(":", 1)[1]
        if logical in {"d_lm_head", "d_model_norm"} or logical.startswith("dout_"):
            return "D_out"
        return "D_LLM"
    if target.startswith("s1:"):
        logical = target.split(":", 1)[1]
        for prefix in ["S1_", "S2_", "S3_", "S4_", "S5_"]:
            if logical.startswith(prefix):
                return prefix[:-1]
        return "S_full_UNCLASSIFIED_EXACT_PREFIX"
    return "UNCLASSIFIED_EXACT_PREFIX"


def v53_first_fault_target(faults: Any) -> str:
    faults = faults or []
    if isinstance(faults, list) and len(faults) >= 1:
        return str((faults[0] or {}).get("target", ""))
    return ""


def v53_load_selected_detector_artifacts(artifact: Path, detectors_csv: str = "") -> Dict[str, Dict[str, Any]]:
    """Load v35.1 selected detector artifacts once for sample-inline scoring.

    This is exact by feature column name. It never fuzzy-matches or renames features.
    Missing exact deployable feature values are fatal in v53 strict mode.
    No fuzzy matching, column guessing, or implicit missing-column filling is allowed.
    """
    if not artifact.is_file():
        raise FileNotFoundError(f"missing v35.1 detector artifact: {artifact}")
    wanted = {x.strip() for x in str(detectors_csv or "").split(",") if x.strip()}
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    with tarfile.open(artifact, "r:gz") as tf:
        tf.extractall(root)
    out: Dict[str, Dict[str, Any]] = {}
    for feat_path in sorted(root.rglob("feature_columns.json")):
        det_dir = feat_path.parent
        det_name = det_dir.name
        if wanted and det_name not in wanted:
            continue
        model_path = det_dir / "model.joblib"
        threshold_path = det_dir / "threshold.json"
        card_path = det_dir / "model_card.json"
        if not model_path.is_file():
            continue
        feature_columns = json.loads(feat_path.read_text(encoding="utf-8"))
        threshold = 0.5
        if threshold_path.is_file():
            threshold = float(json.loads(threshold_path.read_text(encoding="utf-8")).get("threshold", threshold))
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}
        out[det_name] = {
            "detector": det_name,
            "feature_columns": [str(c) for c in feature_columns],
            "model": joblib.load(model_path),
            "threshold": float(threshold),
            "clip_abs": float(card.get("clip_abs", 1_000_000.0)),
            "label_mode": card.get("label_mode", ""),
            "feature_set": card.get("feature_set", ""),
            "model_card": card,
            "_tmp": tmp,
        }
    if not out:
        raise RuntimeError(f"no selected detector artifacts loaded from {artifact}; wanted={sorted(wanted)}")
    return out


def v53_score_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if getattr(p, "ndim", 1) == 2 and p.shape[1] >= 2:
            return np.asarray(p[:, 1], dtype=np.float64)
        return np.asarray(p, dtype=np.float64).reshape(-1)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


class V53InlinePlanningMetricRuntime:
    """Per-sample GT metric helper for experiment analysis only.

    These L2/collision fields are NOT detector inputs. They are emitted so the
    experiment can compute metrics and error breakdown after inference without
    rerunning prediction parsing.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.available = False
        self.reason = "disabled"
        self.metric = None
        self.gt_trajs_dict = {}
        self.gt_trajs_mask_dict = {}
        self.gt_occ_map = {}
        if not self.enabled:
            return
        try:
            from eval_share.metric import PlanningMetric
            import eval_share.evaluation as evaluation_mod
            current_dir = Path(evaluation_mod.__file__).resolve().parent
            with open(current_dir / "gt/planing_gt_segmentation_val", "rb") as f:
                self.gt_occ_map = pickle.load(f)
            for token in list(self.gt_occ_map.keys()):
                if not isinstance(self.gt_occ_map[token], torch.Tensor):
                    self.gt_occ_map[token] = torch.tensor(self.gt_occ_map[token])
                self.gt_occ_map[token] = torch.flip(self.gt_occ_map[token], [-1])
            with open(current_dir / "gt/gt_traj.pkl", "rb") as f:
                self.gt_trajs_dict = pickle.load(f)
            with open(current_dir / "gt/gt_traj_mask.pkl", "rb") as f:
                self.gt_trajs_mask_dict = pickle.load(f)
            self.metric = PlanningMetric(6).to(torch.device("cpu"))
            self.available = True
            self.reason = str(current_dir)
        except Exception as e:
            self.available = False
            self.reason = f"unavailable:{type(e).__name__}:{e}"

    def compute(self, token: str, pred: Any) -> Dict[str, Any]:
        if not self.enabled:
            return {"available": False, "reason": "disabled"}
        if not self.available:
            return {"available": False, "reason": self.reason}
        token = normalize_sample_token(str(token or ""))
        if token not in self.gt_trajs_dict or token not in self.gt_trajs_mask_dict or token not in self.gt_occ_map:
            return {"available": False, "reason": "missing_gt_or_occ", "sample_id": token}
        try:
            gt_traj = torch.tensor(self.gt_trajs_dict[token]).to(torch.device("cpu"))
            gt_mask = torch.tensor(self.gt_trajs_mask_dict[token]).to(torch.device("cpu"))
            occ = self.gt_occ_map[token].to(torch.device("cpu"))
            out = torch.tensor(pred).reshape(gt_mask.shape).to(torch.device("cpu"))
            if out.shape[1] % 2:
                out = out[:, 1:]
            if occ.shape[1] % 2:
                occ = occ[:, 1:]
            if gt_traj.shape[1] % 2:
                gt_traj = gt_traj[:, 1:]
            if gt_mask.shape[1] % 2:
                gt_mask = gt_mask[:, 1:]
            out = out[:, :6]
            gt_traj = gt_traj[:, :6]
            gt_mask = gt_mask[:, :6]
            occ = occ[:, :6]
            l2 = self.metric.compute_L2(out, gt_traj, gt_mask)[0].tolist()
            _, obj_box = self.metric.evaluate_coll(out[:, :, :2], gt_traj[:, :, :2], occ, token=None)
            obj_box = [float(x) for x in obj_box.tolist()]
            return {
                "available": True,
                "sample_id": token,
                "l2_t": l2,
                "collision_t": obj_box,
                "l2_1s": float(sum(l2[:2]) / 2.0),
                "l2_2s": float(sum(l2[:4]) / 4.0),
                "l2_3s": float(sum(l2[:6]) / 6.0),
                "l2_avg": float((sum(l2[:2]) / 2.0 + sum(l2[:4]) / 4.0 + sum(l2[:6]) / 6.0) / 3.0),
                "collision_1s": float(sum(obj_box[:2]) / 2.0 * 100.0),
                "collision_2s": float(sum(obj_box[:4]) / 4.0 * 100.0),
                "collision_3s": float(sum(obj_box[:6]) / 6.0 * 100.0),
                "collision_avg": float(((sum(obj_box[:2]) / 2.0) + (sum(obj_box[:4]) / 4.0) + (sum(obj_box[:6]) / 6.0)) / 3.0 * 100.0),
            }
        except Exception as e:
            return {"available": False, "reason": f"metric_error:{type(e).__name__}:{e}", "sample_id": token}



class V53InlineDetectorRuntime:
    """Detector runs immediately after each sample using the same standard row-record
    construction path that feeds stage_feature_matrix_v26.

    v53 changes from the earlier v53 implementation:
      * it does NOT build a partial ad-hoc feature row;
      * it builds the experiment row_record with the same online-safe fields used by
        summarize_harm(...)->write_training_feature_matrix(...);
      * it loads clean_ref only to build analysis-only harmful labels/deltas; those
        values are saved for post-processing but are not detector input features;
      * it maintains previous-row temporal context online, then post-run parity must
        verify inline scores against the written stage matrix.
    """

    def __init__(self, artifact: Path, detectors_csv: str = "", analysis_metrics: bool = True,
                 clean_ref: Optional[Path] = None, output_sanity_json: Optional[Path] = None):
        self.artifact = Path(artifact)
        self.detectors = v53_load_selected_detector_artifacts(self.artifact, detectors_csv)
        self.metric_runtime = V53InlinePlanningMetricRuntime(enabled=analysis_metrics)
        self.version = "v53_sample_inline_online_v35_1_detector_runtime"
        self.clean_ref = Path(clean_ref) if clean_ref else None
        self.output_sanity_json = Path(output_sanity_json) if output_sanity_json else None
        self.clean_metrics: Dict[str, Dict[str, Any]] = {}
        self.clean_ref_available = False
        self.clean_ref_reason = "not_configured"
        if self.clean_ref is not None:
            try:
                if self.clean_ref.is_file():
                    self.clean_metrics = {normalize_sample_token(str(r.get("sample_id", ""))): r for r in load_jsonl(self.clean_ref) if r.get("sample_id")}
                    self.clean_ref_available = True
                    self.clean_ref_reason = str(self.clean_ref)
                else:
                    self.clean_ref_reason = f"missing:{self.clean_ref}"
            except Exception as e:
                self.clean_ref_available = False
                self.clean_ref_reason = f"error:{type(e).__name__}:{e}"
        self.output_sanity_thresholds = None
        self.output_sanity_thresholds_reason = "not_configured"
        if self.output_sanity_json is not None:
            try:
                if self.output_sanity_json.is_file():
                    self.output_sanity_thresholds = _load_json(self.output_sanity_json)
                    self.output_sanity_thresholds_reason = str(self.output_sanity_json)
                else:
                    self.output_sanity_thresholds_reason = f"missing:{self.output_sanity_json}"
            except Exception as e:
                self.output_sanity_thresholds = None
                self.output_sanity_thresholds_reason = f"error:{type(e).__name__}:{e}"
        self.inline_row_records: List[Dict[str, Any]] = []

    @staticmethod
    def _sf(x: Any, default: float = math.nan) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    def _build_fault_metric_row(self, sample_id: str, output_sanity: Dict[str, Any], e_parse: Dict[str, Any], planning_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Build the per-sample metric row used by summarize_harm, without requiring
        full batch rebuild. Deltas/labels are analysis-only and require clean_ref.
        """
        sid = normalize_sample_token(sample_id)
        f = dict(planning_metrics or {})
        c = self.clean_metrics.get(sid)
        out: Dict[str, Any] = {
            "sample_id": sid,
            "output_sanity_features": output_sanity or {},
            "e_parse_features": e_parse or {},
            "clean_ref_available": int(c is not None),
            "gt_metric_available": int(bool(f.get("available", False))),
        }
        # Fault/GT metrics are online available in experiment mode because this is an evaluation run.
        for k in ["l2_1s", "l2_2s", "l2_3s", "l2_avg", "collision_1s", "collision_2s", "collision_3s", "collision_avg"]:
            out[f"fault_{k}"] = self._sf(f.get(k))
            out[f"clean_{k}"] = self._sf((c or {}).get(k)) if c is not None else math.nan
        # Match summarize_harm delta definitions when clean is present.
        for suffix in ["1s", "2s", "3s", "avg"]:
            fk = f"l2_{suffix}"
            ck = f"collision_{suffix}"
            fv, cv = self._sf(f.get(fk)), self._sf((c or {}).get(fk)) if c is not None else math.nan
            out[f"delta_l2_{suffix}"] = float(fv - cv) if np.isfinite(fv) and np.isfinite(cv) else math.nan
            fv, cv = self._sf(f.get(ck)), self._sf((c or {}).get(ck)) if c is not None else math.nan
            out[f"delta_collision_{suffix}"] = float(fv - cv) if np.isfinite(fv) and np.isfinite(cv) else math.nan
        return out

    def _prepare_runtime_row(self, record: Dict[str, Any], result: Dict[str, Any], sample_id: str, sample_id_full: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        plan_row = dict(result or {})
        plan_row.setdefault("id", sample_id_full)
        ans = plan_row.get("answer", "")
        forced_traj, forced_meta = v48_plan_conv_forced_traj(plan_row)
        if forced_traj is not None:
            traj = forced_traj
            _, answer_parse_meta = robust_retrieve_traj_with_meta(ans)
            e_parse = dict(answer_parse_meta)
            e_parse.update(forced_meta)
        else:
            traj, e_parse = robust_retrieve_traj_with_meta(ans)

        output_sanity = trajectory_sanity_features_from_pred(traj)
        planning_metrics = self.metric_runtime.compute(sample_id, traj)
        metric_row = self._build_fault_metric_row(sample_id, output_sanity, e_parse, planning_metrics)
        output_rep = detect_output_sanity(metric_row, self.output_sanity_thresholds)
        metric_row["output_sanity_report"] = output_rep

        injected = int(bool(record.get("faults")) and int(record.get("is_positive_effective", record.get("is_positive", 0)) or 0) == 1)
        raw_pred = _raw_input_pred_from_record(record)
        critical_pred = _critical_input_pred_from_record(record)
        suspicious_pred = _suspicious_input_pred_from_record(record)
        output_pred = _output_alarm_from_row(metric_row)
        narrow_exception = _v19_patch_strong_exception_from_record(record)
        v11_p0_two_stage = int(critical_pred == 1 or (suspicious_pred == 1 and output_pred == 1))
        v19_output_first = int(output_pred == 1)
        # v53: no harmful labels are constructed for detector input.
        # Labels such as harmful/severe_harmful are posthoc evaluation labels only.
        g = _fault_group_from_record(record)
        target = v53_first_fault_target(record.get("faults", []))

        rr: Dict[str, Any] = {
            **{k: v for k, v in metric_row.items() if k != "output_sanity_report"},
            "sample_id": normalize_sample_token(sample_id),
            "row_source": "experiment_injected" if injected else "experiment_no_fault",
            "is_clean": 0,
            "is_experiment_row": 1,
            "is_fault_selected": int(record.get("is_fault_selected", record.get("is_positive", 0)) or 0),
            "fault_spec_count": int(record.get("fault_spec_count", 0) or 0),
            "fault_attempted_count": int(record.get("fault_attempted_count", 0) or 0),
            "fault_applied_count": int(record.get("fault_applied_count", 0) or 0),
            "fault_effective_count": int(record.get("fault_effective_count", 0) or 0),
            "is_effective_injected": injected,
            "injected_effective": injected,
            # Keep the historical field, but exact target is stored separately in analysis_only.
            "fault_target": target if target else "none",
            "fault_group": g if injected else "experiment_no_fault",
            "raw_input_alarm": raw_pred,
            "critical_input_alarm": critical_pred,
            "suspicious_input_alarm": suspicious_pred,
            "e_output_sanity_alarm": output_pred,
            "v11_p0_two_stage_harm_alarm": v11_p0_two_stage,
            "v19_patch_strong_exception": narrow_exception,
            "v19_output_first_harm_alarm": v19_output_first,
            "critical_reasons": _critical_reasons_from_record(record),
            "output_sanity_report": output_rep,
            "e_parse_features": e_parse,
            "object_vectors": record.get("object_vectors", {}),
            "faults": record.get("faults", []),
            "stage_summaries": record.get("stage_summaries", {}),
            "scene_context_summary": record.get("scene_context_summary", {}),
            "online_context_features": record.get("online_context_features", {}),
            "v26_online_head_features": record.get("v26_online_head_features", {}),
            "v26_agent_features": metric_row.get("v26_agent_features", {}) or compute_v26_agent_context_features(sample_id, traj),
            "v26_ego_pose_features": metric_row.get("v26_ego_pose_features", {}) or compute_v26_ego_pose_context_features(sample_id),
            "v35_behavior_features": v35_behavior_features_from_plan_row(plan_row),
            "stage_feature_flat": flatten_stage_summary_for_training(record.get("stage_summaries", {})),
            "s0_feature_flat": _flatten_numeric_dict(record.get("features", {})),
        }
        rr["row_index"] = len(self.inline_row_records)
        self.inline_row_records.append(rr)
        add_v19_temporal_context_features(self.inline_row_records)
        # Use the mutated latest row after temporal context has been added.
        rr = self.inline_row_records[-1]
        analysis = {
            "parsed_traj": traj,
            "e_parse_features": e_parse,
            "output_sanity_features": output_sanity,
            "output_sanity_report": output_rep,
            "planning_metrics": planning_metrics,
            "clean_ref_available": int(normalize_sample_token(sample_id) in self.clean_metrics),
            "posthoc_label_policy": "labels_are_built_offline_only_not_detector_features",
            "delta_metrics": {k: metric_row.get(k) for k in metric_row.keys() if k.startswith("delta_") or k.startswith("clean_") or k.startswith("fault_")},
        }
        return rr, analysis

    @staticmethod
    def _family_counts(names: List[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for n in names:
            parts = str(n).split("__")
            fam = "__".join(parts[:2]) if len(parts) >= 2 else str(n).split(".")[0]
            out[fam] = out.get(fam, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def score_sample(self, record: Dict[str, Any], result: Dict[str, Any], sample_id: str, sample_id_full: str) -> Dict[str, Any]:
        rr, analysis = self._prepare_runtime_row(record, result, sample_id, sample_id_full)
        flat = v53_deployable_feature_row_from_record(rr)
        detector_outputs: Dict[str, Any] = {}
        any_alarm = 0
        for det_name, info in sorted(self.detectors.items()):
            cols = list(info["feature_columns"])
            vals: List[float] = []
            present = 0
            nonfinite = 0
            absent_value_names: List[str] = []
            for col in cols:
                if col in flat:
                    present += 1
                    raw = flat.get(col)
                else:
                    # Historical v35.1 matrices were built from a union of sparse
                    # per-row keys.  A specific sample may not emit a dynamic key
                    # such as e_report.details.traj_all_zero when the condition is
                    # false, or a temporal prev field on the first processed row.
                    # We still construct the exact schema column explicitly, with
                    # the same numeric sanitization path used for matrix export.
                    # This is not fuzzy matching and not a new feature: it is an
                    # exact reindex to feature_columns.json.
                    absent_value_names.append(col)
                    raw = math.nan
                try:
                    v = float(raw)
                except Exception:
                    v = math.nan
                if not math.isfinite(v):
                    nonfinite += 1
                    v = 0.0
                clip_abs = float(info.get("clip_abs", 1_000_000.0))
                if clip_abs and clip_abs > 0:
                    if v > clip_abs:
                        v = clip_abs
                    elif v < -clip_abs:
                        v = -clip_abs
                vals.append(v)
            X = pd.DataFrame([vals], columns=cols)
            score = float(v53_score_model(info["model"], X)[0])
            threshold = float(info["threshold"])
            alarm = int(score >= threshold)
            any_alarm = int(any_alarm or alarm)
            detector_outputs[det_name] = {
                "score": score,
                "threshold": threshold,
                "alarm": alarm,
                "feature_set": info.get("feature_set", ""),
                "label_mode": info.get("label_mode", ""),
                "feature_count": int(len(cols)),
                "row_feature_present_count": int(present),
                "row_feature_missing_count": 0,
                "row_feature_missing_filled_count": 0,
                "row_feature_absent_value_count": int(len(absent_value_names)),
                "row_feature_nonfinite_sanitized_count": int(nonfinite),
                "row_feature_absent_value_names_head": absent_value_names[:50],
                "row_feature_absent_value_family_counts": self._family_counts(absent_value_names),
            }
        target = v53_first_fault_target(record.get("faults", []))
        return {
            "version": self.version,
            "sample_id": normalize_sample_token(sample_id),
            "raw_sample_id": sample_id_full,
            "row_source": rr.get("row_source", record.get("row_source", "")),
            "detector_timing": "sample_inline_after_current_forward_before_next_sample",
            "detector_input_policy": "v53_deployable_feature_row_from_inference_process_only",
            "final_alarm_any_selected_detector": int(any_alarm),
            "detectors": detector_outputs,
            "analysis_only": {
                "is_positive": int(record.get("is_positive", 0) or 0),
                "is_positive_effective": int(record.get("is_positive_effective", 0) or 0),
                "fault_scope_legacy": record.get("fault_scope", ""),
                "fault_target_exact": target,
                "exact_layer": v53_exact_layer_from_fault_target(target),
                "faults": record.get("faults", []),
                "posthoc_label_policy": analysis.get("posthoc_label_policy", "labels_are_built_offline_only_not_detector_features"),
                "delta_metrics": analysis.get("delta_metrics", {}),
                "posthoc_labels": v53_analysis_label_row_from_deltas(
                    normalize_sample_token(sample_id),
                    rr.get("row_source", record.get("row_source", "")),
                    int(record.get("is_positive_effective", 0) or 0),
                    analysis.get("delta_metrics", {}) or {},
                ),
                "planning_metrics": analysis.get("planning_metrics", {}),
                "clean_ref_available": analysis.get("clean_ref_available", 0),
                "gt_metric_available": int(bool((analysis.get("planning_metrics", {}) or {}).get("available", False))),
                "e_parse_features": analysis.get("e_parse_features", {}),
                "output_sanity_features": analysis.get("output_sanity_features", {}),
                "output_sanity_report": analysis.get("output_sanity_report", {}),
            },
            "deployable_feature_row": flat,
        }

def v53_inline_summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"version": "v53_inline_online_detector_summary", "rows": len(records), "detectors": {}}
    if not records:
        return out
    names = sorted({k for r in records for k in (r.get("detectors", {}) or {}).keys()})
    for name in names:
        alarms = [int((r.get("detectors", {}) or {}).get(name, {}).get("alarm", 0) or 0) for r in records]
        scores = [float((r.get("detectors", {}) or {}).get(name, {}).get("score", 0.0) or 0.0) for r in records]
        out["detectors"][name] = {
            "rows": len(alarms),
            "alarm_count": int(sum(alarms)),
            "alarm_rate": float(sum(alarms) / max(len(alarms), 1)),
            "score_mean": float(np.mean(scores)) if scores else 0.0,
            "score_max": float(np.max(scores)) if scores else 0.0,
        }
    layers: Dict[str, int] = {}
    for r in records:
        layer = str(((r.get("analysis_only", {}) or {}).get("exact_layer", "unknown")))
        layers[layer] = layers.get(layer, 0) + 1
    out["analysis_only_exact_layer_counts"] = layers
    return out


# -----------------------------------------------------------------------------
# Inference scaffold
# -----------------------------------------------------------------------------

def inference_planning_oriented_vlm(args):
    local_rank = int(args.local_rank)
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if local_rank != -1 else 1
    rank = int(os.environ.get("RANK", "0")) if local_rank != -1 else 0

    if local_rank != -1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    device = f"cuda:{local_rank}" if local_rank != -1 else str(args.device)
    if device == "cuda":
        device = "cuda:0"
    torch.cuda.set_device(torch.device(device) if str(device).startswith("cuda") else device)

    tokenizer, model_engine, image_processor, _ = load_model_with_deepspeed(args, device)

    can_bus_capture, can_bus_hook_handle, hook_name = register_can_bus_mlp_hook(model_engine, args.can_bus_mlp_hook_suffix)
    stage_captures, stage_hook_handles, stage_hook_status = register_stage_summary_hooks(
        model_engine, args.stage_hook_specs, strict=args.stage_hook_strict
    )
    v26_online_head_capture, v26_online_head_handle, v26_online_head_hook_status = register_v26_online_head_capture(model_engine)

    v32_s1_runtime = None
    v32_s1_hook_handles: List[Any] = []
    v32_s1_hook_status: Dict[str, Any] = {"registered_count": 0, "missing_count": 0, "registered": {}, "missing": {}}
    current_allowed_targets = S0_TARGET_MODE_TO_TARGETS.get(str(getattr(args, "target_mode", "")), [])
    if "s1_activation" in current_allowed_targets or bool(getattr(args, "enable_s1_hooks", False)):
        v32_s1_runtime, v32_s1_hook_handles, v32_s1_hook_status = register_v32_s1_fault_hooks(model_engine, args.s1_target_specs)
        if "s1_activation" in current_allowed_targets and int(v32_s1_hook_status.get("registered_count", 0)) <= 0:
            raise RuntimeError("target_mode requires at least one registered S1 hook")

    v44_c_runtime = None
    v44_d_runtime = None
    v44_c_hook_handles: List[Any] = []
    v44_d_hook_handles: List[Any] = []
    v44_c_hook_status: Dict[str, Any] = {"registered_count": 0, "missing_count": 0, "registered": {}, "missing": {}}
    v44_d_hook_status: Dict[str, Any] = {"registered_count": 0, "missing_count": 0, "registered": {}, "missing": {}}
    if "c_activation" in current_allowed_targets or bool(getattr(args, "enable_cde_hooks", False)):
        v44_c_runtime, v44_c_hook_handles, v44_c_hook_status = register_v44_activation_fault_hooks(model_engine, args.c_target_specs, prefix="c")
        if "c_activation" in current_allowed_targets and int(v44_c_hook_status.get("registered_count", 0)) <= 0:
            raise RuntimeError("target_mode requires at least one registered C hook")
    if "d_activation" in current_allowed_targets or bool(getattr(args, "enable_cde_hooks", False)):
        v44_d_runtime, v44_d_hook_handles, v44_d_hook_status = register_v44_activation_fault_hooks(model_engine, args.d_target_specs, prefix="d")
        if "d_activation" in current_allowed_targets and int(v44_d_hook_status.get("registered_count", 0)) <= 0:
            raise RuntimeError("target_mode requires at least one registered D hook")

    if args.mode == "experiment":
        if args.calibration_json is None:
            raise ValueError("experiment mode requires --calibration-json")
        with open(args.calibration_json, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        dim_stats = thresholds.get("dim_stats", {})
    else:
        thresholds = None
        dim_stats = None

    data_args = DataArguments(data_path=args.data, lazy_preprocess=True, frames_upbound=32)

    uniad_cfg = Config.fromfile(os.path.join(args.model_path, "llava_arch.py")) if os.path.exists(os.path.join(args.model_path, "llava_arch.py")) else None
    if uniad_cfg is None:
        cfg_paths = [
            os.path.join(PROJECT_ROOT, "projects/configs/stage1_track_map/base_track_map.py"),
            os.path.join(PROJECT_ROOT, "projects/configs/stage1_track_map/base_track_map.py"),
        ]
        for cp in cfg_paths:
            if os.path.exists(cp):
                uniad_cfg = Config.fromfile(cp)
                break
    if uniad_cfg is None:
        raise RuntimeError("Could not locate UniAD config")
    test_dataset = LLaVANuScenesDataset(tokenizer, data_args, uniad_cfg.data.test, llava_test_mode=True, use_uniad_pth=args.use_uniad_pth)
    full_dataset_size = len(test_dataset)

    subset_arg = str(args.subset_ids or "").strip()
    if subset_arg.upper() in {"FULL", "__FULL__", "ALL", "__ALL__"}:
        # v31 full-dataset mode: keep the entire evaluation dataset and build
        # the fault-plan population from the dataset's own stable sample ids.
        subset_ids_list = []
        seen_full_ids = set()
        for idx, entry in enumerate(test_dataset.list_data_dict):
            info = test_dataset.data_infos[idx] if idx < len(test_dataset.data_infos) else None
            candidate_ids = []
            candidate_ids.extend(_collect_entry_candidate_ids(entry, fallback_idx=idx, max_depth=3))
            if info is not None:
                candidate_ids.extend(_collect_entry_candidate_ids(info, fallback_idx=None, max_depth=3))
            sid = None
            for cid in candidate_ids:
                scid = str(cid)
                if scid and scid not in seen_full_ids:
                    sid = scid
                    break
            if sid is None:
                sid = str(idx)
            subset_ids_list.append(sid)
            seen_full_ids.add(sid)
        if rank == 0:
            print(f"[subset] FULL mode requested by --subset-ids {args.subset_ids}")
            print(f"[subset] original dataset size: {full_dataset_size}")
            print(f"[subset] kept full dataset size: {len(test_dataset.list_data_dict)}")
            print(f"[subset] generated unique fault-plan ids: {len(seen_full_ids)}")
    else:
        test_dataset, subset_ids_list = apply_subset_inplace(test_dataset, args.subset_ids, rank=rank)

    data_collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    sampler = ContinuousSceneDistributedSampler(test_dataset, shuffle=False) if local_rank != -1 else None

    fault_plan = None
    if args.mode == "experiment":
        allowed_targets = S0_TARGET_MODE_TO_TARGETS[args.target_mode]
        s1_target_weights = build_s1_target_weights(
            random.Random(int(args.experiment_seed)),
            args.s1_target_policy,
            args.s1_primary_target,
            int(args.s1_primary_weight),
            args.s1_broad_target_weights,
        )
        c_target_weights = parse_v44_named_weights(args.c_target_weights, DEFAULT_V44_C_TARGET_WEIGHTS)
        d_target_weights = parse_v44_named_weights(args.d_target_weights, DEFAULT_V44_D_TARGET_WEIGHTS)
        e_target_weights = parse_v44_named_weights(args.e_target_weights, DEFAULT_V44_E_TARGET_WEIGHTS)
        fault_plan = BlindFaultPlan(
            subset_ids=subset_ids_list,
            allowed_targets=allowed_targets,
            experiment_seed=args.experiment_seed,
            positive_min=args.positive_min,
            positive_max=args.positive_max,
            cardinality_weights=parse_weighted_map(args.cardinality_weights, cast_key=int, cast_val=int),
            bit_op_weights=parse_weighted_map(args.bit_op_weights, cast_key=str, cast_val=int),
            scope_weights=parse_weighted_map(args.mixed_fault_scope_weights, cast_key=str, cast_val=int),
            event_count_weights=parse_weighted_map(args.fault_event_count_weights, cast_key=int, cast_val=int),
            s1_target_weights=s1_target_weights,
            c_target_weights=c_target_weights,
            d_target_weights=d_target_weights,
            e_target_weights=e_target_weights,
            mixed_enabled=any(t in allowed_targets for t in ["s1_activation", "c_activation", "d_activation", "e_output"]),
            strict_single_fault=bool(getattr(args, "strict_single_fault", False)),
        )

    results_dir = os.path.join(os.path.dirname(args.output), "results")
    os.makedirs(results_dir, exist_ok=True)
    if rank == 0 and args.mode == "experiment":
        write_json(os.path.join(results_dir, "fault_plan.json"), fault_plan.summary())

    rank_output = args.output.replace(".json", f"_rank{rank}.json")
    if os.path.exists(rank_output):
        os.remove(rank_output)

    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        sampler=sampler,
        shuffle=False,
    )

    if rank == 0:
        print(f"Full dataset size: {full_dataset_size}")
        print(f"Effective dataset size: {len(test_dataset)}")
        print(f"Number of GPUs (world_size): {world_size}")
        print(f"Approximate samples per GPU: {max(1, len(test_dataset) // world_size)}")
        print(f"Output directory: {os.path.dirname(args.output)}")
        print(f"[{VERSION}] mode={args.mode}")
        print(f"[{VERSION}] target_mode={args.target_mode}")
        print(f"[{VERSION}] allowed_targets={S0_TARGET_MODE_TO_TARGETS.get(args.target_mode, [])}")
        print(f"[{VERSION}] detector_objects={DETECTOR_OBJECTS}")
        print(f"[{VERSION}] can_bus_mlp_hook={hook_name}")
        print(f"[{VERSION}] stage_hook_status={stage_hook_status}")
        print(f"[{VERSION}] v32_s1_hook_status={v32_s1_hook_status}")
        print(f"[{VERSION}] v44_c_hook_status={v44_c_hook_status}")
        print(f"[{VERSION}] v44_d_hook_status={v44_d_hook_status}")
        print(f"[{VERSION}] can_bus_groups={CANBUS_GROUPS}")

    tqdm_bar = tqdm(dataloader, ncols=100, disable=rank != 0)
    if sampler is not None:
        sampler.set_epoch(0)

    local_records: List[Dict[str, Any]] = []
    previous_sample_state = {
        "sample_id": "",
        "was_fault_selected": 0,
        "fault_count": 0,
        "effective_fault_count": 0,
    }

    v53_inline_runtime = None
    v53_inline_rank_records: List[Dict[str, Any]] = []
    v53_inline_rank_path = os.path.join(results_dir, f"v53_inline_detector_records_rank{rank}.jsonl")
    if os.path.exists(v53_inline_rank_path):
        os.remove(v53_inline_rank_path)
    if bool(getattr(args, "v53_inline_detector", False)):
        v53_inline_runtime = V53InlineDetectorRuntime(
            Path(getattr(args, "v53_detector_artifact", "")),
            detectors_csv=str(getattr(args, "v53_inline_detectors", "")),
            analysis_metrics=bool(getattr(args, "v53_inline_analysis_metrics", True)),
            clean_ref=Path(getattr(args, "clean_ref", "")) if str(getattr(args, "clean_ref", "")) else None,
            output_sanity_json=Path(getattr(args, "output_sanity_json", "")) if str(getattr(args, "output_sanity_json", "")) else None,
        )
        if rank == 0:
            print(f"[{VERSION}] v53 sample-inline detector enabled; artifact={getattr(args, 'v53_detector_artifact', '')}; detectors={list(v53_inline_runtime.detectors.keys())}")
            print(f"[{VERSION}] v53 analysis metric runtime: available={v53_inline_runtime.metric_runtime.available}; reason={v53_inline_runtime.metric_runtime.reason}")

    for data in tqdm_bar:
        clear_cuda_between_samples(bool(getattr(args, "clear_cuda_cache_between_samples", False)))

        model_temporal_reset_before = disabled_model_temporal_runtime_state_status()
        if temporal_state_reset_enabled(args):
            model_temporal_reset_before = reset_model_temporal_runtime_state(model_engine)

        # v22 isolation: clone the current batch before any possible injection.
        # This prevents in-place can_bus/data mutation from leaking into later samples.
        if bool(getattr(args, "sample_isolation", True)):
            data = clone_for_sample_isolation(data)
        data = move_data_to_device(data, device)
        if bool(getattr(args, "sample_isolation", True)):
            data = clone_for_sample_isolation(data)

        ensure_batch_size_one(data)
        sample_id_full = str(unwrap_batch_string(data["id"]))
        sample_id = normalize_sample_token(sample_id_full)

        reset_before = reset_runtime_capture_state(can_bus_capture, stage_captures)
        try:
            v26_online_head_capture.reset()
            reset_before["v26_online_head_capture_reset"] = 1
        except Exception:
            reset_before["v26_online_head_capture_reset"] = 0
        original_can_bus_snapshot = snapshot_can_bus_numpy(data)
        isolation_info: Dict[str, Any] = {
            "sample_isolation_enabled": int(bool(getattr(args, "sample_isolation", True))),
            "clear_cuda_cache_between_samples": int(bool(getattr(args, "clear_cuda_cache_between_samples", False))),
            "pre_forward_capture_reset": reset_before,
            "can_bus_snapshot_available": int(original_can_bus_snapshot is not None),
            "previous_sample_id": previous_sample_state.get("sample_id", ""),
            "previous_was_fault_selected": int(previous_sample_state.get("was_fault_selected", 0) or 0),
            "previous_fault_count": int(previous_sample_state.get("fault_count", 0) or 0),
            "previous_effective_fault_count": int(previous_sample_state.get("effective_fault_count", 0) or 0),
            "reset_model_state_between_samples": int(temporal_state_reset_enabled(args)),
            "model_temporal_reset_before_forward": model_temporal_reset_before,
        }

        fault_infos: List[Dict[str, Any]] = []
        specs: List[FaultEventSpec] = []
        s1_specs: List[FaultEventSpec] = []
        c_specs: List[FaultEventSpec] = []
        d_specs: List[FaultEventSpec] = []
        e_specs: List[FaultEventSpec] = []
        is_positive = 0
        fault_scope = "no_fault"
        if args.mode == "experiment":
            is_positive, specs = fault_plan.get_specs(sample_id)
            fault_scope = fault_plan.get_scope(sample_id) if hasattr(fault_plan, "get_scope") else ("S0_only" if int(is_positive) else "no_fault")
            for spec in specs:
                if spec.target == "can_bus":
                    fault_infos.append(apply_one_fault(data, sample_id, spec, args.experiment_seed))
                elif str(spec.target).startswith("s1:"):
                    s1_specs.append(spec)
                elif str(spec.target).startswith("c:"):
                    c_specs.append(spec)
                elif str(spec.target).startswith("d:"):
                    d_specs.append(spec)
                elif str(spec.target).startswith("e:"):
                    e_specs.append(spec)
                else:
                    fault_infos.append({"target": spec.target, "bit_count": int(spec.bit_count), "bit_op": spec.bit_op, "event_id": int(spec.event_id), "applied": 0, "effective": 0, "reason": "unsupported_target"})
        if v32_s1_runtime is not None:
            v32_s1_runtime.begin_sample(sample_id, s1_specs, int(args.experiment_seed))
        if v44_c_runtime is not None:
            v44_c_runtime.begin_sample(sample_id, c_specs, int(args.experiment_seed))
        if v44_d_runtime is not None:
            v44_d_runtime.begin_sample(sample_id, d_specs, int(args.experiment_seed))

        isolation_info.update({
            "fault_scope": fault_scope,
            "s0_fault_spec_count": int(sum(1 for s in specs if s.target == "can_bus")),
            "s1_fault_spec_count": int(len(s1_specs)),
            "c_fault_spec_count": int(len(c_specs)),
            "d_fault_spec_count": int(len(d_specs)),
            "e_fault_spec_count": int(len(e_specs)),
            "post_fault_can_bus_delta_from_entry": max_abs_can_bus_delta(data, original_can_bus_snapshot),
        })

        cb = object_can_bus(data)

        scene_context_summary = summarize_scene_context_from_data(data)
        reset_during = reset_runtime_capture_state(can_bus_capture, stage_captures)
        try:
            v26_online_head_capture.reset()
            reset_during["v26_online_head_capture_reset"] = 1
        except Exception:
            reset_during["v26_online_head_capture_reset"] = 0
        isolation_info["forward_capture_reset"] = reset_during

        start_time = time.time()
        result = inference_data(data, model_engine, tokenizer, args)
        inference_time = time.time() - start_time
        mlp_vec = can_bus_capture.output_vector(require=True)
        hook_summary = can_bus_capture.summary()
        stage_summaries = collect_stage_summaries(stage_captures)
        v26_online_head_features = v26_online_head_capture.summary()
        v35_s1_summary = {}
        if v32_s1_runtime is not None:
            s1_fault_infos, v35_s1_summary = v32_s1_runtime.finish_sample()
            fault_infos.extend(s1_fault_infos)
        v44_c_summary = {}
        if v44_c_runtime is not None:
            c_fault_infos, v44_c_summary = v44_c_runtime.finish_sample()
            fault_infos.extend(c_fault_infos)
        v44_d_summary = {}
        if v44_d_runtime is not None:
            d_fault_infos, v44_d_summary = v44_d_runtime.finish_sample()
            fault_infos.extend(d_fault_infos)
        v44_e_summary = {"pending_fault_count": int(len(e_specs)), "applied_fault_count": 0, "effective_fault_count": 0, "faults": []}
        for spec in e_specs:
            e_info = apply_v48_e1_internal_fault(result, sample_id, spec, int(args.experiment_seed))
            fault_infos.append(e_info)
            v44_e_summary["faults"].append(e_info)
        v44_e_summary["applied_fault_count"] = int(sum(int(f.get("applied", 0) or 0) for f in v44_e_summary["faults"]))
        v44_e_summary["effective_fault_count"] = int(sum(int(f.get("effective", 0) or 0) for f in v44_e_summary["faults"]))

        fault_spec_count = int(len(specs))
        fault_attempted_count = int(len(fault_infos))
        fault_applied_count = int(sum(int(fi.get("applied", 0) or 0) for fi in fault_infos))
        fault_effective_count = int(sum(int(fi.get("effective", 0) or 0) for fi in fault_infos))
        s0_fault_attempted_count = int(sum(1 for fi in fault_infos if str(fi.get("target", "")) == "can_bus"))
        s0_fault_effective_count = int(sum(int(fi.get("effective", 0) or 0) for fi in fault_infos if str(fi.get("target", "")) == "can_bus"))
        s1_fault_attempted_count = int(sum(1 for fi in fault_infos if str(fi.get("target", "")).startswith("s1:")))
        s1_fault_applied_count = int(sum(int(fi.get("applied", 0) or 0) for fi in fault_infos if str(fi.get("target", "")).startswith("s1:")))
        s1_fault_effective_count = int(sum(int(fi.get("effective", 0) or 0) for fi in fault_infos if str(fi.get("target", "")).startswith("s1:")))
        c_fault_attempted_count = int(sum(1 for fi in fault_infos if str(fi.get("target", "")).startswith("c:")))
        c_fault_applied_count = int(sum(int(fi.get("applied", 0) or 0) for fi in fault_infos if str(fi.get("target", "")).startswith("c:")))
        c_fault_effective_count = int(sum(int(fi.get("effective", 0) or 0) for fi in fault_infos if str(fi.get("target", "")).startswith("c:")))
        d_fault_attempted_count = int(sum(1 for fi in fault_infos if str(fi.get("target", "")).startswith("d:")))
        d_fault_applied_count = int(sum(int(fi.get("applied", 0) or 0) for fi in fault_infos if str(fi.get("target", "")).startswith("d:")))
        d_fault_effective_count = int(sum(int(fi.get("effective", 0) or 0) for fi in fault_infos if str(fi.get("target", "")).startswith("d:")))
        e_fault_attempted_count = int(sum(1 for fi in fault_infos if str(fi.get("target", "")).startswith("e:")))
        e_fault_applied_count = int(sum(int(fi.get("applied", 0) or 0) for fi in fault_infos if str(fi.get("target", "")).startswith("e:")))
        e_fault_effective_count = int(sum(int(fi.get("effective", 0) or 0) for fi in fault_infos if str(fi.get("target", "")).startswith("e:")))
        isolation_info.update({
            "fault_spec_count": fault_spec_count,
            "fault_attempted_count": fault_attempted_count,
            "fault_applied_count": fault_applied_count,
            "fault_effective_count": fault_effective_count,
            "s0_fault_attempted_count": s0_fault_attempted_count,
            "s0_fault_effective_count": s0_fault_effective_count,
            "s1_fault_attempted_count": s1_fault_attempted_count,
            "s1_fault_applied_count": s1_fault_applied_count,
            "s1_fault_effective_count": s1_fault_effective_count,
            "c_fault_attempted_count": c_fault_attempted_count,
            "c_fault_applied_count": c_fault_applied_count,
            "c_fault_effective_count": c_fault_effective_count,
            "d_fault_attempted_count": d_fault_attempted_count,
            "d_fault_applied_count": d_fault_applied_count,
            "d_fault_effective_count": d_fault_effective_count,
            "e_fault_attempted_count": e_fault_attempted_count,
            "e_fault_applied_count": e_fault_applied_count,
            "e_fault_effective_count": e_fault_effective_count,
            "v35_s1_summary": v35_s1_summary,
            "v44_c_summary": v44_c_summary,
            "v44_d_summary": v44_d_summary,
            "v44_e_summary": v44_e_summary,
        })

        isolation_info["post_forward_can_bus_delta_from_entry"] = max_abs_can_bus_delta(data, original_can_bus_snapshot)

        features = extract_s0_canbus_features(cb, mlp_vec, dim_stats)
        if args.mode == "calibrate":
            detector_report = {
                "objects": {"can_bus": {"pred_label": None, "soft_score": 0.0}},
                "combined": {"pred_label": None, "alarm_object_count": 0, "max_soft_score": 0.0},
            }
        else:
            detector_report = detect_s0(features, thresholds)

        object_vectors = {
            "can_bus": cb.reshape(-1).astype(np.float64).tolist(),
            "can_bus_mlp": mlp_vec.reshape(-1).astype(np.float64).tolist(),
        }

        online_context_features = compute_v26_command_context_features(data)

        row_source = "calibration_clean"
        if args.mode == "experiment":
            row_source = "experiment_injected" if int(is_positive) == 1 else "experiment_no_fault"

        record = build_record(
            sample_id,
            sample_id_full,
            is_positive,
            fault_infos,
            features,
            detector_report,
            object_vectors,
            hook_summary,
            stage_summaries,
            scene_context_summary,
            row_source=row_source,
            isolation_info=isolation_info,
            online_context_features=online_context_features,
            v26_online_head_features=v26_online_head_features,
            fault_spec_count=fault_spec_count,
            fault_attempted_count=fault_attempted_count,
            fault_applied_count=fault_applied_count,
            fault_effective_count=fault_effective_count,
        )
        record.update({
            "fault_scope": fault_scope,
            "s0_fault_spec_count": int(sum(1 for s in specs if s.target == "can_bus")),
            "s1_fault_spec_count": int(len(s1_specs)),
            "s0_fault_attempted_count": int(s0_fault_attempted_count),
            "s0_fault_effective_count": int(s0_fault_effective_count),
            "s1_fault_attempted_count": int(s1_fault_attempted_count),
            "s1_fault_applied_count": int(s1_fault_applied_count),
            "s1_fault_effective_count": int(s1_fault_effective_count),
            "c_fault_spec_count": int(len(c_specs)),
            "c_fault_attempted_count": int(c_fault_attempted_count),
            "c_fault_applied_count": int(c_fault_applied_count),
            "c_fault_effective_count": int(c_fault_effective_count),
            "d_fault_spec_count": int(len(d_specs)),
            "d_fault_attempted_count": int(d_fault_attempted_count),
            "d_fault_applied_count": int(d_fault_applied_count),
            "d_fault_effective_count": int(d_fault_effective_count),
            "e_fault_spec_count": int(len(e_specs)),
            "e_fault_attempted_count": int(e_fault_attempted_count),
            "e_fault_applied_count": int(e_fault_applied_count),
            "e_fault_effective_count": int(e_fault_effective_count),
            "v35_s1_summary": v35_s1_summary,
            "v44_c_summary": v44_c_summary,
            "v44_d_summary": v44_d_summary,
            "v44_e_summary": v44_e_summary,
        })
        if v53_inline_runtime is not None:
            try:
                inline_rec = v53_inline_runtime.score_sample(record, result, sample_id, sample_id_full)
            except Exception as e:
                if bool(getattr(args, "v53_inline_strict", True)):
                    raise
                inline_rec = {
                    "version": "v53_sample_inline_online_v35_1_detector_runtime_error",
                    "sample_id": sample_id,
                    "raw_sample_id": sample_id_full,
                    "row_source": record.get("row_source", ""),
                    "detector_timing": "sample_inline_after_current_forward_before_next_sample",
                    "error": f"{type(e).__name__}:{e}",
                    "detectors": {},
                    "analysis_only": {
                        "faults": record.get("faults", []),
                        "fault_target_exact": v53_first_fault_target(record.get("faults", [])),
                        "exact_layer": v53_exact_layer_from_fault_target(v53_first_fault_target(record.get("faults", []))),
                    },
                }
            record["v53_inline_detector"] = {
                "final_alarm_any_selected_detector": inline_rec.get("final_alarm_any_selected_detector", 0),
                "detectors": inline_rec.get("detectors", {}),
            }
            v53_inline_rank_records.append(inline_rec)
            with open(v53_inline_rank_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(inline_rec, ensure_ascii=False) + "\n")

        local_records.append(record)

        # Restore mutable inputs and clear hook buffers before the next sample.
        isolation_info["restore_can_bus_after_record"] = restore_can_bus_snapshot(data, original_can_bus_snapshot)
        isolation_info["post_record_capture_reset"] = reset_runtime_capture_state(can_bus_capture, stage_captures)
        try:
            v26_online_head_capture.reset()
            isolation_info["post_record_v26_online_head_capture_reset"] = 1
        except Exception:
            isolation_info["post_record_v26_online_head_capture_reset"] = 0
        if temporal_state_reset_enabled(args):
            isolation_info["model_temporal_reset_after_record"] = reset_model_temporal_runtime_state(model_engine)
        else:
            isolation_info["model_temporal_reset_after_record"] = disabled_model_temporal_runtime_state_status()

        previous_sample_state = {
            "sample_id": sample_id,
            "was_fault_selected": int(is_positive),
            "fault_count": fault_attempted_count,
            "effective_fault_count": fault_effective_count,
        }

        if rank == 0:
            tqdm_bar.set_postfix_str(f"Inference: {inference_time:.2f}s")
        with open(rank_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if args.local_rank != -1:
        rank_records_path = os.path.join(results_dir, f"detector_records_rank{rank}.jsonl")
        write_jsonl(rank_records_path, local_records)
        dist.barrier()

    if rank == 0:
        all_results = []
        all_records = []
        if args.local_rank != -1:
            for r in range(world_size):
                ro = args.output.replace(".json", f"_rank{r}.json")
                if os.path.exists(ro):
                    with open(ro, "r", encoding="utf-8") as f:
                        all_results.extend([json.loads(x) for x in f if x.strip()])
                rr = os.path.join(results_dir, f"detector_records_rank{r}.jsonl")
                if os.path.exists(rr):
                    with open(rr, "r", encoding="utf-8") as f:
                        all_records.extend([json.loads(x) for x in f if x.strip()])
        else:
            if os.path.exists(rank_output):
                with open(rank_output, "r", encoding="utf-8") as f:
                    all_results.extend([json.loads(x) for x in f if x.strip()])
            all_records = list(local_records)

        with open(args.output, "w", encoding="utf-8") as f:
            for row in all_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Successfully merged results into {args.output}")

        records_path = os.path.join(results_dir, "detector_records.jsonl")
        write_jsonl(records_path, all_records)
        if bool(getattr(args, "v53_inline_detector", False)):
            all_inline_records: List[Dict[str, Any]] = []
            if args.local_rank != -1:
                for r in range(world_size):
                    rp = os.path.join(results_dir, f"v53_inline_detector_records_rank{r}.jsonl")
                    if os.path.exists(rp):
                        with open(rp, "r", encoding="utf-8") as f:
                            all_inline_records.extend([json.loads(x) for x in f if x.strip()])
            else:
                all_inline_records = list(v53_inline_rank_records)
            inline_path = os.path.join(results_dir, "v53_inline_detector_records.jsonl")
            write_jsonl(inline_path, all_inline_records)
            write_json(os.path.join(results_dir, "v53_inline_detector_summary.json"), v53_inline_summarize_records(all_inline_records))
            print(f"[{VERSION}] wrote sample-inline detector records to {inline_path}")
        write_json(os.path.join(results_dir, "stage_hook_status_v26.json"), stage_hook_status)
        write_json(os.path.join(results_dir, "stage_summary_distribution_v26.json"), summarize_stage_distributions(all_records))
        write_json(os.path.join(results_dir, "e_scene_context_available_v26.json"), summarize_scene_context_records(all_records))

        if args.mode == "calibrate":
            thresholds = compute_thresholds(all_records, args)
            recompute_features_for_records(all_records, thresholds["dim_stats"])
            thresholds_path = os.path.join(results_dir, "detector_thresholds.json")
            write_json(thresholds_path, thresholds)
            write_json(os.path.join(results_dir, "node_distribution_summary.json"), summarize_feature_distribution(all_records))
            write_json(os.path.join(results_dir, "feature_analysis_debug.json"), build_feature_analysis_debug(all_records, thresholds))
            report = {
                "mode": "calibrate",
                "num_records": len(all_records),
                "calibration_size": len(all_records),
                "implemented_stage": "S0_can_bus_only",
                "version": VERSION,
                "target_mode": args.target_mode,
                "fault_targets": list(FAULT_TARGETS),
                "detector_objects": list(DETECTOR_OBJECTS),
                "thresholds_path": thresholds_path,
                "can_bus_groups": CANBUS_GROUPS,
                "can_bus_group_features": CANBUS_GROUP_FEATURE_MAP,
                "can_bus_mlp_features": CANBUS_MLP_FEATURES,
                "can_bus_mlp_hook": hook_name,
                "stage_hook_status": stage_hook_status,
                "v32_s1_hook_status": v32_s1_hook_status,
                "v44_c_hook_status": v44_c_hook_status,
                "v44_d_hook_status": v44_d_hook_status,
            }
            write_json(os.path.join(results_dir, "detector_report.json"), report)
        else:
            by_target = {}
            by_count = {"1": [], "2": [], "3": []}
            by_group = {g: [] for g in CANBUS_GROUP_ORDER}
            for r in all_records:
                if r["faults"]:
                    tgt = r["faults"][0].get("target")
                    bc = str(r["faults"][0].get("bit_count"))
                    g = r["faults"][0].get("impact", {}).get("group")
                    by_target.setdefault(str(tgt), []).append(r)
                    if bc in by_count:
                        by_count[bc].append(r)
                    if g in by_group:
                        by_group[g].append(r)
            nominal_positive = int(sum(int(r.get("is_positive", 0)) for r in all_records))
            effective_positive = int(sum(int(r.get("is_positive_effective", 0)) for r in all_records))
            generation_failures = int(sum(1 for r in all_records if int(r.get("is_positive", 0)) == 1 and int(r.get("is_positive_effective", 0)) == 0))
            report = {
                "mode": "experiment",
                "num_records": len(all_records),
                "version": VERSION,
                "target_mode": args.target_mode,
                "fault_targets": list(FAULT_TARGETS),
                "detector_objects": list(DETECTOR_OBJECTS),
                "nominal_positive_count": nominal_positive,
                "effective_positive_count": effective_positive,
                "generation_failures": generation_failures,
                "combined": classification_metrics(all_records, key_path="combined"),
                "objects": {
                    "can_bus": classification_metrics(
                        [{**r, "detector_report": {"combined": r["detector_report"]["objects"]["can_bus"]}} for r in all_records],
                        key_path="combined",
                    )
                },
                "by_target": {k: classification_metrics(v, key_path="combined") for k, v in by_target.items()},
                "by_bit_count": {k: classification_metrics(v, key_path="combined") for k, v in by_count.items()},
                "by_can_bus_group": {k: classification_metrics(v, key_path="combined") for k, v in by_group.items()},
                "can_bus_groups": CANBUS_GROUPS,
                "can_bus_group_features": CANBUS_GROUP_FEATURE_MAP,
                "can_bus_mlp_features": CANBUS_MLP_FEATURES,
                "can_bus_mlp_hook": hook_name,
                "stage_hook_status": stage_hook_status,
                "v32_s1_hook_status": v32_s1_hook_status,
                "by_fault_scope": {scope: classification_metrics([r for r in all_records if str(r.get("fault_scope", "no_fault")) == scope], key_path="combined") for scope in sorted(set(str(r.get("fault_scope", "no_fault")) for r in all_records))},
            }
            write_json(os.path.join(results_dir, "node_runtime_summary.json"), summarize_feature_distribution(all_records))
            write_json(os.path.join(results_dir, "feature_analysis_debug.json"), build_feature_analysis_debug(all_records, thresholds))
            write_json(os.path.join(results_dir, "detector_report.json"), report)

    for _h in [can_bus_hook_handle, v26_online_head_handle] + list(stage_hook_handles) + list(v32_s1_hook_handles) + list(v44_c_hook_handles) + list(v44_d_hook_handles):
        try:
            _h.remove()
        except Exception:
            pass


def _safe_mean(vals: List[float]) -> float:
    return float(sum(vals) / max(len(vals), 1))


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def robust_retrieve_traj_with_meta(answer: Any) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """Parse trajectory and return lightweight E-stage parse semantics.

    The metadata is online-computable from the model's current generated answer.
    It does not use GT, clean reference, or harmful labels.
    """
    if isinstance(answer, list):
        raw_text = " ".join(str(x) for x in answer)
    elif isinstance(answer, tuple):
        raw_text = " ".join(str(x) for x in answer)
    else:
        raw_text = str(answer)

    numeric_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_text)
    float_tokens = [t for t in numeric_tokens if ("." in t or "e" in t.lower())]
    marker_tokens = ["trajectory", "traj", "waypoint", "points", "planning", "plan"]
    repeated_char_max = 0
    if raw_text:
        cur = 1
        for i in range(1, len(raw_text)):
            if raw_text[i] == raw_text[i - 1]:
                cur += 1
            else:
                repeated_char_max = max(repeated_char_max, cur)
                cur = 1
        repeated_char_max = max(repeated_char_max, cur)

    meta: Dict[str, Any] = {
        "answer_type": type(answer).__name__,
        "answer_char_len": len(raw_text),
        "answer_token_like_len": len(raw_text.split()),
        "answer_line_count": max(1, raw_text.count("\n") + 1) if raw_text else 0,
        "answer_digit_count": len(re.findall(r"\d", raw_text)),
        "answer_minus_count": raw_text.count("-"),
        "answer_decimal_count": raw_text.count("."),
        "answer_comma_count": raw_text.count(","),
        "answer_colon_count": raw_text.count(":"),
        "answer_semicolon_count": raw_text.count(";"),
        "answer_lparen_count": raw_text.count("("),
        "answer_lbracket_count": raw_text.count("["),
        "answer_numeric_token_count": len(numeric_tokens),
        "answer_float_token_count": len(float_tokens),
        "answer_unique_numeric_token_count": len(set(numeric_tokens)),
        "answer_repeated_char_max_run": repeated_char_max,
        "answer_is_empty": int(len(raw_text.strip()) == 0),
        "answer_is_maxlen_like": int(len(raw_text) >= 512),
        "answer_has_alpha": int(bool(re.search(r"[A-Za-z]", raw_text))),
        "answer_has_chinese": int(bool(re.search(r"[\u4e00-\u9fff]", raw_text))),
        "answer_has_bracket": int(any(ch in raw_text for ch in "[]()<>")),
        "answer_has_trajectory_marker": int(any(tok in raw_text.lower() for tok in marker_tokens)),
        "answer_has_explanatory_text": int(bool(re.search(r"[A-Za-z]{4,}", raw_text))),
    }

    text = re.sub(r'[a-zA-Z]', '', raw_text)
    text = re.sub(r'[\u4e00-\u9fff]', '', text)
    text = re.sub(r'(\d+)\.\.+', r'\1.', text)
    text = re.sub(r'[<>]', '', text)
    text = re.sub(r'(\d+\.\d+)\s+(\d+\.\d+)(?=\s*[,\)])', r'\1, \2', text)
    text = re.sub(r'(\d+\.\d+)-(\d+\.\d+)', r'\1, \2', text)
    text = re.sub(r'(\(\s*-?\d+\.\d+,\s*-?\d+\.\d+),\s*-?\d+\.\d+(\s*\))', r'\1\2', text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'(\d+\.\d+)\.(\d+)', r'\1\2', text)
    text = re.sub(r'(\d+)\.\.(\d+)', r'\1.\2', text)
    text = re.sub(r'(\d+)\.\-(\d+)', r'\1.\2', text)

    coord_pairs = re.findall(r'[\[\(]([-\deE.+]+),\s*([-\deE.+]+)[\]\)]', text)
    meta["raw_coord_pair_count"] = len(coord_pairs)
    meta["parse_success"] = int(len(coord_pairs) > 0)
    meta["parse_padded"] = int(0 < len(coord_pairs) < 6)
    meta["parse_truncated"] = int(len(coord_pairs) > 6)
    meta["parse_fallback_zero"] = int(len(coord_pairs) == 0)
    meta["parse_fail_no_numeric_tokens"] = int(len(coord_pairs) == 0 and len(numeric_tokens) == 0)
    meta["parse_fail_numbers_no_pairs"] = int(len(coord_pairs) == 0 and len(numeric_tokens) > 0)
    meta["parse_fail_brackets_no_pairs"] = int(len(coord_pairs) == 0 and meta.get("answer_has_bracket", 0) == 1)
    meta["parse_fail_no_brackets"] = int(len(coord_pairs) == 0 and meta.get("answer_has_bracket", 0) == 0)

    coords_list: List[Tuple[float, float]] = []
    try:
        coords_list = [(float(x), float(y)) for x, y in coord_pairs]
    except Exception:
        coords_list = []
        meta["parse_success"] = 0
        meta["parse_fallback_zero"] = 1
        meta["parse_exception"] = 1
    else:
        meta["parse_exception"] = 0

    if not coords_list:
        traj = [(0.0, 0.0)] * 6
    else:
        if len(coords_list) < 6:
            coords_list.extend([coords_list[-1]] * (6 - len(coords_list)))
        elif len(coords_list) > 6:
            coords_list = coords_list[:6]
        traj = coords_list

    arr = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
    meta["valid_waypoint_count"] = int(np.isfinite(arr).all(axis=1).sum()) if arr.size else 0
    meta["parsed_all_zero"] = int(float(np.max(np.abs(np.nan_to_num(arr, nan=0.0)))) < 1e-9) if arr.size else 1
    meta["parsed_final_disp"] = float(np.linalg.norm(np.nan_to_num(arr[-1], nan=0.0))) if arr.size else 0.0
    if len(arr) >= 2:
        step = np.linalg.norm(np.diff(np.nan_to_num(arr, nan=0.0), axis=0), axis=1)
        meta["parsed_path_length"] = float(np.sum(step))
        meta["parsed_step_nonzero_count"] = int((step > 1e-6).sum())
        meta["parsed_repeated_waypoint_count"] = int((step <= 1e-9).sum())
    else:
        meta["parsed_path_length"] = 0.0
        meta["parsed_step_nonzero_count"] = 0
        meta["parsed_repeated_waypoint_count"] = 0
    return traj, meta


def robust_retrieve_traj(answer: Any) -> List[Tuple[float, float]]:
    traj, _ = robust_retrieve_traj_with_meta(answer)
    return traj




def v48_plan_conv_forced_traj(row: Dict[str, Any]) -> Tuple[Optional[List[Tuple[float, float]]], Dict[str, Any]]:
    """Return a forced E1 trajectory object stored in plan_conv, if present."""
    if "v48_e1_forced_traj" not in row:
        return None, {"v48_e1_forced_traj_used": 0}
    forced = v48_normalize_trajectory_object(row.get("v48_e1_forced_traj"))
    if not forced:
        return None, {"v48_e1_forced_traj_used": 0, "v48_e1_forced_traj_invalid": 1}
    meta = {
        "v48_e1_forced_traj_used": 1,
        "v48_e1_forced_traj_invalid": 0,
        "v48_e1_fault_meta": row.get("v48_e1_fault_meta", {}),
    }
    arr = np.asarray(forced, dtype=np.float64).reshape(-1, 2)
    meta["valid_waypoint_count"] = int(np.isfinite(arr).all(axis=1).sum()) if arr.size else 0
    meta["parsed_all_zero"] = int(float(np.max(np.abs(np.nan_to_num(arr, nan=0.0)))) < 1e-9) if arr.size else 1
    meta["parsed_final_disp"] = float(np.linalg.norm(np.nan_to_num(arr[-1], nan=0.0))) if arr.size else 0.0
    if len(arr) >= 2:
        step = np.linalg.norm(np.diff(np.nan_to_num(arr, nan=0.0), axis=0), axis=1)
        meta["parsed_path_length"] = float(np.sum(step))
        meta["parsed_step_nonzero_count"] = int((step > 1e-6).sum())
        meta["parsed_repeated_waypoint_count"] = int((step <= 1e-9).sum())
    else:
        meta["parsed_path_length"] = 0.0
        meta["parsed_step_nonzero_count"] = 0
        meta["parsed_repeated_waypoint_count"] = 0
    return forced, meta
def build_pred_files_from_plan_conv(result_dir: Path) -> Dict[str, Any]:
    plan_conv_path = result_dir / 'plan_conv.json'
    if not plan_conv_path.exists():
        raise FileNotFoundError(f'Missing {plan_conv_path}')

    pred = {}
    pred_multi = {}
    parse_meta_rows = []
    invalid = []
    total = 0
    with plan_conv_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            sid = str(row.get('id', ''))
            ans = row.get('answer', '')
            forced_traj, forced_meta = v48_plan_conv_forced_traj(row)
            if forced_traj is not None:
                traj = forced_traj
                _, answer_parse_meta = robust_retrieve_traj_with_meta(ans)
                parse_meta = dict(answer_parse_meta)
                parse_meta.update(forced_meta)
            else:
                traj, parse_meta = robust_retrieve_traj_with_meta(ans)
            if traj == [(0.0, 0.0)] * 6:
                invalid.append(sid)
            pred[sid] = traj
            pred_multi[sid] = [traj]
            parse_meta_rows.append({"sample_id": normalize_sample_token(sid), "raw_sample_id": sid, "e_parse_features": parse_meta})

    pred_path = result_dir / 'pred_trajs_dict.json'
    pred_multi_path = result_dir / 'pred_trajs_multi_modal_dict.json'
    parse_meta_path = result_dir / 'e_parse_metadata_v26.jsonl'
    with pred_path.open('w', encoding='utf-8') as f:
        json.dump(pred, f, ensure_ascii=False)
    with pred_multi_path.open('w', encoding='utf-8') as f:
        json.dump(pred_multi, f, ensure_ascii=False)
    with parse_meta_path.open('w', encoding='utf-8') as f:
        for row in parse_meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    return {
        'count': total,
        'invalid_parse_count': len(invalid),
        'invalid_parse_examples': invalid[:10],
        'pred_path': str(pred_path),
        'pred_multi_path': str(pred_multi_path),
        'e_parse_meta_path': str(parse_meta_path),
    }


def _iter_plan_conv_tokens(path: Path) -> List[str]:
    tokens: List[str] = []
    if not path.exists():
        return tokens
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sid = str(row.get('id', ''))
            if sid.endswith('_trajectory'):
                sid = sid[:-11]
            if sid:
                tokens.append(sid)
    return tokens


def _pred_lookup(pred_trajs_dict: Dict[str, Any], token: str):
    for k in (token + '_trajectory', token):
        if k in pred_trajs_dict:
            return pred_trajs_dict[k]
    return None



def trajectory_sanity_features_from_pred(pred: Any) -> Dict[str, float]:
    """Output trajectory sanity plus v19 collapse decomposition.

    All fields are computed from the current predicted trajectory only. No clean
    reference, GT future trajectory, or next-frame state is used.
    """
    arr = np.asarray(pred, dtype=np.float64)

    def empty(nonfinite_ratio: float = 1.0) -> Dict[str, float]:
        return {
            "traj_nonfinite_ratio": float(nonfinite_ratio),
            "traj_all_zero": 1.0,
            "traj_abs_max": math.inf,
            "step_max": math.inf,
            "step_mean": math.inf,
            "path_length": math.inf,
            "accel_max": math.inf,
            "jerk_max": math.inf,
            "final_disp": math.inf,
            "selected_traj_all_zero": 1.0,
            "selected_traj_nonzero_waypoint_count": 0.0,
            "selected_traj_valid_waypoint_count": 0.0,
            "selected_traj_near_zero_waypoint_fraction": 1.0,
            "candidate_count_proxy": 1.0,
            "candidate_all_zero_count_proxy": 1.0,
            "candidate_nonzero_count_proxy": 0.0,
            "candidate_diversity_proxy": 0.0,
        }

    if arr.size == 0:
        return empty(1.0)
    try:
        arr = arr.reshape(-1, 2)
    except Exception:
        arr = np.zeros((0, 2), dtype=np.float64)
    finite = np.isfinite(arr)
    nonfinite_ratio = 1.0 - float(finite.mean()) if arr.size else 1.0
    if arr.size == 0 or not np.isfinite(arr).any():
        return empty(nonfinite_ratio)
    clean = np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)
    waypoint_norm = np.linalg.norm(clean, axis=1) if clean.size else np.zeros(0, dtype=np.float64)
    step = np.diff(clean, axis=0)
    step_norm = np.linalg.norm(step, axis=1) if step.size else np.zeros(0, dtype=np.float64)
    accel = np.diff(step, axis=0)
    accel_norm = np.linalg.norm(accel, axis=1) if accel.size else np.zeros(0, dtype=np.float64)
    jerk = np.diff(accel, axis=0)
    jerk_norm = np.linalg.norm(jerk, axis=1) if jerk.size else np.zeros(0, dtype=np.float64)
    all_zero = float(np.max(np.abs(clean)) < 1e-9)
    valid_wp = float(np.isfinite(arr).all(axis=1).sum()) if arr.size else 0.0
    nonzero_wp = float((waypoint_norm > 1e-6).sum()) if waypoint_norm.size else 0.0
    near_zero_frac = float((waypoint_norm <= 1e-6).mean()) if waypoint_norm.size else 1.0
    # v22 fine-grained trajectory geometry. These are online output-only features;
    # they do not use clean reference, GT, or future frames. The coordinate convention
    # is left unchanged; x/y are only used as generic longitudinal/lateral proxies.
    final_xy = clean[-1] if clean.size else np.zeros(2, dtype=np.float64)
    first_step = step[0] if step.size else np.zeros(2, dtype=np.float64)
    last_step = step[-1] if step.size else np.zeros(2, dtype=np.float64)
    def _angle(v):
        if float(np.linalg.norm(v)) < 1e-9:
            return 0.0
        return float(math.atan2(float(v[1]), float(v[0])))
    headings = []
    if step.size:
        for v in step:
            if float(np.linalg.norm(v)) >= 1e-9:
                headings.append(_angle(v))
    turn_angles = []
    for a, b in zip(headings[:-1], headings[1:]):
        d = (b - a + math.pi) % (2 * math.pi) - math.pi
        turn_angles.append(abs(float(d)))
    heading_change = 0.0
    if len(headings) >= 2:
        heading_change = abs(float((headings[-1] - headings[0] + math.pi) % (2 * math.pi) - math.pi))
    path_length = float(np.sum(step_norm)) if step_norm.size else 0.0
    final_disp = float(np.linalg.norm(final_xy)) if clean.size else 0.0
    straightness = float(final_disp / (path_length + 1e-9)) if path_length > 0 else 0.0
    x_range = float(np.max(clean[:, 0]) - np.min(clean[:, 0])) if clean.size else 0.0
    y_range = float(np.max(clean[:, 1]) - np.min(clean[:, 1])) if clean.size else 0.0
    step_std = float(np.std(step_norm)) if step_norm.size else 0.0
    step_min = float(np.min(step_norm)) if step_norm.size else 0.0
    curvature_proxy = float(np.sum(turn_angles) / (path_length + 1e-9)) if path_length > 0 else 0.0
    smoothness_proxy = float(np.mean(accel_norm)) if accel_norm.size else 0.0
    endpoint_angle = _angle(final_xy)
    return {
        "traj_nonfinite_ratio": float(nonfinite_ratio),
        "traj_all_zero": all_zero,
        "traj_abs_max": float(np.max(np.abs(clean))),
        "step_max": float(np.max(step_norm)) if step_norm.size else 0.0,
        "step_mean": float(np.mean(step_norm)) if step_norm.size else 0.0,
        "path_length": path_length,
        "accel_max": float(np.max(accel_norm)) if accel_norm.size else 0.0,
        "jerk_max": float(np.max(jerk_norm)) if jerk_norm.size else 0.0,
        "final_disp": final_disp,
        "endpoint_x_proxy": float(final_xy[0]) if final_xy.size >= 1 else 0.0,
        "endpoint_y_proxy": float(final_xy[1]) if final_xy.size >= 2 else 0.0,
        "endpoint_abs_y_proxy": float(abs(final_xy[1])) if final_xy.size >= 2 else 0.0,
        "first_step_norm": float(np.linalg.norm(first_step)) if first_step.size else 0.0,
        "last_step_norm": float(np.linalg.norm(last_step)) if last_step.size else 0.0,
        "initial_heading_proxy": _angle(first_step),
        "final_heading_proxy": _angle(last_step),
        "heading_change_proxy": heading_change,
        "turn_angle_abs_sum_proxy": float(np.sum(turn_angles)) if turn_angles else 0.0,
        "turn_angle_abs_max_proxy": float(np.max(turn_angles)) if turn_angles else 0.0,
        "turn_angle_abs_mean_proxy": float(np.mean(turn_angles)) if turn_angles else 0.0,
        "straightness_proxy": straightness,
        "longitudinal_range_proxy": x_range,
        "lateral_range_proxy": y_range,
        "step_min": step_min,
        "step_std": step_std,
        "curvature_proxy": curvature_proxy,
        "smoothness_proxy": smoothness_proxy,
        "endpoint_angle_proxy": endpoint_angle,
        "near_zero_final_disp_flag": float(int(final_disp <= 0.2)),
        "near_zero_path_length_flag": float(int(path_length <= 0.2)),
        "selected_traj_all_zero": all_zero,
        "selected_traj_nonzero_waypoint_count": nonzero_wp,
        "selected_traj_valid_waypoint_count": valid_wp,
        "selected_traj_near_zero_waypoint_fraction": near_zero_frac,
        "candidate_count_proxy": 1.0,
        "candidate_all_zero_count_proxy": float(int(all_zero >= 0.5)),
        "candidate_nonzero_count_proxy": float(int(all_zero < 0.5)),
        "candidate_diversity_proxy": 0.0,
    }


OUTPUT_SANITY_FEATURES = [
    "traj_nonfinite_ratio",
    "traj_abs_max",
    "step_max",
    "step_mean",
    "path_length",
    "accel_max",
    "jerk_max",
    "final_disp",
    "endpoint_x_proxy",
    "endpoint_y_proxy",
    "endpoint_abs_y_proxy",
    "first_step_norm",
    "last_step_norm",
    "initial_heading_proxy",
    "final_heading_proxy",
    "heading_change_proxy",
    "turn_angle_abs_sum_proxy",
    "turn_angle_abs_max_proxy",
    "turn_angle_abs_mean_proxy",
    "straightness_proxy",
    "longitudinal_range_proxy",
    "lateral_range_proxy",
    "step_min",
    "step_std",
    "curvature_proxy",
    "smoothness_proxy",
    "endpoint_angle_proxy",
    "near_zero_final_disp_flag",
    "near_zero_path_length_flag",
    "selected_traj_nonzero_waypoint_count",
    "selected_traj_valid_waypoint_count",
    "selected_traj_near_zero_waypoint_fraction",
    "candidate_count_proxy",
    "candidate_all_zero_count_proxy",
    "candidate_nonzero_count_proxy",
    "candidate_diversity_proxy",
]


def build_output_sanity_thresholds(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    thresholds: Dict[str, Any] = {
        "version": VERSION,
        "description": "Output-only sanity thresholds calibrated from clean predictions; used only to promote suspicious input alarms.",
        "feature_thresholds": {},
        "all_zero_rate": 0.0,
        "sample_count": len(rows),
    }
    if not rows:
        return thresholds
    thresholds["all_zero_rate"] = float(np.mean([float((r.get("output_sanity_features", {}) or {}).get("traj_all_zero", 0.0)) for r in rows]))
    for feat in OUTPUT_SANITY_FEATURES:
        vals = [float((r.get("output_sanity_features", {}) or {}).get(feat, 0.0)) for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        thr = compute_feature_threshold(vals, kind="upper_only", mad_scale=8.0, strong_mad_scale=12.0, p_high=0.995, p_vhigh=0.999)
        if thr is not None:
            thresholds["feature_thresholds"][feat] = thr
    return thresholds


def detect_output_sanity(row: Dict[str, Any], thresholds: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    feats = row.get("output_sanity_features", {}) or {}
    if thresholds is None:
        return {"pred_label": 0, "alarm_count": 0, "strong_count": 0, "soft_score": 0.0, "details": {}, "reason": "no_thresholds"}
    details: Dict[str, Any] = {}
    hard = 0
    strong = 0
    soft = 0.0
    # Invalid/all-zero parse is suspicious output behavior, but clean calibration can
    # contain such cases. It therefore only contributes one hard hit and never acts
    # alone unless paired with suspicious input.
    all_zero = int(float(feats.get("traj_all_zero", 0.0)) >= 0.5)
    if all_zero:
        hard += 1
        soft += 1.0
        details["traj_all_zero"] = {"value": 1.0, "alarm": 1, "strong_alarm": 0, "margin": 1.0}
    for feat in OUTPUT_SANITY_FEATURES:
        thr = thresholds.get("feature_thresholds", {}).get(feat)
        if thr is None:
            continue
        val = float(feats.get(feat, 0.0))
        outcome = detect_value(val, thr)
        details[feat] = {"value": val, **outcome}
        hard += int(outcome.get("alarm", 0))
        strong += int(outcome.get("strong_alarm", 0))
        soft += float(outcome.get("margin", 0.0))
    # Require either a strong output anomaly, multiple weak output anomalies, or
    # a very large aggregate margin. This is deliberately conservative.
    pred = int(strong >= 1 or hard >= 2 or soft >= 3.0)
    return {"pred_label": pred, "alarm_count": int(hard), "strong_count": int(strong), "soft_score": float(soft), "details": details}



def load_e_parse_metadata(result_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = result_dir / 'e_parse_metadata_v26.jsonl'
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sid = normalize_sample_token(str(row.get('sample_id') or row.get('raw_sample_id') or ''))
            if sid:
                out[sid] = row.get('e_parse_features', {}) or {}
    return out


def v19_ego_motion_context_features_from_canbus_vector(vec: Optional[List[float]]) -> Dict[str, float]:
    """Low-cost context proxy from current can_bus values.

    The field meanings are not assumed as ground truth speed. We expose group-level
    magnitude proxies so the learner/rule analysis can decide whether an all-zero
    trajectory is consistent with a near-stationary current state.
    """
    out: Dict[str, float] = {}
    if not vec:
        return out
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e12, neginf=-1e12)
    out['egoctx__canbus_abs_max'] = float(np.max(np.abs(arr))) if arr.size else 0.0
    for g, idxs in CANBUS_GROUPS.items():
        vals = arr[[i for i in idxs if i < len(arr)]] if arr.size else np.zeros(0)
        if vals.size:
            out[f'egoctx__{g}__abs_sum'] = float(np.sum(np.abs(vals)))
            out[f'egoctx__{g}__abs_max'] = float(np.max(np.abs(vals)))
            out[f'egoctx__{g}__l2'] = float(np.linalg.norm(vals))
    small = [i for i in SMALL_SIGNED_STATE_IDXS if i < len(arr)]
    if small:
        vals = arr[small]
        out['egoctx__motion_proxy_abs_sum'] = float(np.sum(np.abs(vals)))
        out['egoctx__motion_proxy_l2'] = float(np.linalg.norm(vals))
        out['egoctx__motion_proxy_near_zero'] = float(int(np.linalg.norm(vals) < 1e-6))
    return out

def compute_subset_metrics(result_dir: Path, log_path: Path) -> Tuple[Path, Dict[str, Any]]:
    from eval_share.metric import PlanningMetric
    import pickle
    import eval_share.evaluation as evaluation_mod

    pred_path = result_dir / 'pred_trajs_dict.json'
    plan_conv_path = result_dir / 'plan_conv.json'
    if not pred_path.exists():
        raise FileNotFoundError(f'Missing {pred_path}')
    pred_trajs_dict = _load_json(pred_path)
    subset_tokens = _iter_plan_conv_tokens(plan_conv_path)
    e_parse_meta = load_e_parse_metadata(result_dir)

    current_dir = Path(evaluation_mod.__file__).resolve().parent
    with open(current_dir / 'gt/planing_gt_segmentation_val', 'rb') as f:
        gt_occ_map = pickle.load(f)
    for token in list(gt_occ_map.keys()):
        if not isinstance(gt_occ_map[token], torch.Tensor):
            gt_occ_map[token] = torch.tensor(gt_occ_map[token])
        gt_occ_map[token] = torch.flip(gt_occ_map[token], [-1])
    with open(current_dir / 'gt/gt_traj.pkl', 'rb') as f:
        gt_trajs_dict = pickle.load(f)
    with open(current_dir / 'gt/gt_traj_mask.pkl', 'rb') as f:
        gt_trajs_mask_dict = pickle.load(f)

    metric = PlanningMetric(6).to(torch.device('cpu'))
    rows = []
    missing_pred = []
    missing_gt = []

    for token in subset_tokens:
        pred = _pred_lookup(pred_trajs_dict, token)
        if pred is None:
            missing_pred.append(token)
            continue
        if token not in gt_trajs_dict or token not in gt_trajs_mask_dict or token not in gt_occ_map:
            missing_gt.append(token)
            continue

        gt_traj = torch.tensor(gt_trajs_dict[token]).to(torch.device('cpu'))
        gt_mask = torch.tensor(gt_trajs_mask_dict[token]).to(torch.device('cpu'))
        occ = gt_occ_map[token].to(torch.device('cpu'))
        out = torch.tensor(pred).reshape(gt_mask.shape).to(torch.device('cpu'))

        if out.shape[1] % 2:
            out = out[:, 1:]
        if occ.shape[1] % 2:
            occ = occ[:, 1:]
        if gt_traj.shape[1] % 2:
            gt_traj = gt_traj[:, 1:]
        if gt_mask.shape[1] % 2:
            gt_mask = gt_mask[:, 1:]

        out = out[:, :6]
        gt_traj = gt_traj[:, :6]
        gt_mask = gt_mask[:, :6]
        occ = occ[:, :6]

        metric(out, gt_traj, occ, token, gt_mask)

        l2 = metric.compute_L2(out, gt_traj, gt_mask)[0].tolist()
        _, obj_box = metric.evaluate_coll(out[:, :, :2], gt_traj[:, :, :2], occ, token=None)
        obj_box = [float(x) for x in obj_box.tolist()]
        row = {
            'sample_id': token,
            'l2_t': l2,
            'collision_t': obj_box,
            'l2_1s': float(sum(l2[:2]) / 2.0),
            'l2_2s': float(sum(l2[:4]) / 4.0),
            'l2_3s': float(sum(l2[:6]) / 6.0),
            'l2_avg': float((sum(l2[:2]) / 2.0 + sum(l2[:4]) / 4.0 + sum(l2[:6]) / 6.0) / 3.0),
            'collision_1s': float(sum(obj_box[:2]) / 2.0 * 100.0),
            'collision_2s': float(sum(obj_box[:4]) / 4.0 * 100.0),
            'collision_3s': float(sum(obj_box[:6]) / 6.0 * 100.0),
            'collision_avg': float(((sum(obj_box[:2]) / 2.0) + (sum(obj_box[:4]) / 4.0) + (sum(obj_box[:6]) / 6.0)) / 3.0 * 100.0),
            'output_sanity_features': trajectory_sanity_features_from_pred(pred),
            'e_parse_features': e_parse_meta.get(token, {}),
            'v26_agent_features': compute_v26_agent_context_features(token, pred),
            'v26_ego_pose_features': compute_v26_ego_pose_context_features(token),
        }
        rows.append(row)

    # Do not call metric.compute() here. In torchrun/NCCL mode, torchmetrics
    # tries to synchronize CPU states with all_gather and raises:
    # RuntimeError: No backend type associated with device type cpu.
    # The per-sample L2/collision values above are already computed with the
    # official PlanningMetric helper methods, so aggregate them manually.
    if rows:
        l2_arr = np.asarray([r['l2_t'] for r in rows], dtype=np.float64)
        col_arr = np.asarray([r['collision_t'] for r in rows], dtype=np.float64)
    else:
        l2_arr = np.zeros((0, 6), dtype=np.float64)
        col_arr = np.zeros((0, 6), dtype=np.float64)

    def _mean_prefix(arr: np.ndarray, n: int) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr[:, :n]))

    def _mean_step(arr: np.ndarray, idx: int) -> float:
        if arr.size == 0:
            return 0.0
        idx = min(idx, arr.shape[1] - 1)
        return float(np.mean(arr[:, idx]))

    stp3 = {
        'L2_1s': _mean_prefix(l2_arr, 2),
        'L2_2s': _mean_prefix(l2_arr, 4),
        'L2_3s': _mean_prefix(l2_arr, 6),
        'collision_1s': _mean_prefix(col_arr, 2) * 100.0,
        'collision_2s': _mean_prefix(col_arr, 4) * 100.0,
        'collision_3s': _mean_prefix(col_arr, 6) * 100.0,
    }
    stp3['L2_avg'] = float((stp3['L2_1s'] + stp3['L2_2s'] + stp3['L2_3s']) / 3.0)
    stp3['collision_avg'] = float((stp3['collision_1s'] + stp3['collision_2s'] + stp3['collision_3s']) / 3.0)

    uniad = {
        'L2_1s': _mean_step(l2_arr, 1),
        'L2_2s': _mean_step(l2_arr, 3),
        'L2_3s': _mean_step(l2_arr, 5),
        'collision_1s': _mean_step(col_arr, 1) * 100.0,
        'collision_2s': _mean_step(col_arr, 3) * 100.0,
        'collision_3s': _mean_step(col_arr, 5) * 100.0,
    }
    uniad['L2_avg'] = float((uniad['L2_1s'] + uniad['L2_2s'] + uniad['L2_3s']) / 3.0)
    uniad['collision_avg'] = float((uniad['collision_1s'] + uniad['collision_2s'] + uniad['collision_3s']) / 3.0)

    # Write official-style log.
    headers = ["Method", "L2 (m)", "Collision (%)"]
    subs = ["1s", "2s", "3s", "Avg."]
    lines = []
    lines.append("\n-------------------------------------------------------------------")
    lines.append(f"Processed total {len(rows)} samples")
    lines.append(f"gt collision: {metric.gt_collision}")
    lines.append("\n-------------------------------------------------------------------")
    lines.append("UniAD evaluation:")
    lines.append("{:<23} {:<22} {:<20}".format(*headers))
    lines.append("{:<15} {:<5} {:<5} {:<5} {:<8} {:<5} {:<5} {:<5} {:<5}".format("", *subs, *subs))
    lines.append("{:<15} {:<5} {:<5} {:<5} {:<8} {:<5} {:<5} {:<5} {:<5}".format(
        "DriveAgent",
        f"{uniad['L2_1s']:.2f}", f"{uniad['L2_2s']:.2f}", f"{uniad['L2_3s']:.2f}", f"{uniad['L2_avg']:.2f}",
        f"{uniad['collision_1s']:.2f}", f"{uniad['collision_2s']:.2f}", f"{uniad['collision_3s']:.2f}", f"{uniad['collision_avg']:.2f}"
    ))
    lines.append("\n-------------------------------------------------------------------")
    lines.append("STP-3 evaluation:")
    lines.append("{:<23} {:<22} {:<20}".format(*headers))
    lines.append("{:<15} {:<5} {:<5} {:<5} {:<8} {:<5} {:<5} {:<5} {:<5}".format("", *subs, *subs))
    lines.append("{:<15} {:<5} {:<5} {:<5} {:<8} {:<5} {:<5} {:<5} {:<5}".format(
        "DriveAgent",
        f"{stp3['L2_1s']:.2f}", f"{stp3['L2_2s']:.2f}", f"{stp3['L2_3s']:.2f}", f"{stp3['L2_avg']:.2f}",
        f"{stp3['collision_1s']:.2f}", f"{stp3['collision_2s']:.2f}", f"{stp3['collision_3s']:.2f}", f"{stp3['collision_avg']:.2f}"
    ))
    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    out_path = result_dir / 'per_sample_planning_metrics.jsonl'
    with out_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    planning = {
        'available': True,
        'count': len(rows),
        'subset_token_count': len(subset_tokens),
        'missing_pred_count': len(missing_pred),
        'missing_pred_examples': missing_pred[:10],
        'missing_gt_count': len(missing_gt),
        'missing_gt_examples': missing_gt[:10],
        'paths': {
            'eval_results_log': str(log_path),
            'per_sample': str(out_path),
            'pred_trajs_dict': str(pred_path),
        },
        'uniad': uniad,
        'stp3': stp3,
    }
    return out_path, planning


def _detector_pred_from_record(record: Dict[str, Any]) -> int:
    try:
        return int(record.get("detector_report", {}).get("combined", {}).get("pred_label", 0))
    except Exception:
        return 0


def _fault_group_from_record(record: Dict[str, Any]) -> str:
    faults = record.get("faults", []) or []
    if not faults:
        return "clean"
    return str((faults[0].get("impact", {}) or {}).get("group", "unknown"))


def _binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    def sd(a, b):
        return float(a) / float(b) if b else 0.0
    precision = sd(tp, tp + fp)
    recall = sd(tp, tp + fn)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": sd(2 * precision * recall, precision + recall),
        "false_positive_rate": sd(fp, fp + tn),
        "accuracy": sd(tp + tn, len(y_true)),
        "support": len(y_true),
    }


def _combined_int(record: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(record.get("detector_report", {}).get("combined", {}).get(key, default))
    except Exception:
        return int(default)


def _detector_pred_from_record(record: Dict[str, Any]) -> int:
    return _combined_int(record, "pred_label", 0)


def _raw_input_pred_from_record(record: Dict[str, Any]) -> int:
    return _combined_int(record, "raw_pred_label", _detector_pred_from_record(record))


def _critical_input_pred_from_record(record: Dict[str, Any]) -> int:
    return _combined_int(record, "critical_input_alarm", 0)


def _suspicious_input_pred_from_record(record: Dict[str, Any]) -> int:
    return _combined_int(record, "suspicious_input_alarm", 0)




def _flatten_numeric_dict(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Flatten nested numeric detector features for later small-model training."""
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric_dict(v, key))
    elif isinstance(obj, (list, tuple)):
        # Keep short numeric vectors; avoid exploding very long arrays.
        if len(obj) <= 32:
            for i, v in enumerate(obj):
                key = f"{prefix}.{i}" if prefix else str(i)
                out.update(_flatten_numeric_dict(v, key))
    elif isinstance(obj, (int, float)):
        try:
            fv = float(obj)
            if np.isfinite(fv):
                out[prefix] = fv
        except Exception:
            pass
    return out

def _critical_reasons_from_record(record: Dict[str, Any]) -> List[str]:
    try:
        return list(record.get("detector_report", {}).get("objects", {}).get("can_bus", {}).get("critical_reasons", []) or [])
    except Exception:
        return []


def _output_alarm_from_row(row: Dict[str, Any]) -> int:
    """Output-first E-stage alarm.

    This is deliberately based only on the current generated trajectory sanity report,
    not on clean reference metrics or GT deltas. The clean/faulty deltas are used only
    later for offline harmful/harmless labeling.
    """
    output_report = row.get('output_sanity_report', {}) or {}
    return int(
        int(output_report.get('pred_label', 0)) == 1
        or int(output_report.get('alarm_count', 0)) >= 1
    )


def _v19_patch_strong_exception_from_record(record: Dict[str, Any]) -> int:
    """Narrow S0-side fallback for output-first harmful monitoring.

    v13 uses E-stage output sanity as the primary harmful monitor. S0 input-side
    alarms are not used broadly, because v11 showed that many input anomalies are
    tolerated by the model. The only non-catastrophic S0 fallback kept here is a
    narrow patch-angle exception: `patch_angle_pair:strong_ge2`.

    Rationale from v11 post-hoc analysis:
      - output sanity alone: 52/53 main harmful, 15/99 harmless alarms;
      - output sanity + all patch_angle_pair:extreme: 53/53 harmful, 26/99 harmless;
      - output sanity + patch_angle_pair:strong_ge2: 53/53 harmful, 22/99 harmless.

    This is still not a scene-aware collision detector; it is a narrow placeholder
    for the one collision-dominant patch-angle harmful miss observed in v11.
    """
    reasons = _critical_reasons_from_record(record)
    if any(r.endswith(':nonfinite') for r in reasons):
        return 1
    if 'patch_angle_pair:strong_ge2' in reasons:
        return 1
    return 0


def summarize_harm(clean_path: Path, fault_path: Path, output_sanity_path: Optional[Path] = None) -> Dict[str, Any]:
    clean_rows_loaded = load_jsonl(clean_path)
    fault_rows_loaded = load_jsonl(fault_path)
    clean = {normalize_sample_token(str(r['sample_id'])): r for r in clean_rows_loaded if r.get('sample_id')}
    fault = {normalize_sample_token(str(r['sample_id'])): r for r in fault_rows_loaded if r.get('sample_id')}
    # v53: preserve inference/per-sample output order so previous-frame-only
    # temporal features can match sample-inline detector scoring. Do not sort by
    # token here; sorting breaks online parity for temporal feature sets.
    shared = []
    seen_shared = set()
    for r in fault_rows_loaded:
        sid0 = normalize_sample_token(str(r.get('sample_id', '')))
        if sid0 and sid0 in clean and sid0 not in seen_shared:
            shared.append(sid0)
            seen_shared.add(sid0)

    if output_sanity_path is None:
        cand = clean_path.parent / 'output_sanity_thresholds.json'
        output_sanity_path = cand if cand.exists() else None
    output_sanity_thresholds = None
    if output_sanity_path is not None and output_sanity_path.exists():
        output_sanity_thresholds = _load_json(output_sanity_path)

    rows = []
    for sid in shared:
        c, f = clean[sid], fault[sid]
        output_rep = detect_output_sanity(f, output_sanity_thresholds)
        rows.append({
            'sample_id': sid,
            'delta_l2_1s': f['l2_1s'] - c['l2_1s'],
            'delta_l2_2s': f['l2_2s'] - c['l2_2s'],
            'delta_l2_3s': f['l2_3s'] - c['l2_3s'],
            'delta_l2_avg': f['l2_avg'] - c['l2_avg'],
            'delta_collision_1s': f['collision_1s'] - c['collision_1s'],
            'delta_collision_2s': f['collision_2s'] - c['collision_2s'],
            'delta_collision_3s': f['collision_3s'] - c['collision_3s'],
            'delta_collision_avg': f['collision_avg'] - c['collision_avg'],
            'clean_l2_avg': c['l2_avg'],
            'fault_l2_avg': f['l2_avg'],
            'clean_collision_avg': c['collision_avg'],
            'fault_collision_avg': f['collision_avg'],
            'output_sanity_report': output_rep,
            'output_sanity_features': f.get('output_sanity_features', {}),
            'e_parse_features': f.get('e_parse_features', {}),
        })

    def mean(k):
        vals = [float(r[k]) for r in rows if np.isfinite(float(r[k]))]
        return float(np.mean(vals)) if vals else 0.0

    out = {
        'available': True,
        'version': VERSION,
        'design': 'v19 deployment-aware harmful monitor: keeps v18 all-online stage features, adds clean-row false-alarm accounting, previous-frame-only temporal features, and output-collapse decomposition.',
        'clean_ref': str(clean_path),
        'fault_ref': str(fault_path),
        'output_sanity_thresholds': str(output_sanity_path) if output_sanity_path else None,
        'shared_count': len(shared),
        'harm_thresholds': {
            'mild_l2_avg': DEFAULT_MILD_HARM_L2_THRESHOLD,
            'main_l2_avg': DEFAULT_HARM_L2_THRESHOLD,
            'collision_avg': DEFAULT_HARM_COLLISION_THRESHOLD,
            'strict_collision_avg': DEFAULT_STRICT_COLLISION_THRESHOLD,
        },
        'delta_l2_avg_mean': mean('delta_l2_avg'),
        'delta_collision_avg_mean': mean('delta_collision_avg'),
        'delta_l2_1s_mean': mean('delta_l2_1s'),
        'delta_l2_2s_mean': mean('delta_l2_2s'),
        'delta_l2_3s_mean': mean('delta_l2_3s'),
        'delta_collision_1s_mean': mean('delta_collision_1s'),
        'delta_collision_2s_mean': mean('delta_collision_2s'),
        'delta_collision_3s_mean': mean('delta_collision_3s'),
    }

    record_path = fault_path.parent / 'results' / 'detector_records.jsonl'
    if not record_path.exists():
        out['detector_harmful_metrics_available'] = False
        out['detector_harmful_metrics_reason'] = f'missing {record_path}'
        return out

    records = {normalize_sample_token(str(r.get('sample_id', ''))): r for r in load_jsonl(record_path) if r.get('sample_id')}
    clean_record_path = clean_path.parent / 'results' / 'detector_records.jsonl'
    clean_records = {normalize_sample_token(str(r.get('sample_id', ''))): r for r in load_jsonl(clean_record_path) if r.get('sample_id')} if clean_record_path.exists() else {}
    fault_behavior_lookup = v35_load_plan_behavior_lookup(fault_path.parent)
    clean_behavior_lookup = v35_load_plan_behavior_lookup(clean_path.parent)

    strategy_names = [
        'raw_input_alarm',
        'critical_input_only',
        'v11_p0_two_stage_harm_alarm',
        'e_output_sanity_only',
        'v19_patch_strong_exception',
        'v19_output_first_harm_alarm',
    ]
    y_raw_true: List[int] = []
    y_raw_pred: List[int] = []
    y_injected_main_true: List[int] = []
    y_injected_mild_true: List[int] = []
    y_injected_strict_true: List[int] = []
    injected_strategy_preds: Dict[str, List[int]] = {k: [] for k in strategy_names}
    group_stats: Dict[str, Dict[str, Any]] = {}
    attribution: Dict[str, Dict[str, int]] = {k: {'harmful_tp': 0, 'harmless_alarm': 0} for k in strategy_names}
    row_records: List[Dict[str, Any]] = []

    for row in rows:
        sid = row['sample_id']
        rec = records.get(sid)
        if rec is None:
            continue
        injected = int(bool(rec.get('faults')) and int(rec.get('is_positive_effective', rec.get('is_positive', 0))) == 1)
        raw_pred = _raw_input_pred_from_record(rec)
        critical_pred = _critical_input_pred_from_record(rec)
        suspicious_pred = _suspicious_input_pred_from_record(rec)
        output_pred = _output_alarm_from_row(row)
        narrow_exception = _v19_patch_strong_exception_from_record(rec)
        v11_p0_two_stage = int(critical_pred == 1 or (suspicious_pred == 1 and output_pred == 1))
        # v13 large-sample results showed the patch exception was a negative contributor.
        # v19 therefore keeps it as a separate diagnostic strategy, but the primary
        # output-first harmful alarm equals E-output sanity only.
        v19_output_first = int(output_pred == 1)

        strategy_pred_map = {
            'raw_input_alarm': raw_pred,
            'critical_input_only': critical_pred,
            'v11_p0_two_stage_harm_alarm': v11_p0_two_stage,
            'e_output_sanity_only': output_pred,
            'v19_patch_strong_exception': narrow_exception,
            'v19_output_first_harm_alarm': v19_output_first,
        }

        mild = int(injected and (row['delta_l2_avg'] >= DEFAULT_MILD_HARM_L2_THRESHOLD or row['delta_collision_avg'] >= DEFAULT_HARM_COLLISION_THRESHOLD))
        harmful = int(injected and (row['delta_l2_avg'] >= DEFAULT_HARM_L2_THRESHOLD or row['delta_collision_avg'] >= DEFAULT_HARM_COLLISION_THRESHOLD))
        strict = int(injected and (row['delta_l2_avg'] >= DEFAULT_HARM_L2_THRESHOLD or row['delta_collision_avg'] >= DEFAULT_STRICT_COLLISION_THRESHOLD))
        harmless_injected = int(injected and not harmful)

        # Raw-fault metrics: positives are effective injected faults; negatives are clean/uninjected samples.
        y_raw_true.append(injected)
        y_raw_pred.append(raw_pred)

        if injected:
            y_injected_main_true.append(harmful)
            y_injected_mild_true.append(mild)
            y_injected_strict_true.append(strict)
            for name, pred in strategy_pred_map.items():
                injected_strategy_preds[name].append(int(pred))
                if harmful and pred:
                    attribution[name]['harmful_tp'] += 1
                if harmless_injected and pred:
                    attribution[name]['harmless_alarm'] += 1

        g = _fault_group_from_record(rec)
        gs = group_stats.setdefault(g, {
            'samples': 0, 'injected': 0,
            'mild_harmful': 0, 'main_harmful': 0, 'strict_harmful': 0,
            'harmless_injected': 0,
            'strategies': {name: {
                'pred': 0,
                'main_tp': 0,
                'main_fn': 0,
                'harmless_alarm': 0,
                'harmless_pass': 0,
            } for name in strategy_names},
        })
        gs['samples'] += 1
        gs['injected'] += injected
        gs['mild_harmful'] += mild
        gs['main_harmful'] += harmful
        gs['strict_harmful'] += strict
        gs['harmless_injected'] += harmless_injected
        for name, pred in strategy_pred_map.items():
            pred = int(pred)
            gs['strategies'][name]['pred'] += pred
            gs['strategies'][name]['main_tp'] += int(harmful and pred)
            gs['strategies'][name]['main_fn'] += int(harmful and not pred)
            gs['strategies'][name]['harmless_alarm'] += int(harmless_injected and pred)
            gs['strategies'][name]['harmless_pass'] += int(harmless_injected and not pred)

        row_records.append({
            **{k: v for k, v in row.items() if k != 'output_sanity_report'},
            'sample_id': sid,
            'row_source': 'experiment_injected' if injected else 'experiment_no_fault',
            'is_clean': 0,
            'is_experiment_row': 1,
            'is_fault_selected': int(rec.get('is_fault_selected', rec.get('is_positive', 0)) or 0),
            'fault_spec_count': int(rec.get('fault_spec_count', 0) or 0),
            'fault_attempted_count': int(rec.get('fault_attempted_count', 0) or 0),
            'fault_applied_count': int(rec.get('fault_applied_count', 0) or 0),
            'fault_effective_count': int(rec.get('fault_effective_count', 0) or 0),
            'is_effective_injected': injected,
            'injected_effective': injected,
            'fault_target': 'can_bus' if int(rec.get('is_fault_selected', rec.get('is_positive', 0)) or 0) else 'none',
            'fault_group': g if injected else 'experiment_no_fault',
            'mild_harmful': mild,
            'main_harmful': harmful,
            'strict_harmful': strict,
            'raw_input_alarm': raw_pred,
            'critical_input_alarm': critical_pred,
            'suspicious_input_alarm': suspicious_pred,
            'e_output_sanity_alarm': output_pred,
            'v11_p0_two_stage_harm_alarm': v11_p0_two_stage,
            'v19_patch_strong_exception': narrow_exception,
            'v19_output_first_harm_alarm': v19_output_first,
            'critical_reasons': _critical_reasons_from_record(rec),
            'output_sanity_report': row.get('output_sanity_report', {}),
            'e_parse_features': row.get('e_parse_features', {}),
            'object_vectors': rec.get('object_vectors', {}),
            'faults': rec.get('faults', []),
            'stage_summaries': rec.get('stage_summaries', {}),
            'scene_context_summary': rec.get('scene_context_summary', {}),
            'online_context_features': rec.get('online_context_features', {}),
            'v26_online_head_features': rec.get('v26_online_head_features', {}),
            'v26_agent_features': row.get('v26_agent_features', {}),
            'v26_ego_pose_features': row.get('v26_ego_pose_features', {}),
            'v35_behavior_features': fault_behavior_lookup.get(sid, {'feat__behout_feature_found': 0.0}),
            'stage_feature_flat': flatten_stage_summary_for_training(rec.get('stage_summaries', {})),
            's0_feature_flat': _flatten_numeric_dict(rec.get('features', {})),
        })

    def _harmful_objective(true_labels: List[int], pred_labels: List[int]) -> Dict[str, Any]:
        m = _binary_metrics(true_labels, pred_labels)
        # In harmful objective, false_positive_rate equals harmless injected alarm rate.
        m['harmless_injected_alarm_rate'] = m['false_positive_rate']
        m['harmless_injected_pass_rate'] = 1.0 - m['false_positive_rate'] if (m.get('tn', 0) + m.get('fp', 0)) else 0.0
        return m

    strategy_metrics = {}
    for name, pred in injected_strategy_preds.items():
        strategy_metrics[name] = {
            'mild_harmful_objective': _harmful_objective(y_injected_mild_true, pred),
            'main_harmful_objective': _harmful_objective(y_injected_main_true, pred),
            'strict_collision_harmful_objective': _harmful_objective(y_injected_strict_true, pred),
        }

    for _, gs in group_stats.items():
        for name, ss in gs['strategies'].items():
            ss['main_harmful_recall'] = float(ss['main_tp']) / float(gs['main_harmful']) if gs['main_harmful'] else None
            ss['harmless_alarm_rate'] = float(ss['harmless_alarm']) / float(gs['harmless_injected']) if gs['harmless_injected'] else None
            ss['harmless_pass_rate'] = float(ss['harmless_pass']) / float(gs['harmless_injected']) if gs['harmless_injected'] else None

    # Add clean / uninjected rows for deployment false-alarm accounting.
    # These rows are online-feature-only negatives. They are not used to define
    # harmful labels or future-looking deltas.
    clean_row_records: List[Dict[str, Any]] = []
    for sid, cmetric in sorted(clean.items()):
        rec = clean_records.get(sid, {}) or {}
        output_rep = detect_output_sanity(cmetric, output_sanity_thresholds)
        output_pred = _output_alarm_from_row({'output_sanity_report': output_rep, 'output_sanity_features': cmetric.get('output_sanity_features', {})})
        raw_pred = _raw_input_pred_from_record(rec) if rec else 0
        critical_pred = _critical_input_pred_from_record(rec) if rec else 0
        suspicious_pred = _suspicious_input_pred_from_record(rec) if rec else 0
        clean_row_records.append({
            'sample_id': sid,
            'row_source': 'calibration_clean',
            'is_clean': 1,
            'is_experiment_row': 0,
            'is_fault_selected': 0,
            'fault_spec_count': 0,
            'fault_attempted_count': 0,
            'fault_applied_count': 0,
            'fault_effective_count': 0,
            'is_effective_injected': 0,
            'injected_effective': 0,
            'fault_target': 'clean',
            'fault_group': 'calibration_clean',
            'mild_harmful': 0,
            'main_harmful': 0,
            'strict_harmful': 0,
            'delta_l2_avg': 0.0,
            'delta_collision_avg': 0.0,
            'raw_input_alarm': raw_pred,
            'critical_input_alarm': critical_pred,
            'suspicious_input_alarm': suspicious_pred,
            'e_output_sanity_alarm': output_pred,
            'v11_p0_two_stage_harm_alarm': int(critical_pred == 1 or (suspicious_pred == 1 and output_pred == 1)),
            'v19_patch_strong_exception': _v19_patch_strong_exception_from_record(rec) if rec else 0,
            'v19_output_first_harm_alarm': int(output_pred == 1),
            'critical_reasons': _critical_reasons_from_record(rec) if rec else [],
            'output_sanity_features': cmetric.get('output_sanity_features', {}),
            'output_sanity_report': output_rep,
            'e_parse_features': cmetric.get('e_parse_features', {}),
            'object_vectors': rec.get('object_vectors', {}) if rec else {},
            'faults': [],
            'stage_summaries': rec.get('stage_summaries', {}) if rec else {},
            'scene_context_summary': rec.get('scene_context_summary', {}) if rec else {},
            'online_context_features': rec.get('online_context_features', {}) if rec else {},
            'v26_online_head_features': rec.get('v26_online_head_features', {}) if rec else {},
            'v26_agent_features': cmetric.get('v26_agent_features', {}),
            'v26_ego_pose_features': cmetric.get('v26_ego_pose_features', {}),
            'v35_behavior_features': clean_behavior_lookup.get(sid, {'feat__behout_feature_found': 0.0}),
            'stage_feature_flat': flatten_stage_summary_for_training(rec.get('stage_summaries', {}) if rec else {}),
            's0_feature_flat': _flatten_numeric_dict(rec.get('features', {}) if rec else {}),
        })
    row_records.extend(clean_row_records)
    for i, rr in enumerate(row_records):
        rr['row_index'] = i

    # Add v19 temporal context before writing training/matrix files.
    add_v19_temporal_context_features(row_records)

    # Deployment-aware false-alarm summary.
    clean_total = len(clean_row_records)
    deployment_summary: Dict[str, Any] = {'clean_total': clean_total, 'strategies': {}}
    for name in strategy_names:
        clean_alarm = int(sum(1 for rr in clean_row_records if int(rr.get(name, 0) or 0) == 1))
        met = strategy_metrics.get(name, {}).get('main_harmful_objective', {})
        harmless_alarm = int(met.get('harmless_alarm', 0) or 0)
        harmless_total = int(met.get('harmless_total', 0) or 0)
        deployment_negative_total = harmless_total + clean_total
        deployment_false_alarm = harmless_alarm + clean_alarm
        deployment_summary['strategies'][name] = {
            'harmless_injected_alarm': harmless_alarm,
            'harmless_injected_total': harmless_total,
            'clean_alarm': clean_alarm,
            'clean_total': clean_total,
            'clean_false_alarm_rate': float(clean_alarm) / float(clean_total) if clean_total else None,
            'deployment_false_alarm': deployment_false_alarm,
            'deployment_negative_total': deployment_negative_total,
            'deployment_false_alarm_rate': float(deployment_false_alarm) / float(deployment_negative_total) if deployment_negative_total else None,
        }

    # Save detailed rows and attributions next to the per-sample metrics.
    detail_path = fault_path.parent / 'harmfulness_rows_v26.jsonl'
    with detail_path.open('w', encoding='utf-8') as f:
        for rr in row_records:
            f.write(json.dumps(rr, ensure_ascii=False) + '\n')

    training_path = fault_path.parent / 'stage_training_rows_v26.jsonl'
    with training_path.open('w', encoding='utf-8') as f:
        for rr in row_records:
            f.write(json.dumps({
                'row_index': rr.get('row_index'),
                'sample_id': rr.get('sample_id'),
                'row_source': rr.get('row_source'),
                'is_experiment_row': rr.get('is_experiment_row', 0),
                'is_fault_selected': rr.get('is_fault_selected', 0),
                'fault_spec_count': rr.get('fault_spec_count', 0),
                'fault_attempted_count': rr.get('fault_attempted_count', 0),
                'fault_applied_count': rr.get('fault_applied_count', 0),
                'fault_effective_count': rr.get('fault_effective_count', 0),
                'is_clean': rr.get('is_clean', 0),
                'is_effective_injected': rr.get('is_effective_injected', rr.get('injected_effective')),
                'fault_target': rr.get('fault_target', 'can_bus'),
                'injected_effective': rr.get('injected_effective'),
                'fault_group': rr.get('fault_group'),
                'fault_scope': rr.get('fault_scope', 'no_fault'),
                's0_fault_spec_count': rr.get('s0_fault_spec_count', 0),
                's1_fault_spec_count': rr.get('s1_fault_spec_count', 0),
                's0_fault_attempted_count': rr.get('s0_fault_attempted_count', 0),
                's0_fault_effective_count': rr.get('s0_fault_effective_count', 0),
                's1_fault_attempted_count': rr.get('s1_fault_attempted_count', 0),
                's1_fault_effective_count': rr.get('s1_fault_effective_count', 0),
                'v35_s1_summary': rr.get('v35_s1_summary', {}),
                'faults': rr.get('faults', []),
                'mild_harmful': rr.get('mild_harmful'),
                'main_harmful': rr.get('main_harmful'),
                'strict_harmful': rr.get('strict_harmful'),
                'delta_l2_avg': rr.get('delta_l2_avg'),
                'delta_collision_avg': rr.get('delta_collision_avg'),
                'raw_input_alarm': rr.get('raw_input_alarm'),
                'e_output_sanity_alarm': rr.get('e_output_sanity_alarm'),
                'v19_output_first_harm_alarm': rr.get('v19_output_first_harm_alarm'),
                'output_sanity_features': rr.get('output_sanity_features', {}),
                'output_sanity_report': rr.get('output_sanity_report', {}),
                'e_parse_features': rr.get('e_parse_features', {}),
                'temporal_context_features': rr.get('temporal_context_features', {}),
                'stage_feature_flat': rr.get('stage_feature_flat', {}),
                's0_feature_flat': rr.get('s0_feature_flat', {}),
                'scene_context_summary': rr.get('scene_context_summary', {}),
                'isolation_info': rr.get('isolation_info', {}),
                'critical_reasons': rr.get('critical_reasons', []),
                'v35_behavior_features': rr.get('v35_behavior_features', {}),
            }, ensure_ascii=False) + '\n')
    schema_path = fault_path.parent / 'stage_training_schema_v26.json'
    schema = {
        'version': VERSION,
        'description': 'Per-sample rows for later lightweight harmful-fault predictor training. Harm labels are for offline training/evaluation only, not online detector input.',
        'label_columns': ['mild_harmful', 'main_harmful', 'strict_harmful'],
        'delta_columns': ['delta_l2_avg', 'delta_collision_avg'],
        'feature_columns': ['output_sanity_features', 'e_parse_features', 'previous-frame temporal_context_features', 'stage_feature_flat', 's0_feature_flat', 'scene_context_summary', 'online metadata'],
        'data_leakage_note': 'Do not use delta_* or harmful labels as online detector inputs. Previous-frame temporal features use past rows only; no next-frame/future information is used.',
    }
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding='utf-8')

    matrix_csv_path = fault_path.parent / 'stage_feature_matrix_v26.csv'
    matrix_jsonl_path = fault_path.parent / 'stage_feature_matrix_v26.jsonl'
    matrix_meta = write_training_feature_matrix(row_records, matrix_csv_path, matrix_jsonl_path)
    matrix_meta_path = fault_path.parent / 'stage_feature_matrix_meta_v26.json'
    matrix_meta_path.write_text(json.dumps({
        'version': VERSION,
        'description': 'Flat feature matrix for offline lightweight harmful-classifier training. Do not use delta_* or label columns as online detector inputs.',
        **matrix_meta,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[{VERSION}] wrote {matrix_csv_path}")
    print(f"[{VERSION}] wrote {matrix_jsonl_path}")
    print(f"[{VERSION}] wrote {matrix_meta_path}")

    saved_files = {
        'harmfulness_rows_v26': str(detail_path),
        'stage_training_rows_v26': str(training_path),
        'stage_training_schema_v26': str(schema_path),
        'stage_feature_matrix_v26_csv': str(matrix_csv_path),
        'stage_feature_matrix_v26_jsonl': str(matrix_jsonl_path),
        'stage_feature_matrix_meta_v26': str(matrix_meta_path),
    }
    for strategy_name in strategy_names:
        safe = strategy_name.replace('/', '_')
        fn_path = fault_path.parent / f'harmful_fn_records_{safe}.jsonl'
        ha_path = fault_path.parent / f'harmless_alarm_records_{safe}.jsonl'
        with fn_path.open('w', encoding='utf-8') as f:
            for rr in row_records:
                if rr.get('injected_effective') and rr.get('main_harmful') and not rr.get(strategy_name):
                    f.write(json.dumps(rr, ensure_ascii=False) + '\n')
        with ha_path.open('w', encoding='utf-8') as f:
            for rr in row_records:
                if rr.get('injected_effective') and not rr.get('main_harmful') and rr.get(strategy_name):
                    f.write(json.dumps(rr, ensure_ascii=False) + '\n')
        saved_files[f'harmful_fn_records_{safe}'] = str(fn_path)
        saved_files[f'harmless_alarm_records_{safe}'] = str(ha_path)

    attribution_path = fault_path.parent / 'rule_attribution_summary_v26.json'
    attribution_path.write_text(json.dumps(attribution, ensure_ascii=False, indent=2), encoding='utf-8')
    saved_files['rule_attribution_summary_v26'] = str(attribution_path)

    out.update({
        'detector_metrics_available': True,
        'raw_fault_detection': {
            'definition': 'positive=effective injected fault; negative=clean/uninjected sample',
            'metrics': _binary_metrics(y_raw_true, y_raw_pred),
        },
        'harmful_fault_detection': {
            'definition': 'positive=harmful injected fault; negative=harmless injected fault; clean samples excluded',
            'primary_strategy': 'e_output_sanity_only',
            'strategies': strategy_metrics,
        },
        'by_group_harmfulness': group_stats,
        'rule_attribution_summary': attribution,
        'deployment_false_alarm': deployment_summary,
        'saved_files': saved_files,
    })
    return out


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()

    # Integrated post-eval mode. This replaces the old separate post_eval*.py file.
    parser.add_argument("--post-eval-only", action="store_true")
    parser.add_argument("--result-dir", type=str, default="")
    parser.add_argument("--clean-ref", type=str, default="")
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--output-sanity-json", type=str, default="")

    # Optional automatic post-eval after normal calibration/experiment inference.
    parser.add_argument("--run-post-eval", action="store_true")

    # v53: sample-inline detector. This runs selected v35.1 detector artifacts
    # immediately after each sample forward, before the next sample starts.
    parser.add_argument("--v53-inline-detector", action="store_true",
                        help="Run selected v35.1 detector artifacts inside the per-sample inference loop.")
    parser.add_argument("--v53-detector-artifact", type=str,
                        default="/root/autodl-tmp/DriveVLA/pack_output_analysis/v35_1_exported_selected_detectors_for_reuse.tar.gz")
    parser.add_argument("--v53-inline-detectors", type=str, default="",
                        help="Optional comma-separated exact detector names from the artifact. Empty means all selected detectors.")
    parser.add_argument("--v53-inline-analysis-metrics", action="store_true", default=True,
                        help="Emit analysis-only per-sample L2/collision against GT in inline detector record. Never used as detector input.")
    parser.add_argument("--v53-disable-inline-analysis-metrics", dest="v53_inline_analysis_metrics", action="store_false")
    parser.add_argument("--v53-inline-strict", action="store_true", default=True,
                        help="Fail immediately if inline detector scoring fails for any sample.")
    parser.add_argument("--v53-inline-nonstrict", dest="v53_inline_strict", action="store_false")

    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume-from-output", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-uniad-pth", action="store_true")
    parser.add_argument("--subset-ids", type=str, default="")
    parser.add_argument("--mode", type=str, choices=["calibrate", "experiment"], default=None)
    parser.add_argument("--target-mode", type=str, default="A1_can_bus", choices=list(S0_TARGET_MODE_TO_TARGETS.keys()))
    parser.add_argument("--experiment-seed", type=int, default=1234)
    parser.add_argument("--positive-min", type=int, default=800)
    parser.add_argument("--positive-max", type=int, default=900)
    parser.add_argument("--calibration-json", type=str, default=None)
    parser.add_argument("--cardinality-weights", type=str, default=CARDINALITY_DEFAULT)
    parser.add_argument("--bit-op-weights", type=str, default=BIT_OP_DEFAULT)
    parser.add_argument("--mixed-fault-scope-weights", type=str, default="S0_only:40,S1_only:30,S0S1_mixed:30",
                        help="v32 mixed mode scope weights among S0_only/S1_only/S0S1_mixed for injected samples.")
    parser.add_argument("--fault-event-count-weights", type=str, default="1:6,2:3,3:1",
                        help="Number of fault events per selected S0 or S1 side.")
    parser.add_argument("--strict-single-fault", action="store_true",
                        help="v35 corrected口径: each injected sample has exactly one fault event, located either in S0 or S1; S0S1_mixed and multi-event samples are forbidden.")
    parser.add_argument("--s1-target-specs", type=str, default=DEFAULT_V32_S1_TARGET_SPECS,
                        help="Comma-separated logical_name=module_suffix list for S1 activation hooks.")
    parser.add_argument("--s1-target-policy", type=str, default="weighted", choices=["weighted", "primary_only", "uniform_all"],
                        help="S1 target sampling policy. weighted uses --s1-primary-weight plus --s1-broad-target-weights.")
    parser.add_argument("--s1-primary-target", type=str, default=DEFAULT_V32_S1_PRIMARY_TARGET)
    parser.add_argument("--s1-primary-weight", type=int, default=70)
    parser.add_argument("--s1-broad-target-weights", type=str, default=DEFAULT_V32_S1_BROAD_TARGET_WEIGHTS)
    parser.add_argument("--enable-s1-hooks", action="store_true", help="Register S1 hooks even outside A1_can_bus_s1_mixed; intended for debugging.")
    parser.add_argument("--c-target-specs", type=str, default=DEFAULT_V44_C_TARGET_SPECS, help="Comma-separated C target specs name=suffix for v44 C/D/E OOD using v35.1 model hooks.")
    parser.add_argument("--d-target-specs", type=str, default=DEFAULT_V44_D_TARGET_SPECS, help="Comma-separated D target specs name=suffix for v44 C/D/E OOD using v35.1 model hooks.")
    parser.add_argument("--c-target-weights", type=str, default=DEFAULT_V44_C_TARGET_WEIGHTS, help="Weighted C target sampling, name:weight comma format.")
    parser.add_argument("--d-target-weights", type=str, default=DEFAULT_V44_D_TARGET_WEIGHTS, help="Weighted D target sampling, name:weight comma format.")
    parser.add_argument("--e-target-weights", type=str, default=DEFAULT_V44_E_TARGET_WEIGHTS, help="Weighted E output perturbation sampling, name:weight comma format.")
    parser.add_argument("--enable-cde-hooks", action="store_true", help="Register C/D hooks even outside CDE_ood target modes; intended for readiness/debugging.")
    parser.add_argument("--can-bus-mlp-hook-suffix", type=str, default=DEFAULT_CAN_BUS_MLP_HOOK_SUFFIX)
    parser.add_argument("--stage-hook-specs", type=str, default=DEFAULT_STAGE_HOOK_SPECS,
                        help="Comma-separated name=suffix stage hooks. Missing hooks are non-fatal unless --stage-hook-strict is set.")
    parser.add_argument("--stage-hook-strict", action="store_true",
                        help="Fail if any requested stage hook cannot be registered.")
    parser.add_argument("--disable-sample-isolation", dest="sample_isolation", action="store_false",
                        help="Disable v22 per-sample batch cloning and can_bus snapshot/restore. Not recommended.")
    parser.set_defaults(sample_isolation=True)
    parser.add_argument("--clear-cuda-cache-between-samples", action="store_true",
                        help="Synchronize and empty CUDA cache between samples. Slower; useful for debugging only.")
    parser.add_argument("--reset-model-state-between-samples", dest="reset_model_state_between_samples", action="store_true",
                        help="Reset model temporal runtime state before and after every sample. Default for v22.")
    parser.add_argument("--disable-model-state-reset", dest="reset_model_state_between_samples", action="store_false",
                        help="Disable v22 temporal-state isolation. Intended only for ablation/debugging.")
    parser.set_defaults(reset_model_state_between_samples=True)
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("V50_MAX_NEW_TOKENS", "192")),
                        help="Generation cap. v50.3 defaults to 192 to avoid 512-token runaway decoding; set V50_MAX_NEW_TOKENS=512 to reproduce the old cap.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--zero-stage", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", -1)))
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", -1)))
    parser.add_argument("--dist-url", default="env://", type=str)
    args = parser.parse_args()

    if args.post_eval_only:
        raise ValueError("v53 disables legacy --post-eval-only because legacy harmful labels are not approved. Use tools_scan/export_v53_online_matrices.py for posthoc harmful/severe_harmful evaluation.")
        if not args.result_dir:
            raise ValueError("--post-eval-only requires --result-dir")
        result_dir = Path(args.result_dir)
        log_path = Path(args.log_path) if args.log_path else (result_dir / "eval_results.log")
        parse_info = build_pred_files_from_plan_conv(result_dir)
        per_sample_path, planning = compute_subset_metrics(result_dir, log_path)
        planning["parser"] = parse_info
        (result_dir / "planning_metrics.json").write_text(json.dumps(planning, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.clean_ref:
            output_thr = build_output_sanity_thresholds(load_jsonl(per_sample_path))
            (result_dir / "output_sanity_thresholds.json").write_text(json.dumps(output_thr, ensure_ascii=False, indent=2), encoding="utf-8")
        harm = {"available": False, "reason": "no_clean_ref"}
        if args.clean_ref:
            clean_ref = Path(args.clean_ref)
            if clean_ref.exists():
                harm = summarize_harm(clean_ref, per_sample_path, (Path(args.output_sanity_json) if args.output_sanity_json else None))
            else:
                harm = {"available": False, "reason": f"missing_clean_ref:{clean_ref}"}
        (result_dir / "harmfulness_summary.json").write_text(json.dumps(harm, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", result_dir / "pred_trajs_dict.json")
        print("wrote", result_dir / "pred_trajs_multi_modal_dict.json")
        print("wrote", log_path)
        print("wrote", result_dir / "planning_metrics.json")
        print("wrote", per_sample_path)
        print("wrote", result_dir / "harmfulness_summary.json")
        return

    if not args.model_path:
        raise ValueError("normal inference mode requires --model-path")
    if not args.subset_ids:
        raise ValueError("normal inference mode requires --subset-ids")
    if args.mode is None:
        raise ValueError("normal inference mode requires --mode calibrate|experiment")

    if args.local_rank != -1:
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(args.local_rank)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(args.world_size)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

    if args.resume_from_output is not None:
        args.output = args.resume_from_output
    if args.model_base is None:
        args.model_base = args.model_path.split("checkpoints/")[-1].replace("/", "_")
        if args.output is None:
            args.output = f"output/{args.model_base}/{time.strftime('%Y%m%d_%H%M%S')}/plan_conv.json"

    inference_planning_oriented_vlm(args)

    if args.local_rank in [-1, 0]:
        print(f">>> Inference results saved to {args.output}")
        if args.run_post_eval:
            raise ValueError("v53 disables legacy --run-post-eval because legacy harmful labels are not approved. Use tools_scan/export_v53_online_matrices.py for posthoc harmful/severe_harmful evaluation.")
            result_dir = Path(args.output).resolve().parent
            log_path = result_dir / "eval_results.log"
            parse_info = build_pred_files_from_plan_conv(result_dir)
            per_sample_path, planning = compute_subset_metrics(result_dir, log_path)
            planning["parser"] = parse_info
            (result_dir / "planning_metrics.json").write_text(json.dumps(planning, ensure_ascii=False, indent=2), encoding="utf-8")
            if not args.clean_ref:
                output_thr = build_output_sanity_thresholds(load_jsonl(per_sample_path))
                (result_dir / "output_sanity_thresholds.json").write_text(json.dumps(output_thr, ensure_ascii=False, indent=2), encoding="utf-8")
            harm = {"available": False, "reason": "no_clean_ref"}
            if args.clean_ref:
                clean_ref = Path(args.clean_ref)
                if clean_ref.exists():
                    harm = summarize_harm(clean_ref, per_sample_path, (Path(args.output_sanity_json) if args.output_sanity_json else None))
                else:
                    harm = {"available": False, "reason": f"missing_clean_ref:{clean_ref}"}
            (result_dir / "harmfulness_summary.json").write_text(json.dumps(harm, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f">>> Post-eval results saved to {result_dir}")


if __name__ == "__main__":
    main()
