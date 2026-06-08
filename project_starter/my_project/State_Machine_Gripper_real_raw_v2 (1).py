#!/usr/bin/env python3
"""
@file state_machine.py
@brief SEARCHING / TRACKING / GRASP state machine for the Rizon4s (Titania)
       mouse-chasing project.

  Reads (from the unified CV, mouse_pose_tail_stream_real_1.py):
    opensai::perception::mouse_center_world         (Vector3d or []) - mouse center
    opensai::perception::mouse_tailbase_world       (Vector3d or []) - mouse tail base
    opensai::controllers::Titania::cartesian_controller::cartesian_task::current_position
                                                    (Vector3d) - current control point position
  Writes:
    opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_position
                                                    (Vector3d) - EE position goal
    opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_orientation
                                                    (Matrix3d) - EE orientation goal

CV pipeline key usage:
    mouse_center_world   -> SEARCHING / TRACKING / LIFT (detected?, overhead?, tracking goal)
    mouse_tailbase_world -> PRE_GRASP / GRASP (grasp target)

NOTE on coordinate frames (Titania real robot):
  - xml config: <motionForceTask linkName="flange">
                <compliantFrame xyz="0 0 0.15" rpy="0 0 0" />
  - So current_position = flange origin + 0.15m along flange local z axis
  - Camera offset from sim controller.cpp: [0.074, -0.01, 0.136] relative to flange origin
  - Camera offset relative to control point: [0.074, -0.01, 0.136 - 0.15] = [0.074, -0.01, -0.014]
  - GRASP z target must also account for the 0.15m compliantFrame offset

State graph:
    SEARCHING --mouse detected--> TRACKING --held overhead 3s--> GRASP
       ^                            |                              |
       |    <--mouse lost-----------+                              |
       |    <--grasp done OR fail (mouse strayed / lost)----------+

Launch order: redis-server -> opensai controller -> this script
(the unified CV, mouse_pose_tail_stream_real_1.py, is auto-launched from here).
"""

import atexit
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import redis

# ===================== Debug flags =====================
DEBUG_LOCK_SEARCHING = False

# ===================== Loop rate =====================
LOOP_HZ = 100

# ===================== Redis setup =====================
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# --- CV pipeline keys ---
# Mouse center: used in SEARCHING / TRACKING / LIFT.
PERCEPTION_RAW_KEY      = "opensai::perception::mouse_center_world"
# Tail base: used in PRE_GRASP / GRASP to target the grasp point.
PERCEPTION_TAIL_KEY     = "opensai::perception::mouse_tailbase_world"

# --- Titania real robot opensai keys ---
EE_POS_KEY         = "opensai::controllers::Titania::cartesian_controller::cartesian_task::current_position"
EE_DESIRED_POS_KEY = "opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_position"
EE_DESIRED_ORI_KEY = "opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_orientation"

# --- F/T sensor (for force-based grasp verification) ---
EE_TCP_FORCE_KEY   = "opensai::sensors::Titania::ft_sensor::tcp_force"

# --- Joint task (for null-space posture bias) ---
JOINT_POSITIONS_KEY    = "opensai::sensors::Titania::joint_positions"
JOINT_TASK_GOAL_KEY    = "opensai::controllers::Titania::cartesian_controller::joint_task::goal_position"

ACTIVE_CONTROLLER_KEY  = "opensai::controllers::Titania::active_controller_name"
OTG_LINEAR_VEL_KEY    = "opensai::controllers::Titania::cartesian_controller::cartesian_task::otg_max_linear_velocity"


GRIPPER_PARAMETERS_KEY    = "opensai::commands::Titania::gripper::parameters"
GRIPPER_CURRENT_WIDTH_KEY = "opensai::sensors::Titania::gripper::width"
GRIPPER_GRASP_FORCE_KEY   = "opensai::sensors::Titania::gripper::grasp_force"
GRIPPER_OPEN_PARAMS    = (0.10, 0.10, 10 ) # width m, speed m/s, force N
GRIPPER_CLOSE_PARAMS   = (0.0,0.10,60.0) # close fully, 3N grasp force
GRIPPER_SETTLE_TIME    = 1.5                  # s, wait after final close before exit


GRASP_FORCE_THRESHOLD = 2.5       # N, |grasp_force| -> object held
                                 
GRASP_FORCE_SAMPLES   = 10        # number of samples to average (~0.1s)

# OTG velocity values
OTG_LINEAR_VEL_NORMAL = 0.3 #m/s
OTG_LINEAR_VEL_GRASP  = 0.25   # m/s, slow descent during grasp


# ---- Serialization helpers ----
def read_vector(key):
    return np.array(json.loads(redis_client.get(key)), dtype=float).flatten()

def write_vector(key, vec):
    redis_client.set(key, json.dumps([float(x) for x in vec]))

def read_matrix(key):
    return np.array(json.loads(redis_client.get(key)), dtype=float)

def write_matrix(key, mat):
    redis_client.set(key, json.dumps([[float(x) for x in row] for row in mat]))

def write_gripper_parameters(width, speed, force):
    """Custom string format expected by driver: 'w s f' (12 decimals, spaces)."""
    s = f"{float(width):.12f} {float(speed):.12f} {float(force):.12f}"
    redis_client.set(GRIPPER_PARAMETERS_KEY, s)


def read_gripper_width():
    """Return current gripper opening width in meters (driver writes a plain
    float string). Returns None on failure so callers can fall back gracefully."""
    try:
        raw = redis_client.get(GRIPPER_CURRENT_WIDTH_KEY)
        if raw is None:
            return None
        return float(raw)
    except Exception:
        return None


def read_tcp_force():
    """Return TCP force [fx, fy, fz] in N, or None on error. Logged for
    debugging; the actual grasp decision uses gripper grasp_force instead
    (pose-dependent gravity-comp residuals on TCP force are larger than the
    mouse weight signal, so it can't be used for grasp detection)."""
    try:
        return read_vector(EE_TCP_FORCE_KEY)
    except Exception:
        return None


def sample_tcp_force_mean(n_samples=GRASP_FORCE_SAMPLES, dt=0.01):
    """Average n samples of TCP force. Returns 3-vec or None.
    Blocks for ~n_samples*dt seconds. Used for info/debug logging only."""
    forces = []
    for _ in range(n_samples):
        f = read_tcp_force()
        if f is not None:
            forces.append(f)
        time.sleep(dt)
    if not forces:
        return None
    return np.mean(np.array(forces), axis=0)


def read_gripper_grasp_force():
    """Return the gripper's current grasp force in N. Driver writes a plain
    float string. Returns None on failure."""
    try:
        raw = redis_client.get(GRIPPER_GRASP_FORCE_KEY)
        if raw is None:
            return None
        return float(raw)
    except Exception:
        return None


def sample_gripper_grasp_force_mean(n_samples=GRASP_FORCE_SAMPLES, dt=0.01):
    """Average n samples of gripper grasp_force. Returns float or None.
    Blocks for ~n_samples*dt seconds."""
    forces = []
    for _ in range(n_samples):
        f = read_gripper_grasp_force()
        if f is not None:
            forces.append(f)
        time.sleep(dt)
    if not forces:
        return None
    return float(np.mean(forces))


# ---- CV perception helpers ----
def read_perception_raw():
    """Unified CV (mouse_pose_tail_stream_real_1.py): mouse center.
    Used in SEARCHING / TRACKING / LIFT.
    Returns (detected: bool, pos: np.array or None).
    CV z is ignored — mouse is treated as lying on the z=0 plane."""
    try:
        val = json.loads(redis_client.get(PERCEPTION_RAW_KEY))
        if isinstance(val, list) and len(val) == 3:
            pos = np.array(val, dtype=float)
            pos[2] = 0.0          # <-- ignore CV z entirely
            return True, pos
        return False, None
    except Exception:
        return False, None


def read_perception_tail():
    """Unified CV (mouse_pose_tail_stream_real_1.py): mouse tail base.
    Used in PRE_GRASP / GRASP. Returns (detected, pos or None).
    Key is unset/empty-list when the YOLO pose model has no valid detection.
    CV z is ignored — tail is treated as lying on the z=0 plane."""
    try:
        raw = redis_client.get(PERCEPTION_TAIL_KEY)
        if raw is None:
            return False, None
        val = json.loads(raw)
        if isinstance(val, list) and len(val) == 3:
            pos = np.array(val, dtype=float)
            pos[2] = 0.0
            return True, pos
        return False, None
    except Exception:
        return False, None




# ===================== Control point -> Camera conversion =====================
# compliantFrame offset from flange origin: xyz="0 0 0.15"
# So current_position = flange + 0.15m along flange local z
#
# Camera offset from flange origin (sim controller.cpp): [0.074, -0.01, 0.136]
# Camera offset from control point: [0.074, -0.01, 0.136 - 0.15] = [0.074, -0.01, -0.014]
#
# To get camera world position:
#   camera_world = control_point_world + R_flange @ CONTROL_POINT_TO_CAMERA_LOCAL
COMPLIANT_FRAME_Z      = 0.15
CONTROL_POINT_TO_CAMERA_LOCAL = np.array([0.074, -0.01, 0.136 - COMPLIANT_FRAME_Z])
# = np.array([0.074, -0.01, -0.014])

def get_camera_world_pos(control_point_pos):
    """Convert control point world position to camera world position."""
    try:
            R_flange = read_matrix(EE_DESIRED_ORI_KEY)
            return control_point_pos + R_flange @ CONTROL_POINT_TO_CAMERA_LOCAL
    except Exception:
        return control_point_pos   # fallback if orientation not yet available


# ===================== Look-at rotation (roll-continuous) =====================
def compute_look_at_rotation(p_ee, p_target, x_ref):
    """Build R so camera +Z points from p_ee toward p_target, keeping ROLL
    continuous frame-to-frame. Returns (R, new_x_ref)."""
    z_des = p_target - p_ee
    n = np.linalg.norm(z_des)
    if n < 1e-6:
        return None, x_ref
    z_des = z_des / n

    x_des = x_ref - z_des * np.dot(z_des, x_ref)
    if np.linalg.norm(x_des) < 1e-3:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, z_des)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        x_des = fallback - z_des * np.dot(z_des, fallback)

    x_des = x_des / np.linalg.norm(x_des)
    y_des = np.cross(z_des, x_des)
    y_des = y_des / np.linalg.norm(y_des)

    R = np.column_stack([x_des, y_des, z_des])
    return R, x_des


# ===================== SEARCHING constants =====================
SEARCH_ENDPOINT_A = np.array([0.52, -0.33, 0.53])   
SEARCH_ENDPOINT_B = np.array([0.52,  0.33, 0.53])   

ARRIVE_TOL = 0.03   # m

# Only transition SEARCHING -> TRACKING when the mouse lies inside the camera's
# central region 
CENTER_AREA_HORIZ = 0.30   # m  

# SEARCH_ORI = np.array([[ 1.0,  0.0,  0.0],
#                        [ 0.0, -1.0,  0.0],
#                        [ 0.0,  0.0, -1.0]])

SEARCH_ORI = np.array([[ -1.0,  0.0,  0.0],
                       [ 0.0, 1.0,  0.0],
                       [ 0.0,  0.0, -1.0]])


# ===================== TRACKING constants =====================
HOVER_HEIGHT   = 0.30   # m, vertical offset above mouse (world +z)
MAX_GOAL_SPEED = 0.15 # m/s
                
# Distance-scaled approach speed for the CV-TRACKING states (TRACKING / PRE_GRASP).

MAX_GOAL_SPEED_FAR  = 0.25  # m/s, cap when far from the target
MAX_GOAL_SPEED_NEAR = 0.15   # m/s, floor when overhead (the v1 known-stable speed)
GOAL_SPEED_GAIN     = 1.5    # (m/s)/m, speed ramp per meter of horizontal error

LOOK_AT_FREEZE_HORIZ = 0.08   # m

# Position deadband: once the goal is within this horizontal distance of the
# (noisy) CV estimate, STOP updating the goal's xy and let it lock. 
POS_DEADBAND_HORIZ = 0.015   # m

# overhead gate EMA-filtered (kills single-frame CV noise) and debounced
MOUSE_EMA_ALPHA           = 0.2   # EMA weight on each new CV center sample
GRASP_TRIGGER_HOLD_FRAMES = 2    


# ===================== GRASP constants =====================
# current_position = control point = flange + 0.15m (compliantFrame z offset)
# FLANGE_TO_TIP: flange origin -> gripper tip (calibrate empirically)
# GRASP_TIP_Z:   target gripper tip height above table (world z)
#
# flange z at grasp   = GRASP_TIP_Z + FLANGE_TO_TIP
# control point z     = flange z + COMPLIANT_FRAME_Z
# => GRASP_CONTROL_POINT_Z = GRASP_TIP_Z + FLANGE_TO_TIP - COMPLIANT_FRAME_Z

FLANGE_TO_TIP  = 0.19    # m, flange -> gripper tip
GRASP_TIP_Z    = 0.02# m, target gripper tip height above table

GRASP_CONTROL_POINT_Z = GRASP_TIP_Z + FLANGE_TO_TIP - COMPLIANT_FRAME_Z
# = 0.02 + 0.20 - 0.15 = 0.07 m

GRASP_TRIGGER_HORIZ = 0.04   # m, "directly overhead" threshold (TRACKING -> PRE_GRASP)
                             
GRASP_TRIGGER_TIME  = 2.0    # s, 

PRE_GRASP_CENTER_STABLE_HORIZ = 0.10

PRE_GRASP_ANCHOR_SETTLE = 0.5   # s

PRE_GRASP_CENTER_ABORT_HORIZ = 0.20   # m
PRE_GRASP_ABORT_FRAMES       = 50     # = 0.50s @ 100Hz

PRE_GRASP_DROPOUT_TOLERANCE_FRAMES = 20   # = 0.20s @ 100Hz

GRASP_ABORT_HORIZ   = 0.08   # m
GRASP_DONE_TOL      = 0.02   # m, control point within this of GRASP_CONTROL_POINT_Z = done
GRASP_TIMEOUT       = 30.0   # s, abort to SEARCHING if descent doesn't finish in time

# Fixed xy compensation added to mouse xy when computing the descent target.
# Empirical: gripper tip lands ~3cm short in +x without this, so push goal +x.
GRASP_XY_COMP       = np.array([0.02, 0.0])   # m, [dx, dy]


# ===================== LIFT constants =====================
LIFT_HEIGHT  = 0.25  # m, raise EE this much above grasp pose before DONE
LIFT_TOL     = 0.01   # m, control point within this of lift target z = done
LIFT_TIMEOUT = 5.0    # s, give up waiting and exit anyway


# ===================== PLACE constants =====================
# After LIFT, transport the mouse to PLACE_POSITION_XY, descend to
# PLACE_POSITION_Z, and only then open the gripper 
PLACE_POSITION_XY = np.array([0.75, -0.45])   # m, world frame [x, y]
PLACE_POSITION_Z  = 0.08                       # m, control-point z at release
                                            
PLACE_ARRIVE_TOL  = 0.04                       # m, xy distance considered "at target".
                                             
PLACE_Z_TOL       = 0.08                       # m, z distance considered "at target".
                                               
PLACE_OPEN_SETTLE = 0.5                        # s, wait after opening gripper


# ===================== SEARCHING state =====================
def enter_searching(ctx):
    ctx["search_target"] = SEARCH_ENDPOINT_A
    ctx["search_dbg_t0"] = time.time()
    write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)
    # Restore normal OTG velocity (in case we came from GRASP)
    redis_client.set(OTG_LINEAR_VEL_KEY, json.dumps(OTG_LINEAR_VEL_NORMAL))
    print(f"[state] -> SEARCHING (center gate horiz < {CENTER_AREA_HORIZ:.2f}m)")


def do_searching(ctx):
    target = ctx["search_target"]
    write_vector(EE_DESIRED_POS_KEY, target)
    write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)

    ee_pos = read_vector(EE_POS_KEY)   # control point position
    if np.linalg.norm(target - ee_pos) < ARRIVE_TOL:
        if np.allclose(target, SEARCH_ENDPOINT_A):
            ctx["search_target"] = SEARCH_ENDPOINT_B
            print("[search] reached A, heading to B")
        else:
            ctx["search_target"] = SEARCH_ENDPOINT_A
            print("[search] reached B, heading to A")

    if DEBUG_LOCK_SEARCHING:
        return "SEARCHING"

    # Center-area gate: only transition to TRACKING when the mouse falls inside
    detected, mouse_raw = read_perception_raw()
    if detected:
        camera_pos = get_camera_world_pos(ee_pos)
        horiz_cam_to_mouse = np.linalg.norm((mouse_raw - camera_pos)[:2])

        # Periodic visibility: once a second print EE / camera / mouse / horiz so
        # we can see why the center gate is or isn't triggering.
        if time.time() - ctx.get("search_dbg_t0", 0.0) > 1.0:
            ctx["search_dbg_t0"] = time.time()
            print(f"[search] ee_xyz=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
                  f"cam_xy=({camera_pos[0]:.3f},{camera_pos[1]:.3f}) "
                  f"mouse_xy=({mouse_raw[0]:.3f},{mouse_raw[1]:.3f}) "
                  f"horiz={horiz_cam_to_mouse:.3f}m "
                  f"(gate < {CENTER_AREA_HORIZ:.2f}) "
                  f"target_xy=({ctx['search_target'][0]:.3f},{ctx['search_target'][1]:.3f})",
                  flush=True)

        if horiz_cam_to_mouse < CENTER_AREA_HORIZ:
            print(f"[search] mouse in center area "
                  f"(horiz={horiz_cam_to_mouse:.3f} < {CENTER_AREA_HORIZ:.2f}) -> TRACKING")
            return "TRACKING"
    else:
        if time.time() - ctx.get("search_dbg_t0", 0.0) > 1.0:
            ctx["search_dbg_t0"] = time.time()
            print(f"[search] ee_xyz=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
                  f"target_xy=({ctx['search_target'][0]:.3f},{ctx['search_target'][1]:.3f}) "
                  f"mouse=NOT DETECTED",
                  flush=True)

    return "SEARCHING"


# ===================== TRACKING state =====================
def approach_max_step(horiz):
    """Distance-scaled per-loop goal slew limit for the CV-tracking states.
    Fast when far from the target, decaying to MAX_GOAL_SPEED_NEAR once overhead
    so the camera->CV->goal->EE position loop stays damped and the EE doesn't
    jitter above the mouse. Returns the max goal step (meters) for this loop tick."""
    speed = float(np.clip(horiz * GOAL_SPEED_GAIN,
                          MAX_GOAL_SPEED_NEAR, MAX_GOAL_SPEED_FAR))
    return speed / LOOP_HZ


def enter_tracking(ctx):
    ctx["goal_filtered"]  = read_vector(EE_POS_KEY)
    ctx["x_ref"]          = SEARCH_ORI[:, 0].copy()
    ctx["track_dbg_t0"]   = time.time()
    ctx["mouse_ema"]      = None   # EMA of CV center, set on first detection
    ctx["overhead_count"] = 0      # consecutive frames horiz under the gate
    print("[state] -> TRACKING")


def do_tracking(ctx):
    detected_raw, mouse_raw = read_perception_raw()
    if not detected_raw:
        print("[track] mouse lost -> SEARCHING")
        return "SEARCHING"

    # EMA-filter the (noisy) CV center.
    if ctx.get("mouse_ema") is None:
        ctx["mouse_ema"] = mouse_raw.copy()
    else:
        ctx["mouse_ema"] = (MOUSE_EMA_ALPHA * mouse_raw
                            + (1.0 - MOUSE_EMA_ALPHA) * ctx["mouse_ema"])
    mouse_f = ctx["mouse_ema"]

    control_point_pos = read_vector(EE_POS_KEY)
    camera_pos        = get_camera_world_pos(control_point_pos)

    horiz = np.linalg.norm((mouse_f - control_point_pos)[:2])

    target_hover_z = mouse_f[2] + HOVER_HEIGHT
    z_at_hover     = abs(control_point_pos[2] - target_hover_z) < 0.05

    # Debounce: horiz must hold under the gate for GRASP_TRIGGER_HOLD_FRAMES
    # consecutive frames before we commit, instead of a single lucky frame.
    if horiz < GRASP_TRIGGER_HORIZ and z_at_hover:
        ctx["overhead_count"] += 1
    else:
        ctx["overhead_count"] = 0

    # Periodic visibility: log horiz + hold progress once a second.
    if time.time() - ctx.get("track_dbg_t0", 0.0) > 1.0:
        ctx["track_dbg_t0"] = time.time()
        print(f"[track] horiz={horiz:.3f}m (gate < {GRASP_TRIGGER_HORIZ:.3f}, "
              f"hold {ctx['overhead_count']}/{GRASP_TRIGGER_HOLD_FRAMES}) "
              f"ee_z={control_point_pos[2]:.3f} "
              f"(hover target {target_hover_z:.3f}, at_hover={z_at_hover}) "
              f"mouse_xy=({mouse_f[0]:.3f},{mouse_f[1]:.3f}) "
              f"ee_xy=({control_point_pos[0]:.3f},{control_point_pos[1]:.3f})",
              flush=True)

    if ctx["overhead_count"] >= GRASP_TRIGGER_HOLD_FRAMES:
        ctx["pre_grasp_mouse_xy"] = mouse_f[:2].copy()
        print(f"[track] overhead held {GRASP_TRIGGER_HOLD_FRAMES} frames "
              f"(horiz={horiz:.3f}) -> PRE_GRASP")
        return "PRE_GRASP"

    # Position goal uses the filtered center.
    goal_raw = mouse_f.copy()
    goal_raw[2] += HOVER_HEIGHT
    delta    = goal_raw - ctx["goal_filtered"]

    if np.linalg.norm(delta[:2]) < POS_DEADBAND_HORIZ:
        delta[0] = 0.0
        delta[1] = 0.0
    # Distance-scaled slew: fast when far, decays to the stable 0.15 overhead.
    max_step = approach_max_step(horiz)
    nrm = np.linalg.norm(delta)
    if nrm > max_step:
        delta = max_step * delta / nrm
    ctx["goal_filtered"] = ctx["goal_filtered"] + delta
    write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])

    # Look-at: freeze the orientation once we're close (horiz < freeze gate) to
    # break the camera-aim feedback loop. Far away, keep aiming so the mouse
    # stays in frame; the filtered center keeps the aim steady.
    if horiz >= LOOK_AT_FREEZE_HORIZ:
        R_des, ctx["x_ref"] = compute_look_at_rotation(camera_pos, mouse_f,
                                                       ctx["x_ref"])
        if R_des is not None:
            write_matrix(EE_DESIRED_ORI_KEY, R_des)

    return "TRACKING"


# ===================== PRE_GRASP state =====================
# Re-aim the EE above the mouse TAIL (from the CV, mouse_pose_tail_stream_real_1).
# Orientation stays at SEARCH_ORI (flange parallel to table) to avoid the
# look-at feedback loop.
def enter_pre_grasp(ctx):
    detected_tail, tail_pos    = read_perception_tail()
    _,             mouse_raw_c = read_perception_raw()
    ee_pos = read_vector(EE_POS_KEY)
    if detected_tail:
        ctx["pre_grasp_tail_xy"] = tail_pos[:2].copy()
    else:
        ctx["pre_grasp_tail_xy"] = ctx["pre_grasp_mouse_xy"].copy()

    # Body axis EMA: vector from mouse center to tail base in xy. Used at
    # GRASP entry to yaw the gripper 
    if detected_tail and mouse_raw_c is not None:
        ctx["pre_grasp_body_vec"] = tail_pos[:2] - mouse_raw_c[:2]
    else:
        ctx["pre_grasp_body_vec"] = np.array([1.0, 0.0])   # arbitrary fallback

    ctx["pre_grasp_hover_z"]       = float(ee_pos[2])   # keep current hover height
    ctx["goal_filtered"]           = ee_pos.copy()
    ctx["pre_grasp_t0"]            = None               # timer starts once overhead
    ctx["pre_grasp_dropout_count"] = 0                  # consecutive CV-dropout frames
    ctx["pre_grasp_dbg_t0"]        = time.time()

    ctx["pre_grasp_mouse_xy"]      = None
    ctx["pre_grasp_anchor_t0"]     = time.time()
    ctx["pre_grasp_abort_count"]   = 0                  # consecutive far-drift frames
    write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)
    print(f"[state] -> PRE_GRASP (target tail xy={ctx['pre_grasp_tail_xy']}, "
          f"hover_z={ctx['pre_grasp_hover_z']:.3f}, "
          f"body_vec0={ctx['pre_grasp_body_vec']}, "
          f"hold {GRASP_TRIGGER_TIME:.1f}s once overhead)")


def do_pre_grasp(ctx):
    # Tail (noisy) drives the EE goal; center (stable) gates the hold timer.
    detected_tail,   tail_pos  = read_perception_tail()
    detected_center, mouse_raw = read_perception_raw()

    if not detected_tail or not detected_center:
        ctx["pre_grasp_dropout_count"] += 1
        if ctx["pre_grasp_dropout_count"] > PRE_GRASP_DROPOUT_TOLERANCE_FRAMES:
            print(f"[pre_grasp] tail/center lost "
                  f"{ctx['pre_grasp_dropout_count']} frames "
                  f"(> {PRE_GRASP_DROPOUT_TOLERANCE_FRAMES}) -> TRACKING")
            return "TRACKING"
        
        write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])
        write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)
        return "PRE_GRASP"

    # Good frame: clear the dropout counter and run the normal logic.
    ctx["pre_grasp_dropout_count"] = 0

    # EMA the tail xy: YOLO tail keypoint is noisy (±15mm). 
    TAIL_EMA_ALPHA = 0.1
    ctx["pre_grasp_tail_xy"] = (
        TAIL_EMA_ALPHA * tail_pos[:2]
        + (1.0 - TAIL_EMA_ALPHA) * ctx["pre_grasp_tail_xy"]
    )

    # EMA the body axis vector (tail - center in xy) so the GRASP yaw is
    # stable. 
    BODY_VEC_EMA_ALPHA = 0.05
    body_vec_now = tail_pos[:2] - mouse_raw[:2]
    ctx["pre_grasp_body_vec"] = (
        BODY_VEC_EMA_ALPHA * body_vec_now
        + (1.0 - BODY_VEC_EMA_ALPHA) * ctx["pre_grasp_body_vec"]
    )

    goal_raw = np.array([ctx["pre_grasp_tail_xy"][0],
                         ctx["pre_grasp_tail_xy"][1],
                         ctx["pre_grasp_hover_z"]])
    delta    = goal_raw - ctx["goal_filtered"]
    # Position deadband (same as TRACKING): lock the goal xy once it's over the
    # tail so the noisy tail keypoint doesn't jitter the EE during the 3s hold.
    if np.linalg.norm(delta[:2]) < POS_DEADBAND_HORIZ:
        delta[0] = 0.0
        delta[1] = 0.0
    max_step = approach_max_step(np.linalg.norm(delta[:2]))
    nrm = np.linalg.norm(delta)
    if nrm > max_step:
        delta = max_step * delta / nrm
    ctx["goal_filtered"] = ctx["goal_filtered"] + delta

    write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])
    write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)

    if ctx["pre_grasp_mouse_xy"] is None:
        if time.time() - ctx["pre_grasp_anchor_t0"] >= PRE_GRASP_ANCHOR_SETTLE:
            ctx["pre_grasp_mouse_xy"] = mouse_raw[:2].copy()
            print(f"[pre_grasp] anchored center at "
                  f"({mouse_raw[0]:.3f},{mouse_raw[1]:.3f}) "
                  f"after {PRE_GRASP_ANCHOR_SETTLE:.1f}s settle")
        else:
            return "PRE_GRASP"

    center_drift = np.linalg.norm(mouse_raw[:2] - ctx["pre_grasp_mouse_xy"])

    # Escape hatch: a sustained large drift means the mouse genuinely relocated
    # (not noise). Fall back to TRACKING to re-acquire rather than hang here
    # forever with the timer never starting.
    if center_drift > PRE_GRASP_CENTER_ABORT_HORIZ:
        ctx["pre_grasp_abort_count"] += 1
        if ctx["pre_grasp_abort_count"] > PRE_GRASP_ABORT_FRAMES:
            print(f"[pre_grasp] center drifted {center_drift:.3f}m "
                  f"(> {PRE_GRASP_CENTER_ABORT_HORIZ:.2f}) for "
                  f"{ctx['pre_grasp_abort_count']} frames -> TRACKING (re-acquire)")
            return "TRACKING"
    else:
        ctx["pre_grasp_abort_count"] = 0

    if time.time() - ctx.get("pre_grasp_dbg_t0", 0.0) > 1.0:
        ctx["pre_grasp_dbg_t0"] = time.time()
        if ctx["pre_grasp_t0"] is None:
            timer_state = "NOT STARTED"
        else:
            timer_state = f"holding {time.time() - ctx['pre_grasp_t0']:.1f}/{GRASP_TRIGGER_TIME:.1f}s"
        print(f"[pre_grasp] drift={center_drift:.3f}m "
              f"(gate < {PRE_GRASP_CENTER_STABLE_HORIZ:.3f}) "
              f"timer={timer_state} "
              f"mouse_raw=({mouse_raw[0]:.3f},{mouse_raw[1]:.3f}) "
              f"anchor=({ctx['pre_grasp_mouse_xy'][0]:.3f},"
              f"{ctx['pre_grasp_mouse_xy'][1]:.3f}) "
              f"tail=({ctx['pre_grasp_tail_xy'][0]:.3f},"
              f"{ctx['pre_grasp_tail_xy'][1]:.3f})",
              flush=True)

    if center_drift < PRE_GRASP_CENTER_STABLE_HORIZ:
        if ctx["pre_grasp_t0"] is None:
            ctx["pre_grasp_t0"] = time.time()
            print(f"[pre_grasp] mouse center stable "
                  f"(drift={center_drift:.3f}m), starting "
                  f"{GRASP_TRIGGER_TIME:.1f}s hold")
        elif time.time() - ctx["pre_grasp_t0"] >= GRASP_TRIGGER_TIME:
            print(f"[pre_grasp] held stable -> GRASP "
                  f"(descent target tail xy={ctx['pre_grasp_tail_xy']})")
            return "GRASP"
    else:
        if ctx["pre_grasp_t0"] is not None:
            print(f"[pre_grasp] mouse moved "
                  f"(center drift={center_drift:.3f}m > "
                  f"{PRE_GRASP_CENTER_STABLE_HORIZ:.3f}), reset hold timer")
            ctx["pre_grasp_t0"] = None

    return "PRE_GRASP"


# ===================== GRASP state =====================
def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def enter_grasp(ctx):
    # Descend onto the tail xy that PRE_GRASP locked onto (CV will be
    # occluded by the gripper from here on, so we freeze the target now).
    ctx["grasp_xy"]      = ctx["pre_grasp_tail_xy"].copy() + GRASP_XY_COMP
    ctx["goal_filtered"] = read_vector(EE_POS_KEY)
    ctx["grasp_t0"]      = time.time()

    body = ctx["pre_grasp_body_vec"]
    nrm  = float(np.linalg.norm(body))
    if nrm < 1e-3:
        theta = 0.0
    else:
        theta = float(np.arctan2(body[1] / nrm, body[0] / nrm))
        if theta >  np.pi / 2:
            theta -= np.pi
        elif theta < -np.pi / 2:
            theta += np.pi
    ctx["grasp_ori"] = _rotz(theta) @ SEARCH_ORI

    write_matrix(EE_DESIRED_ORI_KEY, ctx["grasp_ori"])
    # Slow down OTG for safe descent
    redis_client.set(OTG_LINEAR_VEL_KEY, json.dumps(OTG_LINEAR_VEL_GRASP))
    # Open gripper at start of descent
    write_gripper_parameters(*GRIPPER_OPEN_PARAMS)
    print(f"[state] -> GRASP (tail tip xy={ctx['grasp_xy']}, "
          f"target_z={GRASP_CONTROL_POINT_Z:.3f}, "
          f"body_vec={body}, yaw={np.degrees(theta):+.1f}deg, "
          f"otg_vel={OTG_LINEAR_VEL_GRASP}, gripper=open {GRIPPER_OPEN_PARAMS})")


def do_grasp(ctx):
    """Descend control point to GRASP_CONTROL_POINT_Z.
    control point = flange + 0.15m (compliantFrame), so:
      GRASP_CONTROL_POINT_Z = GRASP_TIP_Z + FLANGE_TO_TIP - COMPLIANT_FRAME_Z

    CV is ignored once GRASP starts (gripper occludes camera). xy target is
    locked to ctx['grasp_xy'] captured at enter_grasp. Timeout falls back to
    SEARCHING."""

    if time.time() - ctx["grasp_t0"] > GRASP_TIMEOUT:
        print(f"[grasp] timeout {GRASP_TIMEOUT:.0f}s -> SEARCHING (fail)")
        return "SEARCHING"

    goal_raw = np.array([ctx["grasp_xy"][0],
                         ctx["grasp_xy"][1],
                         GRASP_CONTROL_POINT_Z])

    delta    = goal_raw - ctx["goal_filtered"]
    max_step = MAX_GOAL_SPEED / LOOP_HZ
    nrm = np.linalg.norm(delta)
    if nrm > max_step:
        delta = max_step * delta / nrm
    ctx["goal_filtered"] = ctx["goal_filtered"] + delta

    write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])
    write_matrix(EE_DESIRED_ORI_KEY, ctx["grasp_ori"])

    ee_pos = read_vector(EE_POS_KEY)
    if abs(ee_pos[2] - GRASP_CONTROL_POINT_Z) < GRASP_DONE_TOL:
        write_gripper_parameters(*GRIPPER_CLOSE_PARAMS)
        print(f"[grasp] descent complete; sending close {GRIPPER_CLOSE_PARAMS}, "
              f"settling {GRIPPER_SETTLE_TIME:.1f}s")
        time.sleep(GRIPPER_SETTLE_TIME)

        post_close_grasp_force = sample_gripper_grasp_force_mean()
        ctx["grasp_force_after_close"] = post_close_grasp_force

        # Informational logs.
        width_after  = read_gripper_width()
        tcp_force_at_grasp = sample_tcp_force_mean(n_samples=5)
        print(f"[grasp] after close+settle: width={width_after}, "
              f"gripper_grasp_force={post_close_grasp_force} N, "
              f"tcp_force={tcp_force_at_grasp} (info only)")
        print(f"[grasp] -> LIFT (decision will use grasp_force "
              f"threshold |f| >= {GRASP_FORCE_THRESHOLD} N)")
        return "LIFT"

    return "GRASP"


# ===================== LIFT state =====================
# After the gripper has closed on the mouse, raise the EE LIFT_HEIGHT meters
# along world +z while keeping orientation = SEARCH_ORI (flange parallel to
# the table). Exit to DONE once the target z is reached (or on timeout).
def enter_lift(ctx):
    current_pos = read_vector(EE_POS_KEY)
    ctx["lift_target"]   = current_pos.copy()
    ctx["lift_target"][2] += LIFT_HEIGHT
    ctx["lift_t0"]       = time.time()
    ctx["goal_filtered"] = current_pos.copy()
    ctx["lift_verified"] = False   # force check runs once after reaching lift_z
    post_close = ctx.get("grasp_force_after_close")
    print(f"[state] -> LIFT (from z={current_pos[2]:.3f} "
          f"to z={ctx['lift_target'][2]:.3f}, "
          f"grasp_force after close+settle = {post_close} N; "
          f"will re-check at lift top)")


def do_lift(ctx):
    delta    = ctx["lift_target"] - ctx["goal_filtered"]
    max_step = MAX_GOAL_SPEED / LOOP_HZ
    nrm = np.linalg.norm(delta)
    if nrm > max_step:
        delta = max_step * delta / nrm
    ctx["goal_filtered"] = ctx["goal_filtered"] + delta

    write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])

    write_matrix(EE_DESIRED_ORI_KEY, ctx.get("grasp_ori", SEARCH_ORI))

    ee_pos    = read_vector(EE_POS_KEY)
    reached   = abs(ee_pos[2] - ctx["lift_target"][2]) < LIFT_TOL
    timed_out = time.time() - ctx["lift_t0"] > LIFT_TIMEOUT

    if not (reached or timed_out) or ctx.get("lift_verified", False):
        return "LIFT"

    ctx["lift_verified"] = True
    if timed_out and not reached:
        print(f"[lift] timeout {LIFT_TIMEOUT:.1f}s at z={ee_pos[2]:.3f} "
              f"(target {ctx['lift_target'][2]:.3f}) -- running grasp_force "
              f"check anyway")
    else:
        print(f"[lift] reached target z={ee_pos[2]:.3f}, sampling grasp_force")

    grasp_force_now = sample_gripper_grasp_force_mean()
    post_close      = ctx.get("grasp_force_after_close")
    tcp_force_now   = sample_tcp_force_mean(n_samples=5)
    print(f"[lift] grasp_force after close = {post_close} N, "
          f"now = {grasp_force_now} N, "
          f"tcp_force = {tcp_force_now} (info only)")

    if grasp_force_now is None:
        print(f"[lift] WARNING: grasp_force unavailable -> assuming SUCCESS, "
              f"-> PLACE")
        return "PLACE"

    if abs(grasp_force_now) >= GRASP_FORCE_THRESHOLD:
        print(f"[lift] GRASP CONFIRMED (|grasp_force|={abs(grasp_force_now):.2f}N "
              f">= {GRASP_FORCE_THRESHOLD}N) -> PLACE")
        return "PLACE"

    print(f"[lift] GRASP EMPTY (|grasp_force|={abs(grasp_force_now):.2f}N < "
          f"{GRASP_FORCE_THRESHOLD}N) -> releasing, back to SEARCHING")
    write_gripper_parameters(*GRIPPER_OPEN_PARAMS)
    return "SEARCHING"


# ===================== PLACE state =====================

def enter_place(ctx):
    ee_pos = read_vector(EE_POS_KEY)
    ctx["place_xy_target"] = np.array(PLACE_POSITION_XY, dtype=float)
    ctx["place_z_final"]   = float(PLACE_POSITION_Z)
    ctx["place_transit_z"] = float(ee_pos[2])   # hold post-LIFT z during xy transit
    ctx["place_phase"]     = "TRANSIT"
    ctx["place_enter_t0"]  = time.time()   # for elapsed-time logging only
    ctx["place_dbg_t0"]    = time.time()
    ctx["place_arrived"]   = False
    ctx["goal_filtered"]   = ee_pos.copy()
    # Restore normal OTG velocity for transport (GRASP slowed it down).
    redis_client.set(OTG_LINEAR_VEL_KEY, json.dumps(OTG_LINEAR_VEL_NORMAL))
    print(f"[state] -> PLACE (xy_target={ctx['place_xy_target']}, "
          f"transit_z={ctx['place_transit_z']:.3f}, "
          f"final_z={ctx['place_z_final']:.3f}, "
          f"otg_vel={OTG_LINEAR_VEL_NORMAL})")


def do_place(ctx):

    if ctx["place_phase"] == "TRANSIT":
        goal_z = ctx["place_transit_z"]
    else:
        goal_z = ctx["place_z_final"]
    goal_raw = np.array([ctx["place_xy_target"][0],
                         ctx["place_xy_target"][1],
                         goal_z])

    delta    = goal_raw - ctx["goal_filtered"]
    max_step = MAX_GOAL_SPEED / LOOP_HZ
    nrm = np.linalg.norm(delta)
    if nrm > max_step:
        delta = max_step * delta / nrm
    ctx["goal_filtered"] = ctx["goal_filtered"] + delta

    write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])
    write_matrix(EE_DESIRED_ORI_KEY, ctx.get("grasp_ori", SEARCH_ORI))

    ee_pos = read_vector(EE_POS_KEY)
    horiz  = np.linalg.norm(ee_pos[:2] - ctx["place_xy_target"])
    z_err  = abs(ee_pos[2] - ctx["place_z_final"])

    if time.time() - ctx.get("place_dbg_t0", 0.0) > 1.0:
        ctx["place_dbg_t0"] = time.time()
        elapsed = time.time() - ctx["place_enter_t0"]
        print(f"[place/{ctx['place_phase']}] "
              f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
              f"target=({ctx['place_xy_target'][0]:.3f},"
              f"{ctx['place_xy_target'][1]:.3f},{ctx['place_z_final']:.3f}) "
              f"horiz={horiz:.3f} (tol {PLACE_ARRIVE_TOL:.2f}) "
              f"z_err={z_err:.3f} (tol {PLACE_Z_TOL:.2f}) "
              f"elapsed={elapsed:.1f}s",
              flush=True)

    goal_horiz = np.linalg.norm(ctx["goal_filtered"][:2] - ctx["place_xy_target"])
    if ctx["place_phase"] == "TRANSIT" and goal_horiz < PLACE_ARRIVE_TOL:
        ctx["place_phase"] = "DESCEND"
        print(f"[place] goal centered over target (goal_horiz={goal_horiz:.3f}) "
              f"-> DESCEND from z={ee_pos[2]:.3f} to z={ctx['place_z_final']:.3f}")

    # Open the gripper once the EE has truly descended to PLACE_POSITION_Z. 
    if (ctx["place_phase"] == "DESCEND" and not ctx["place_arrived"]
            and z_err < PLACE_Z_TOL):
        ctx["place_arrived"]  = True
        ctx["place_drop_t0"]  = time.time()
        write_gripper_parameters(*GRIPPER_OPEN_PARAMS)
        print(f"[place] descended (z_err={z_err:.3f}, horiz={horiz:.3f}) -> "
              f"opening gripper {GRIPPER_OPEN_PARAMS}, settle {PLACE_OPEN_SETTLE:.1f}s")

    if ctx["place_arrived"] and \
       time.time() - ctx["place_drop_t0"] >= PLACE_OPEN_SETTLE:
        print("[place] mouse released -> RETURN_HOME")
        return "RETURN_HOME"

    return "PLACE"


# ===================== RETURN_HOME state =====================

RETURN_LIFT_HEIGHT = 0.20   # m
RETURN_ARRIVE_TOL  = 0.04   # m

def enter_return_home(ctx):
    ee_pos = read_vector(EE_POS_KEY)
    ctx["return_phase"]  = "LIFT"
    ctx["return_lift_z"] = float(ee_pos[2]) + RETURN_LIFT_HEIGHT
    ctx["goal_filtered"] = ee_pos.copy()
    ctx["return_dbg_t0"] = time.time()
    # Transport speed back to normal (GRASP slowed OTG; PLACE already restored
    # it, but re-assert here to be safe).
    redis_client.set(OTG_LINEAR_VEL_KEY, json.dumps(OTG_LINEAR_VEL_NORMAL))
    print(f"[state] -> RETURN_HOME (lift to z={ctx['return_lift_z']:.3f}, "
          f"then home={ctx['home_pos']})")

def do_return_home(ctx):
    home_pos = ctx["home_pos"]
    home_ori = ctx.get("home_ori", SEARCH_ORI)

    if ctx["return_phase"] == "LIFT":
        goal_raw = np.array([ctx["goal_filtered"][0],
                             ctx["goal_filtered"][1],
                             ctx["return_lift_z"]])
    else:
        goal_raw = home_pos.copy()

    delta    = goal_raw - ctx["goal_filtered"]
    max_step = MAX_GOAL_SPEED / LOOP_HZ
    nrm = np.linalg.norm(delta)
    if nrm > max_step:
        delta = max_step * delta / nrm
    ctx["goal_filtered"] = ctx["goal_filtered"] + delta

    write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])
    write_matrix(EE_DESIRED_ORI_KEY, home_ori)

    ee_pos = read_vector(EE_POS_KEY)

    if time.time() - ctx.get("return_dbg_t0", 0.0) > 1.0:
        ctx["return_dbg_t0"] = time.time()
        print(f"[return/{ctx['return_phase']}] "
              f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
              f"home=({home_pos[0]:.3f},{home_pos[1]:.3f},{home_pos[2]:.3f})",
              flush=True)

    # LIFT -> HOME
    if ctx["return_phase"] == "LIFT":
        if abs(ctx["goal_filtered"][2] - ctx["return_lift_z"]) < 1e-3:
            ctx["return_phase"] = "HOME"
            print(f"[return] lifted (goal z={ctx['goal_filtered'][2]:.3f}) -> HOME")
        return "RETURN_HOME"

    # HOME -> DONE 
    home_err = float(np.linalg.norm(ee_pos - home_pos))
    if home_err < RETURN_ARRIVE_TOL:
        print(f"[return] reached home pose (err={home_err:.3f}m) -> DONE")
        return "DONE"

    return "RETURN_HOME"


# ===================== CV subprocess management =====================
# We auto-launch mouse_pose_tail_stream_real_2.py 
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
CV_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "mouse_pose_tail_stream_real_2.py"
CV_VENV_PYTHON = _PROJECT_ROOT / "venv" / "bin" / "python"

_cv_processes = []

def launch_cv_subprocess():
    if not CV_SCRIPT_PATH.exists():
        raise RuntimeError(f"CV script not found: {CV_SCRIPT_PATH}")
    if CV_VENV_PYTHON.exists():
        python_exe = str(CV_VENV_PYTHON)
    else:
        print(f"[cv] WARNING: venv python not found at {CV_VENV_PYTHON}, "
              f"falling back to {sys.executable}")
        python_exe = sys.executable

    cv_log_path = "/tmp/cv_pose_stream.log"
    cv_log = open(cv_log_path, "w")
    print(f"[cv] launching {CV_SCRIPT_PATH.name} with {python_exe} "
          f"(output -> {cv_log_path})")
    p = subprocess.Popen(
        [python_exe, str(CV_SCRIPT_PATH)],
        stdout=cv_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,   # own process group so we can SIGINT cleanly
    )
    _cv_processes.append(p)
    return p


def kill_cv_subprocesses():
    if not _cv_processes:
        return
    for p in _cv_processes:
        if p.poll() is None:
            try:
                # Send SIGINT to the whole group so RealSense/cv2 shut down
                # cleanly via the CV script's own signal handler.
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                continue
    deadline = time.time() + 3.0
    for p in _cv_processes:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    _cv_processes.clear()


atexit.register(kill_cv_subprocesses)


def _handle_term_signal(signum, _frame):
    print(f"\n[main] caught signal {signum}, shutting down CV subprocess(es)")
    kill_cv_subprocesses()
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_term_signal)


# ===================== Main loop =====================
def wait_for_key(key, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if redis_client.get(key) is not None:
            return True
        time.sleep(0.1)
    return False


def main():
    print("Waiting for controller to come online...")

    # Switch to cartesian_controller (required for real robot)
    # redis_client.set(ACTIVE_CONTROLLER_KEY, "cartesian_controller")
    # print("Switched active controller to: cartesian_controller")

    if not wait_for_key(EE_POS_KEY):
        raise RuntimeError(f"{EE_POS_KEY} never appeared -- is the opensai "
                           f"controller running?")

    # Null-space posture bias:
    if wait_for_key(JOINT_POSITIONS_KEY, timeout=5.0):
        try:
            initial_joints = read_vector(JOINT_POSITIONS_KEY)
            redis_client.set(JOINT_TASK_GOAL_KEY,
                             json.dumps([float(x) for x in initial_joints]))
            print(f"[startup] null-space joint goal set to {initial_joints} "
                  f"(captured from current pose)")
        except Exception as e:
            print(f"[startup] WARNING: could not set null-space joint goal: {e}")
    else:
        print(f"[startup] WARNING: {JOINT_POSITIONS_KEY} unavailable; "
              f"null-space will use controller default")

    # Clear any STALE perception values left in redis by a previous run 
    redis_client.delete(PERCEPTION_RAW_KEY, PERCEPTION_TAIL_KEY)

    # Auto-launch the CV pipeline.
    launch_cv_subprocess()
    print(f"[cv] cleared stale perception keys; waiting for fresh "
          f"{PERCEPTION_RAW_KEY} ...")
    if not wait_for_key(PERCEPTION_RAW_KEY, timeout=30.0):
        raise RuntimeError(f"{PERCEPTION_RAW_KEY} never appeared -- "
                           f"did the CV script fail to start?")
    print(f"[cv] waiting for {PERCEPTION_TAIL_KEY} ...")
    if not wait_for_key(PERCEPTION_TAIL_KEY, timeout=30.0):
        raise RuntimeError(f"{PERCEPTION_TAIL_KEY} never appeared -- "
                           f"did the CV script fail to start?")
    print("[cv] both perception keys present.")

    # Open gripper at startup
    write_gripper_parameters(*GRIPPER_OPEN_PARAMS)
    print(f"[startup] gripper open {GRIPPER_OPEN_PARAMS}")

    ctx = {}
    # Capture the session "home" pose BEFORE enter_searching 
    ctx["home_pos"] = read_vector(EE_POS_KEY)
    try:
        ctx["home_ori"] = read_matrix(EE_DESIRED_ORI_KEY)
    except Exception:
        ctx["home_ori"] = SEARCH_ORI
    print(f"[startup] home pose captured: pos={ctx['home_pos']}")

    state = "SEARCHING"
    enter_searching(ctx)

    period = 1.0 / LOOP_HZ

    try:
        while True:
            t0 = time.time()

            if state == "SEARCHING":
                nxt = do_searching(ctx)
                if nxt != state:
                    state = nxt
                    enter_tracking(ctx)

            elif state == "TRACKING":
                nxt = do_tracking(ctx)
                if nxt != state:
                    state = nxt
                    if nxt == "PRE_GRASP":
                        enter_pre_grasp(ctx)
                    elif nxt == "SEARCHING":
                        enter_searching(ctx)

            elif state == "PRE_GRASP":
                nxt = do_pre_grasp(ctx)
                if nxt != state:
                    state = nxt
                    if nxt == "GRASP":
                        enter_grasp(ctx)
                    elif nxt == "TRACKING":
                        enter_tracking(ctx)

            elif state == "GRASP":
                nxt = do_grasp(ctx)
                if nxt != state:
                    state = nxt
                    if nxt == "LIFT":
                        enter_lift(ctx)
                    elif nxt == "DONE":
                        print("[state] -> DONE, exiting state machine")
                        break
                    else:
                        enter_searching(ctx)

            elif state == "LIFT":
                nxt = do_lift(ctx)
                if nxt != state:
                    state = nxt
                    if nxt == "PLACE":
                        enter_place(ctx)
                    elif nxt == "DONE":
                        print("[state] -> DONE, exiting state machine")
                        break

            elif state == "PLACE":
                nxt = do_place(ctx)
                if nxt != state:
                    state = nxt
                    if nxt == "RETURN_HOME":
                        enter_return_home(ctx)
                    elif nxt == "DONE":
                        print("[state] -> DONE, exiting state machine")
                        break

            elif state == "RETURN_HOME":
                nxt = do_return_home(ctx)
                if nxt != state:
                    state = nxt
                    if nxt == "DONE":
                        print("[state] -> DONE, exiting state machine")
                        break

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nState machine stopped.")


if __name__ == "__main__":
    main()