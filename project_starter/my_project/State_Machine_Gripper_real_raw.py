#!/usr/bin/env python3
"""
@file state_machine.py
@brief SEARCHING / TRACKING / GRASP state machine for the Rizon4s (Titania)
       mouse-chasing project.

  Reads:
    opensai::perception::desired_position   (Vector3d or []) - raw mouse position
    opensai::controllers::Titania::cartesian_controller::cartesian_task::current_position
                                            (Vector3d) - current control point position
  Writes:
    opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_position
    opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_orientation
    opensai::commands::Titania::gripper::mode
    opensai::commands::Titania::gripper::parameters

Coordinate frames (Titania real robot):
  - link7 → flange: xyz=[0,0,0.124], rpy=[0,0,π] (180° rotation around z)
  - flange → control point (compliantFrame): xyz=[0,0,0.15] in flange local z
  - camera in link7 frame: pos=[0.074,-0.01,0.136] (William)
  - camera in flange frame: Rz(180°) @ ([0.074,-0.01,0.136]-[0,0,0.124])
                           = [-0.074, 0.01, 0.012]
  - camera relative to control point: [-0.074, 0.01, 0.012-0.15] = [-0.074, 0.01, -0.138]
  - FLANGE_TO_TIP = 0.20m (from URDF closed_fingers_tcp: grav_base_link + 0.20m)

GRASP sequence:
    1. open gripper
    2. descend to GRASP_CONTROL_POINT_Z
    3. close gripper (grasp mouse)
    4. lift back to GRASP_LIFT_Z
    5. open gripper (release mouse)
    6. -> SEARCHING

State graph:
    SEARCHING --detected--> TRACKING --overhead 3s--> GRASP --> SEARCHING

Launch order: opensai (with tom_and_jerry.xml) -> CV detect -> this script.
"""

import time
import json
import numpy as np
import redis

# ===================== Debug flags =====================
DEBUG_LOCK_SEARCHING = False


# ===================== Loop rate =====================
LOOP_HZ = 100


# ===================== Redis setup =====================
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# --- Safety checks ---
CONFIG_FILE_KEY      = "::sai-interfaces-webui::config_file_name"
EXPECTED_CONFIG_FILE = "tom_and_jerry.xml"   # filename only, not full path

# --- CV pipeline ---
PERCEPTION_RAW_KEY    = "opensai::perception::desired_position"

# --- Titania controller ---
EE_POS_KEY            = "opensai::controllers::Titania::cartesian_controller::cartesian_task::current_position"
EE_DESIRED_POS_KEY    = "opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_position"
EE_DESIRED_ORI_KEY    = "opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_orientation"
ACTIVE_CONTROLLER_KEY = "opensai::controllers::Titania::active_controller_name"
OTG_LINEAR_VEL_KEY    = "opensai::controllers::Titania::cartesian_controller::cartesian_task::otg_max_linear_velocity"

# --- Gripper ---
# ⚠️  Verify exact format with William before running on real robot
GRIPPER_MODE_KEY  = "opensai::commands::Titania::gripper::mode"
GRIPPER_PARAM_KEY = "opensai::commands::Titania::gripper::parameters"

# OTG velocity
OTG_LINEAR_VEL_NORMAL = 0.15   # m/s, matches xml
OTG_LINEAR_VEL_GRASP  = 0.10   # m/s, slow descent


# ===================== Serialization helpers =====================
def read_vector(key):
    return np.array(json.loads(redis_client.get(key)), dtype=float).flatten()

def write_vector(key, vec):
    redis_client.set(key, json.dumps([float(x) for x in vec]))

def read_matrix(key):
    return np.array(json.loads(redis_client.get(key)), dtype=float)

def write_matrix(key, mat):
    redis_client.set(key, json.dumps([[float(x) for x in row] for row in mat]))


# ===================== CV perception helpers =====================
def read_perception_raw():
    """Returns (detected: bool, pos: np.array or None)."""
    try:
        val = json.loads(redis_client.get(PERCEPTION_RAW_KEY))
        if isinstance(val, list) and len(val) == 3:
            return True, np.array(val, dtype=float)
        return False, None
    except Exception:
        return False, None


# ===================== Gripper helpers =====================
# ⚠️  Format unconfirmed — verify with William or check opensai gripper docs
def gripper_open():
    redis_client.set(GRIPPER_MODE_KEY, "move")
    redis_client.set(GRIPPER_PARAM_KEY, json.dumps({"width": 0.08, "speed": 0.1}))
    print("[gripper] open")

def gripper_close():
    redis_client.set(GRIPPER_MODE_KEY, "grasp")
    redis_client.set(GRIPPER_PARAM_KEY, json.dumps({"width": 0.01, "speed": 0.05, "force": 20.0}))
    print("[gripper] close (grasp)")

GRIPPER_WAIT = 1.5   # s, wait after each gripper command


# ===================== Control point -> Camera conversion =====================
# link7 → flange: xyz=[0,0,0.124] + Rz(180°)
# camera in link7: [0.074, -0.01, 0.136]
# camera in flange: Rz(180°) @ ([0.074,-0.01,0.136] - [0,0,0.124])
#                 = [-1,0,0; 0,-1,0; 0,0,1] @ [0.074,-0.01,0.012]
#                 = [-0.074, 0.01, 0.012]
# control point in flange: [0,0,0.15] (compliantFrame)
# camera relative to control point: [-0.074, 0.01, 0.012-0.15] = [-0.074, 0.01, -0.138]
COMPLIANT_FRAME_Z             = 0.15
CONTROL_POINT_TO_CAMERA_LOCAL = np.array([-0.074, 0.01, -0.138])

def get_camera_world_pos(control_point_pos):
    """Convert control point world pos to camera world pos."""
    try:
        R_flange = read_matrix(EE_DESIRED_ORI_KEY)
        return control_point_pos + R_flange @ CONTROL_POINT_TO_CAMERA_LOCAL
    except Exception:
        return control_point_pos


# ===================== Look-at rotation (roll-continuous) =====================
def compute_look_at_rotation(p_ee, p_target, x_ref):
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
    return np.column_stack([x_des, y_des, z_des]), x_des


# ===================== SEARCHING constants =====================
SEARCH_ENDPOINT_A = np.array([0.60, -0.35, 0.75])   # <-- replace after measuring
SEARCH_ENDPOINT_B = np.array([0.60,  0.35, 0.75])   # <-- replace after measuring
ARRIVE_TOL        = 0.03   # m

SEARCH_ORI = np.array([[ 1.0,  0.0,  0.0],
                       [ 0.0, -1.0,  0.0],
                       [ 0.0,  0.0, -1.0]])


# ===================== TRACKING constants =====================
HOVER_HEIGHT   = 0.30   # m, control point above mouse
MAX_GOAL_SPEED = 0.15   # m/s


# ===================== GRASP constants =====================
# URDF: closed_fingers_tcp = grav_base_link(=flange) + 0.20m
# GRASP_CONTROL_POINT_Z = GRASP_TIP_Z + FLANGE_TO_TIP - COMPLIANT_FRAME_Z
FLANGE_TO_TIP         = 0.20    # m, from URDF closed_fingers_tcp (calibrate empirically)
GRASP_TIP_Z           = 0.03    # m, target gripper tip height above table
GRASP_CONTROL_POINT_Z = GRASP_TIP_Z + FLANGE_TO_TIP - COMPLIANT_FRAME_Z
# = 0.03 + 0.20 - 0.15 = 0.08 m

GRASP_XY_COMP       = np.array([0.06, 0.0])   # <-- calibrate empirically

GRASP_TRIGGER_HORIZ = 0.05   # m
GRASP_TRIGGER_TIME  = 3.0    # s
GRASP_ABORT_HORIZ   = 0.08   # m
GRASP_DONE_TOL      = 0.02   # m

GRASP_LIFT_Z        = 0.30   # m, control point z to lift to after grasping
GRASP_LIFT_TOL      = 0.03   # m


# ===================== SEARCHING state =====================
def enter_searching(ctx):
    ctx["search_target"] = SEARCH_ENDPOINT_A
    write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)
    redis_client.set(OTG_LINEAR_VEL_KEY, json.dumps([OTG_LINEAR_VEL_NORMAL]))
    print("[state] -> SEARCHING")


def do_searching(ctx):
    detected, _ = read_perception_raw()
    if not DEBUG_LOCK_SEARCHING and detected:
        return "TRACKING"

    target = ctx["search_target"]
    write_vector(EE_DESIRED_POS_KEY, target)
    write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)

    ee_pos = read_vector(EE_POS_KEY)
    if np.linalg.norm(target - ee_pos) < ARRIVE_TOL:
        if np.allclose(target, SEARCH_ENDPOINT_A):
            ctx["search_target"] = SEARCH_ENDPOINT_B
            print("[search] reached A, heading to B")
        else:
            ctx["search_target"] = SEARCH_ENDPOINT_A
            print("[search] reached B, heading to A")

    return "SEARCHING"


# ===================== TRACKING state =====================
def enter_tracking(ctx):
    ctx["goal_filtered"]  = read_vector(EE_POS_KEY)
    ctx["x_ref"]          = np.array([1.0, 0.0, 0.0])
    ctx["overhead_since"] = None
    print("[state] -> TRACKING")


def do_tracking(ctx):
    detected_raw, mouse_raw = read_perception_raw()
    if not detected_raw:
        print("[track] mouse lost -> SEARCHING")
        return "SEARCHING"

    control_point_pos = read_vector(EE_POS_KEY)
    camera_pos        = get_camera_world_pos(control_point_pos)
    horiz             = np.linalg.norm((mouse_raw - control_point_pos)[:2])

    if horiz < GRASP_TRIGGER_HORIZ:
        if ctx["overhead_since"] is None:
            ctx["overhead_since"] = time.time()
            print(f"[track] overhead (horiz={horiz:.3f}), 3s timer started")
        elif time.time() - ctx["overhead_since"] >= GRASP_TRIGGER_TIME:
            print("[track] held overhead 3s -> GRASP")
            return "GRASP"
    else:
        if ctx["overhead_since"] is not None:
            print(f"[track] lost overhead (horiz={horiz:.3f}), timer reset")
        ctx["overhead_since"] = None

    goal_raw = mouse_raw.copy()
    goal_raw[2] += HOVER_HEIGHT
    delta    = goal_raw - ctx["goal_filtered"]
    max_step = MAX_GOAL_SPEED / LOOP_HZ
    nrm = np.linalg.norm(delta)
    if nrm > max_step:
        delta = max_step * delta / nrm
    ctx["goal_filtered"] = ctx["goal_filtered"] + delta

    R_des, ctx["x_ref"] = compute_look_at_rotation(camera_pos, mouse_raw, ctx["x_ref"])
    write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])
    if R_des is not None:
        write_matrix(EE_DESIRED_ORI_KEY, R_des)

    return "TRACKING"


# ===================== GRASP state =====================
# ctx["grasp_phase"] tracks sub-steps:
#   "opening"    wait for gripper to open
#   "descending" move down to GRASP_CONTROL_POINT_Z
#   "closing"    wait for gripper to close
#   "lifting"    move up to GRASP_LIFT_Z
#   "releasing"  wait for gripper to open, then done

def enter_grasp(ctx):
    _, mouse_pos = read_perception_raw()
    ctx["grasp_xy"]      = mouse_pos[:2].copy() + GRASP_XY_COMP
    ctx["goal_filtered"] = read_vector(EE_POS_KEY)
    ctx["grasp_phase"]   = "opening"
    ctx["phase_t0"]      = time.time()
    write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)
    redis_client.set(OTG_LINEAR_VEL_KEY, json.dumps([OTG_LINEAR_VEL_GRASP]))
    gripper_open()
    print(f"[state] -> GRASP | phase=opening | xy={ctx['grasp_xy']} "
          f"target_z={GRASP_CONTROL_POINT_Z:.3f}")


def do_grasp(ctx):
    detected, mouse_pos = read_perception_raw()
    phase = ctx["grasp_phase"]

    # Phase 1: wait for gripper to open
    if phase == "opening":
        if time.time() - ctx["phase_t0"] >= GRIPPER_WAIT:
            ctx["grasp_phase"] = "descending"
            print("[grasp] gripper open -> descending")
        return "GRASP"

    # Phase 2: descend
    if phase == "descending":
        if not detected:
            print("[grasp] mouse lost during descent -> SEARCHING (fail)")
            return "SEARCHING"
        target_xy = mouse_pos[:2] + GRASP_XY_COMP
        if np.linalg.norm(target_xy - ctx["grasp_xy"]) > GRASP_ABORT_HORIZ:
            print("[grasp] mouse drifted -> SEARCHING (fail)")
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
        write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)

        ee_pos = read_vector(EE_POS_KEY)
        if abs(ee_pos[2] - GRASP_CONTROL_POINT_Z) < GRASP_DONE_TOL:
            ctx["grasp_phase"] = "closing"
            ctx["phase_t0"]    = time.time()
            gripper_close()
            print("[grasp] reached target z -> closing gripper")
        return "GRASP"

    # Phase 3: wait for gripper to close
    if phase == "closing":
        if time.time() - ctx["phase_t0"] >= GRIPPER_WAIT:
            ctx["grasp_phase"] = "lifting"
            print("[grasp] gripper closed -> lifting")
        return "GRASP"

    # Phase 4: lift back up
    if phase == "lifting":
        lift_goal = np.array([ctx["grasp_xy"][0],
                              ctx["grasp_xy"][1],
                              GRASP_LIFT_Z])
        delta    = lift_goal - ctx["goal_filtered"]
        max_step = MAX_GOAL_SPEED / LOOP_HZ
        nrm = np.linalg.norm(delta)
        if nrm > max_step:
            delta = max_step * delta / nrm
        ctx["goal_filtered"] = ctx["goal_filtered"] + delta
        write_vector(EE_DESIRED_POS_KEY, ctx["goal_filtered"])
        write_matrix(EE_DESIRED_ORI_KEY, SEARCH_ORI)

        ee_pos = read_vector(EE_POS_KEY)
        if abs(ee_pos[2] - GRASP_LIFT_Z) < GRASP_LIFT_TOL:
            ctx["grasp_phase"] = "releasing"
            ctx["phase_t0"]    = time.time()
            gripper_open()
            print("[grasp] lifted -> releasing mouse")
        return "GRASP"

    # Phase 5: wait for gripper to open, then done
    if phase == "releasing":
        if time.time() - ctx["phase_t0"] >= GRIPPER_WAIT:
            print("[grasp] released -> SEARCHING (success)")
            return "SEARCHING"
        return "GRASP"

    return "SEARCHING"   # fallback


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

    # ── Safety check: confirm correct config file is loaded ──────────────────
    config_file = redis_client.get(CONFIG_FILE_KEY)
    if config_file is None:
        raise RuntimeError("Config file key not found -- is opensai running?")
    config_file = config_file.decode("utf-8")
    if config_file != EXPECTED_CONFIG_FILE:
        raise RuntimeError(f"Wrong config file: got '{config_file}', "
                           f"expected '{EXPECTED_CONFIG_FILE}'. "
                           f"Please load the correct config in opensai.")
    print(f"Config file confirmed: {config_file}")

    # ── Switch to cartesian_controller and wait for confirmation ─────────────
    print("Switching to cartesian_controller...")
    while redis_client.get(ACTIVE_CONTROLLER_KEY).decode("utf-8") != "cartesian_controller":
        redis_client.set(ACTIVE_CONTROLLER_KEY, "cartesian_controller")
    print("cartesian_controller active")

    # ── Wait for EE position key to appear ───────────────────────────────────
    if not wait_for_key(EE_POS_KEY):
        raise RuntimeError(f"{EE_POS_KEY} never appeared -- is the opensai "
                           f"controller running?")

    ctx   = {}
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
                    if nxt == "GRASP":
                        enter_grasp(ctx)
                    elif nxt == "SEARCHING":
                        enter_searching(ctx)

            elif state == "GRASP":
                nxt = do_grasp(ctx)
                if nxt != state:
                    state = nxt
                    enter_searching(ctx)

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        print("\nState machine stopped.")


if __name__ == "__main__":
    main()
