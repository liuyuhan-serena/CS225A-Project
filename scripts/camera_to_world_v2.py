import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import flexivrdk


ROBOT_SN = "Rizon4s-063394"


# ============================================================
# Convert Flexiv pose -> 4x4 homogeneous transform
#
# Input:
# [x, y, z, qw, qx, qy, qz]
#
# Output:
# 4x4 transform matrix
# ============================================================
def pose_to_matrix(pose):

    x, y, z, qw, qx, qy, qz = pose

    T = np.eye(4)

    # scipy uses [qx, qy, qz, qw]
    rot = R.from_quat([qx, qy, qz, qw])

    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [x, y, z]

    return T


# ============================================================
# Build homogeneous transform from Euler XYZ + translation
# ============================================================
def euler_xyz_to_matrix(rx_deg, ry_deg, rz_deg, translation):

    T = np.eye(4)

    rot = R.from_euler(
        'xyz',
        [rx_deg, ry_deg, rz_deg],
        degrees=True
    )

    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = translation

    return T


# ============================================================
# Pretty print transform
# ============================================================
def print_transform(name, T):

    print(f"\n{name}:")
    print(np.array_str(T, precision=4, suppress_small=True))


# ============================================================
# MAIN
# ============================================================
try:

    print("Connecting to robot...")

    robot = flexivrdk.Robot(ROBOT_SN)

    print("Connected.")

    # allow state update
    time.sleep(1)

    state = robot.states()

    # ========================================================
    # WORLD -> FLANGE
    #
    # From robot state:
    # [x, y, z, qw, qx, qy, qz]
    # ========================================================
    T_world_flange = pose_to_matrix(state.tcp_pose)

    # ========================================================
    # FLANGE -> CAMERA
    #
    # Your manually measured camera pose
    # ========================================================
    T_flange_camera = euler_xyz_to_matrix(
        90.0,
        180.0,
        90.0,
        [0.045, 0.0, 0.046]
    )

    # ========================================================
    # WORLD -> CAMERA
    # ========================================================
    T_world_camera = T_world_flange @ T_flange_camera

    # ========================================================
    # PRINT RESULTS
    # ========================================================
    np.set_printoptions(precision=4, suppress=True)

    print_transform("T_world_flange", T_world_flange)

    print_transform("T_flange_camera", T_flange_camera)

    print_transform("T_world_camera", T_world_camera)

    # ========================================================
    # Camera position in world
    # ========================================================
    cam_pos_world = T_world_camera[:3, 3]

    print("\nCamera position in world:")
    print(cam_pos_world)

    # ========================================================
    # Example:
    # Transform a point from camera frame -> world frame
    #
    # Example point:
    # 10 cm in front of camera optical center
    # ========================================================
    p_cam = np.array([
        0.0,
        0.0,
        0.10,
        1.0
    ])

    p_world = T_world_camera @ p_cam

    print("\nExample point in camera frame:")
    print(p_cam[:3])

    print("\nPoint transformed into world frame:")
    print(p_world[:3])

except Exception as e:

    print("\nERROR:")
    print(e)