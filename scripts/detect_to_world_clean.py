import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import redis
import ast
import threading

redis_client = redis.Redis(decode_responses=True)
LINK7_TRANSFORM_KEY = "opensai::sensors::Titania::link7_transform"
FLANGE_TRANSFORM_KEY = "opensai::sensors::Titania::flange_transform"

def worker(stop_event, delay):
    redis_client_local = redis.Redis(decode_responses=True)
    time.sleep(delay)

    while not stop_event.is_set():
        # TCP
        T_flange_to_base_frame = get_redis_transform(
            redis_client_local,
            FLANGE_TRANSFORM_KEY
        )
        T_tcp_to_flange = np.eye(4)
        T_tcp_to_flange[:3, -1] = np.array([0, 0, 0.20])  # from CAD

        T_tcp_to_base_frame = T_flange_to_base_frame @ T_tcp_to_flange

        print_transform("T_tcp_to_base_frame", T_tcp_to_base_frame)

        stop_event.wait(0.1)


def get_redis_transform(client, key):
    matrix_str = client.get(key)

    if matrix_str is None:
        raise RuntimeError(f"Redis key not found: {key}")

    return string_to_4x4_numpy(matrix_str)


def string_to_4x4_numpy(matrix_str):
    """
    Convert a string representation of a 4x4 matrix to a NumPy array.
    """
    matrix = np.array(ast.literal_eval(matrix_str), dtype=float)

    if matrix.shape != (4, 4):
        raise ValueError("Input must represent a 4x4 matrix.")

    return matrix

# ============================================================
# CONFIG
# ============================================================

def rot_x(theta):
    """
    Rotation matrix around X-axis.
    theta: angle in radians
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])


def rot_y(theta):
    """
    Rotation matrix around Y-axis.
    theta: angle in radians
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(theta):
    """
    Rotation matrix around Z-axis.
    theta: angle in radians
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

MODEL_PATH = "../models/best.pt"

# Rotation matrix from camera frame to EE frame
# (3x3 rotation matrix)
# CAM_ROT_MATRIX = np.array([
# [0,  np.cos(np.deg2rad(20)),  np.cos(np.deg2rad(110))],
#    [-1,            0,      0],
#    [0,         np.cos(np.deg2rad(110)), np.sin(np.deg2rad(20))]
# ])
CAM_ROT_MATRIX = rot_y(np.deg2rad(-20)) @ rot_z(np.deg2rad(90))

# Translation vector from camera frame to EE frame
# (3x1 translation vector in meters)
CAM_TRANS_VECTOR = np.array([
    0.074,
    -0.01,
    0.136
])


# ============================================================
# Pretty print
# ============================================================
def print_transform(name, T):

    print(f"\n{name}:")
    print(np.array_str(T, precision=4, suppress_small=True))


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")


# ============================================================
# START REALSENSE
# ============================================================

print("Starting RealSense D405...")

pipeline = rs.pipeline()

config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    15
)

config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    15
)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

# Camera intrinsics
color_stream = profile.get_stream(rs.stream.color)

intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

fx = intrinsics.fx
fy = intrinsics.fy
cx = intrinsics.ppx
cy = intrinsics.ppy

print("\nCamera intrinsics:")
print(f"fx = {fx}")
print(f"fy = {fy}")
print(f"cx = {cx}")
print(f"cy = {cy}")


# ============================================================
# CAMERA IN EE TRANSFORM
# ============================================================

T_ee_camera = np.eye(4)
T_ee_camera[:3, :3] = CAM_ROT_MATRIX
T_ee_camera[:3, 3] = CAM_TRANS_VECTOR

print_transform("T_ee_camera", T_ee_camera)


# ============================================================
# MAIN LOOP
# ============================================================

print("\nControls:")
print("  SPACE = detect object + compute world position")
print("  q     = quit")

stop_event = threading.Event()
t1 = threading.Thread(target=worker, args=(stop_event, 1), daemon=True)
t1.start()

try:

    while True:

        # ----------------------------------------------------
        # Get frames
        # ----------------------------------------------------
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()

        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        display = color_image.copy()

        cv2.putText(
            display,
            "SPACE = detect object",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("D405", display)

        key = cv2.waitKey(1)

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------
        if key == ord('q'):
            break

        # ----------------------------------------------------
        # DETECT OBJECT
        # ----------------------------------------------------
        if key == 32:  # SPACE

            print("\nRunning detection...")

            # ------------------------------------------------
            # Current robot transforms from Redis
            # ------------------------------------------------
            T_link7_to_base_frame = get_redis_transform(
                redis_client,
                LINK7_TRANSFORM_KEY
            )

            T_camera_to_link7 = np.eye(4)
            T_camera_to_link7[:3, :3] = CAM_ROT_MATRIX
            T_camera_to_link7[:3, -1] = CAM_TRANS_VECTOR

            T_world_camera = T_link7_to_base_frame @ T_camera_to_link7

            # TCP
            T_flange_to_base_frame = get_redis_transform(
                redis_client,
                FLANGE_TRANSFORM_KEY
            )
            T_tcp_to_flange = np.eye(4)
            T_tcp_to_flange[:3, -1] = np.array([0, 0, 0.20])  # from CAD

            T_tcp_to_base_frame = T_flange_to_base_frame @ T_tcp_to_flange

            print_transform(
                "T_world_camera",
                T_world_camera
            )

            print_transform(
                "T_tcp_to_base_frame",
                T_tcp_to_base_frame
            )

            # ------------------------------------------------
            # YOLO inference
            # ------------------------------------------------
            results = model(color_image)

            annotated = color_image.copy()

            found = False

            for r in results:

                boxes = r.boxes

                for box in boxes:

                    found = True

                    # ----------------------------------------
                    # Bounding box
                    # ----------------------------------------
                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    # ----------------------------------------
                    # Center pixel
                    # ----------------------------------------
                    center_x = int((x1 + x2) / 2)

                    center_y = int((y1 + y2) / 2)

                    # ----------------------------------------
                    # Depth
                    # ----------------------------------------
                    depth = depth_frame.get_distance(
                        center_x,
                        center_y
                    )

                    # Skip invalid depth
                    if depth <= 0:
                        continue

                    # ----------------------------------------
                    # Pixel -> camera coordinates
                    # ----------------------------------------
                    Z = depth

                    X = (
                        (center_x - cx)
                        * Z
                        / fx
                    )

                    Y = (
                        (center_y - cy)
                        * Z
                        / fy
                    )

                    # ----------------------------------------
                    # Point in camera frame
                    # ----------------------------------------
                    p_cam = np.array([
                        X,
                        Y,
                        Z,
                        1.0
                    ])

                    # ----------------------------------------
                    # Camera -> world
                    # ----------------------------------------
                    p_world = (
                        T_world_camera @ p_cam
                    )

                    # ----------------------------------------
                    # PRINT RESULTS
                    # ----------------------------------------
                    print("\n================================")
                    print("DETECTION")
                    print("================================")

                    print("\nPixel:")
                    print(f"({center_x}, {center_y})")

                    print("\nCamera frame:")
                    print(
                        f"X = {X:.4f} m"
                    )
                    print(
                        f"Y = {Y:.4f} m"
                    )
                    print(
                        f"Z = {Z:.4f} m"
                    )

                    print("\nWorld frame:")
                    print(
                        f"X = {p_world[0]:.4f} m"
                    )
                    print(
                        f"Y = {p_world[1]:.4f} m"
                    )
                    print(
                        f"Z = {p_world[2]:.4f} m"
                    )

                    print("================================")

                    # ----------------------------------------
                    # Draw visualization
                    # ----------------------------------------
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        annotated,
                        (center_x, center_y),
                        6,
                        (0, 0, 255),
                        -1
                    )

                    text = (
                        f"W: "
                        f"{p_world[0]:.3f}, "
                        f"{p_world[1]:.3f}, "
                        f"{p_world[2]:.3f}"
                    )

                    cv2.putText(
                        annotated,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            if not found:
                print("\nNo objects detected.")

            cv2.imshow(
                "Detection Result",
                annotated
            )

finally:

    stop_event.set()
    t1.join(timeout=1)

    pipeline.stop()

    cv2.destroyAllWindows()

    print("\nStopped.")
