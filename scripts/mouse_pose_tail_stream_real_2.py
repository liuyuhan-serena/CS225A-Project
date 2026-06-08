import argparse
import ast
import json
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import redis
from ultralytics import YOLO


LINK7_TRANSFORM_KEY = "opensai::sensors::Titania::link7_transform"

MOUSE_CENTER_WORLD_KEY = "opensai::perception::mouse_center_world"
MOUSE_TAILBASE_WORLD_KEY = "opensai::perception::mouse_tailbase_world"
MOUSE_CENTER_TO_TAIL_VECTOR_WORLD_KEY = "opensai::perception::mouse_center_to_tail_vector_world"
MOUSE_POSE_METADATA_KEY = "opensai::perception::mouse_pose_metadata"
DISPLAY_WINDOW_NAME = "Mouse Pose Stream"
DISPLAY_WINDOWED_SIZE = (1280, 960)

# Legacy key consumed by the state machine's SEARCHING/TRACKING/LIFT logic.
# Mirroring the mouse center here lets the new CV replace the old detector
# (detect_world_stream_to_redis_deproject.py) so we don't need two scripts
# fighting over the same RealSense camera.
LEGACY_DESIRED_POSITION_KEY = "opensai::perception::desired_position"

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "pose_best" / "pose_best.pt")

CAM_TRANS_VECTOR = np.array([0.074, -0.01, 0.136], dtype=float)


def rot_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


CAM_ROT_MATRIX = rot_y(np.deg2rad(-20)) @ rot_z(np.deg2rad(90))


@dataclass(frozen=True)
class MousePosePacket:
    seq: int
    timestamp: float
    center_world: list | None
    tailbase_world: list | None
    center_to_tail_vector_world: list | None
    confidence: float | None
    center_pixel: list | None
    tailbase_pixel: list | None
    box_xyxy: list | None
    center_depth: float | None
    tailbase_depth: float | None
    raw_center_world: list | None = None
    raw_tailbase_world: list | None = None
    center_outlier_rejected: bool = False
    tailbase_outlier_rejected: bool = False
    center_filter_median: list | None = None
    tailbase_filter_median: list | None = None


def string_to_4x4_numpy(matrix_str):
    matrix = np.array(ast.literal_eval(matrix_str), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Input must represent a 4x4 matrix.")
    return matrix

def get_redis_transform(client, key):
    matrix_str = client.get(key)
    if matrix_str is None:
        raise RuntimeError(f"Redis key not found: {key}")
    return string_to_4x4_numpy(matrix_str)


def python_list_string(values):
    return json.dumps(values, separators=(",", ":"))


class MedianOutlierFilter:
    def __init__(self, window_size, max_distance, reset_after_rejections):
        self.window = deque(maxlen=max(1, window_size))
        self.max_distance = max_distance
        self.reset_after_rejections = max(1, reset_after_rejections)
        self.rejections = 0

    def update(self, point):
        if point is None:
            current_median = self.median()
            median_list = current_median.tolist() if current_median is not None else None
            return None, False, median_list

        point_array = np.array(point, dtype=float)
        current_median = self.median()

        if current_median is not None:
            distance = float(np.linalg.norm(point_array - current_median))

            if distance > self.max_distance:
                self.rejections += 1

                if self.rejections < self.reset_after_rejections:
                    return current_median.tolist(), True, current_median.tolist()

                self.window.clear()

        self.rejections = 0
        self.window.append(point_array)
        updated_median = self.median()

        return updated_median.tolist(), False, updated_median.tolist()

    def median(self):
        if not self.window:
            return None

        return np.median(np.stack(self.window, axis=0), axis=0)


class HampelEmaFilter:
    """Low-lag, outlier-rejecting position filter for the TRACKING center signal.

    Two responsibilities, deliberately decoupled:

    1. Outlier rejection (kept identical to MedianOutlierFilter, since it already
       kills false detections on similar-colored objects): a sample farther than
       `max_distance` from the running window MEDIAN is treated as a spike and
       dropped; after `reset_after_rejections` consecutive drops the window resets
       (the mouse genuinely teleported / a new object is the real target).

    2. Output smoothing: instead of publishing the window MEDIAN (which lags a
       moving target by ~window/2 frames -- the cause of the sluggish tracking),
       it publishes an EMA of the ACCEPTED RAW samples. The median is used ONLY as
       the robust reference for step 1, never as the output. The EMA tracks with
       ~1-frame lag and its responsiveness is tunable via `ema_alpha`
       (higher = faster + less smooth, lower = slower + smoother). This is what
       makes TRACKING both faster and smooth: spikes are rejected (no smoothing
       penalty on inliers), and the in-track signal is lightly low-passed rather
       than median-delayed.

    Used for the mouse CENTER only. The tailbase keeps the heavier MedianOutlier-
    Filter on purpose -- PRE_GRASP/GRASP want a rock-steady target during the hold,
    where lag does not matter (the mouse is meant to be stationary there).
    """

    def __init__(self, window_size, max_distance, reset_after_rejections, ema_alpha):
        self.window = deque(maxlen=max(1, window_size))
        self.max_distance = max_distance
        self.reset_after_rejections = max(1, reset_after_rejections)
        self.ema_alpha = float(ema_alpha)
        self.rejections = 0
        self.ema = None

    def update(self, point):
        if point is None:
            # No detection this frame: report NOT-detected (None), exactly like
            # MedianOutlierFilter. Do NOT hold the stale EMA here -- holding it
            # makes the state machine believe the mouse is still present, so
            # TRACKING never reports "lost" and keeps re-entering PRE_GRASP, which
            # then bounces back on the (correctly) missing tail. The EE ends up
            # camped on a phantom center, shaking between the two states.
            # (Spike rejection below still holds the EMA -- that is a different
            # case: a bad sample DURING an otherwise live detection stream.)
            current_median = self.median()
            median_list = current_median.tolist() if current_median is not None else None
            return None, False, median_list

        point_array = np.array(point, dtype=float)
        current_median = self.median()

        if current_median is not None:
            distance = float(np.linalg.norm(point_array - current_median))

            if distance > self.max_distance:
                self.rejections += 1

                if self.rejections < self.reset_after_rejections:
                    # Reject the spike: hold the last EMA output so a false
                    # detection can't yank the goal. Median/window unchanged.
                    held = (self.ema.tolist() if self.ema is not None
                            else current_median.tolist())
                    return held, True, current_median.tolist()

                # Too many consecutive rejections -> the target really moved.
                self.window.clear()
                self.ema = None

        self.rejections = 0
        self.window.append(point_array)

        # EMA on ACCEPTED RAW samples -> low-lag, smooth output (NOT the median).
        if self.ema is None:
            self.ema = point_array.copy()
        else:
            self.ema = (self.ema_alpha * point_array
                        + (1.0 - self.ema_alpha) * self.ema)

        updated_median = self.median()
        return self.ema.tolist(), False, updated_median.tolist()

    def median(self):
        if not self.window:
            return None

        return np.median(np.stack(self.window, axis=0), axis=0)


def put_latest(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def sample_depth(depth_frame, center_x, center_y, radius):
    depths = []
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            depth = depth_frame.get_distance(x, y)
            if depth > 0:
                depths.append(depth)

    if not depths:
        return 0.0

    return float(np.median(depths))


def deproject_to_world(pixel, depth_frame, intrinsics, T_world_camera, radius):
    u, v = int(pixel[0]), int(pixel[1])

    depth = sample_depth(depth_frame, u, v, radius)
    if depth <= 0:
        return None, 0.0

    X, Y, Z = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [u, v],
        depth
    )

    p_cam = np.array([X, Y, Z, 1.0], dtype=float)
    p_world = T_world_camera @ p_cam

    return [
        float(p_world[0]),
        float(p_world[1]),
        float(p_world[2])
    ], depth


def compute_mouse_pose_world(
    results,
    depth_frame,
    intrinsics,
    T_world_camera,
    depth_sample_radius,
    confidence_threshold,
    tailbase_kpt_index,
):
    best = None
    best_conf = -1.0

    for result in results:
        if result.boxes is None or result.keypoints is None:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        keypoints_xy = result.keypoints.xy.cpu().numpy()

        for i in range(len(boxes_xyxy)):
            confidence = float(confs[i])
            if confidence < confidence_threshold:
                continue

            if tailbase_kpt_index >= keypoints_xy.shape[1]:
                continue

            center_pixel = [
                int(boxes_xywh[i][0]),
                int(boxes_xywh[i][1])
            ]

            tailbase_pixel = [
                int(keypoints_xy[i][tailbase_kpt_index][0]),
                int(keypoints_xy[i][tailbase_kpt_index][1])
            ]

            center_world, center_depth = deproject_to_world(
                center_pixel,
                depth_frame,
                intrinsics,
                T_world_camera,
                depth_sample_radius
            )

            tailbase_world, tailbase_depth = deproject_to_world(
                tailbase_pixel,
                depth_frame,
                intrinsics,
                T_world_camera,
                depth_sample_radius
            )

            if center_world is None or tailbase_world is None:
                continue

            center_to_tail_vector_world = (
                np.array(tailbase_world) - np.array(center_world)
            ).tolist()

            if confidence > best_conf:
                best_conf = confidence
                best = {
                    "center_world": center_world,
                    "tailbase_world": tailbase_world,
                    "center_to_tail_vector_world": center_to_tail_vector_world,
                    "confidence": confidence,
                    "center_pixel": center_pixel,
                    "tailbase_pixel": tailbase_pixel,
                    "box_xyxy": boxes_xyxy[i].astype(int).tolist(),
                    "center_depth": center_depth,
                    "tailbase_depth": tailbase_depth,
                }

    return best


def publisher(stop_event, packets, host, port):
    redis_client = redis.Redis(
        host=host,
        port=port,
        decode_responses=True
    )

    while not stop_event.is_set():
        try:
            packet = packets.get(timeout=0.1)
        except queue.Empty:
            continue

        pipe = redis_client.pipeline(transaction=True)

        center = packet.center_world if packet.center_world is not None else []
        tail = packet.tailbase_world if packet.tailbase_world is not None else []
        vec = packet.center_to_tail_vector_world if packet.center_to_tail_vector_world is not None else []

        pipe.set(MOUSE_CENTER_WORLD_KEY, python_list_string(center))
        pipe.set(MOUSE_TAILBASE_WORLD_KEY, python_list_string(tail))
        pipe.set(MOUSE_CENTER_TO_TAIL_VECTOR_WORLD_KEY, python_list_string(vec))
        # Mirror the mouse center to the legacy key so the state machine's
        # SEARCHING / TRACKING / LIFT logic keeps working without the old CV.
        pipe.set(LEGACY_DESIRED_POSITION_KEY, python_list_string(center))

        metadata = {
            "seq": packet.seq,
            "timestamp": packet.timestamp,
            "confidence": packet.confidence,
            "center_pixel": packet.center_pixel,
            "tailbase_pixel": packet.tailbase_pixel,
            "box_xyxy": packet.box_xyxy,
            "center_depth": packet.center_depth,
            "tailbase_depth": packet.tailbase_depth,
            "raw_center_world": packet.raw_center_world,
            "raw_tailbase_world": packet.raw_tailbase_world,
            "center_outlier_rejected": packet.center_outlier_rejected,
            "tailbase_outlier_rejected": packet.tailbase_outlier_rejected,
            "center_filter_median": packet.center_filter_median,
            "tailbase_filter_median": packet.tailbase_filter_median,
            "center_key": MOUSE_CENTER_WORLD_KEY,
            "tailbase_key": MOUSE_TAILBASE_WORLD_KEY,
            "vector_key": MOUSE_CENTER_TO_TAIL_VECTOR_WORLD_KEY,
        }

        pipe.set(MOUSE_POSE_METADATA_KEY, json.dumps(metadata, separators=(",", ":")))
        pipe.execute()
        packets.task_done()


def draw_mouse_pose(image, pose):
    annotated = image.copy()

    if pose is None:
        cv2.putText(
            annotated,
            "No mouse pose detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        return annotated

    x1, y1, x2, y2 = pose["box_xyxy"]
    cx, cy = pose["center_pixel"]
    tx, ty = pose["tailbase_pixel"]

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
    cv2.putText(
        annotated,
        "CENTER",
        (cx + 8, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

    cv2.circle(annotated, (tx, ty), 6, (255, 0, 0), -1)
    cv2.putText(
        annotated,
        "TAIL",
        (tx + 8, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )

    cv2.arrowedLine(
        annotated,
        (cx, cy),
        (tx, ty),
        (0, 255, 255),
        3
    )

    label = f"conf={pose['confidence']:.2f}"
    cv2.putText(
        annotated,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    return annotated


def setup_display_window():
    cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_WINDOW_NAME, *DISPLAY_WINDOWED_SIZE)


def set_display_fullscreen(enabled):
    cv2.setWindowProperty(
        DISPLAY_WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL,
    )


def handle_display_key(stop_event, fullscreen_enabled):
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        stop_event.set()
    elif key == ord("f"):
        fullscreen_enabled = not fullscreen_enabled
        set_display_fullscreen(fullscreen_enabled)
        if not fullscreen_enabled:
            cv2.resizeWindow(DISPLAY_WINDOW_NAME, *DISPLAY_WINDOWED_SIZE)

    return fullscreen_enabled


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream YOLO11-pose mouse center/tailbase world-frame pose to Redis."
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--depth-sample-radius", type=int, default=2)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--outlier-distance", type=float, default=0.15)
    parser.add_argument("--reset-after-rejections", type=int, default=6)
    # Responsiveness of the CENTER filter's EMA output (TRACKING). Higher = faster
    # and less smooth, lower = slower and smoother. The window median is still used
    # for outlier rejection; this only controls the published (smoothed) value.
    # The tailbase keeps the heavier plain-median filter (stable grasp hold).
    parser.add_argument("--center-ema-alpha", type=float, default=0.5)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--tailbase-kpt-index", type=int, default=1)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    redis_client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        decode_responses=True
    )
    redis_client.ping()

    stop_event = threading.Event()
    packets = queue.Queue(maxsize=1)

    publisher_thread = threading.Thread(
        target=publisher,
        args=(stop_event, packets, args.redis_host, args.redis_port),
        daemon=True
    )

    def stop(_signum=None, _frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print("Loading YOLO11 pose model...")
    model = YOLO(args.model)
    print(f"Model loaded: {args.model}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, args.camera_fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.camera_fps)

    print("Starting RealSense D405...")
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    T_camera_to_link7 = np.eye(4)
    T_camera_to_link7[:3, :3] = CAM_ROT_MATRIX
    T_camera_to_link7[:3, 3] = CAM_TRANS_VECTOR

    publisher_thread.start()

    seq = 0
    last_publish_time = 0.0
    publish_period = 1.0 / args.publish_hz if args.publish_hz > 0 else 0.0
    # CENTER (TRACKING): low-lag Hampel + EMA -- reject false detections without
    # the median's group delay, so the arm tracks a moving mouse faster and smooth.
    center_filter = HampelEmaFilter(
        window_size=args.median_window,
        max_distance=args.outlier_distance,
        reset_after_rejections=args.reset_after_rejections,
        ema_alpha=args.center_ema_alpha,
    )
    # TAILBASE (PRE_GRASP / GRASP): unchanged heavy median -- the grasp hold wants
    # a rock-steady target and the mouse is stationary there, so lag is harmless.
    tailbase_filter = MedianOutlierFilter(
        window_size=args.median_window,
        max_distance=args.outlier_distance,
        reset_after_rejections=args.reset_after_rejections,
    )

    print("\nStreaming mouse pose to Redis:")
    print(f"  center:   {MOUSE_CENTER_WORLD_KEY}")
    print(f"  tailbase: {MOUSE_TAILBASE_WORLD_KEY}")
    print(f"  vector:   {MOUSE_CENTER_TO_TAIL_VECTOR_WORLD_KEY}")
    print(f"  metadata: {MOUSE_POSE_METADATA_KEY}")
    print("  q = quit")
    print("  f = toggle fullscreen")

    if not args.no_display:
        setup_display_window()

    fullscreen_enabled = False

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            now = time.time()

            if now - last_publish_time < publish_period:
                if not args.no_display:
                    cv2.imshow(DISPLAY_WINDOW_NAME, color_image)
                    fullscreen_enabled = handle_display_key(
                        stop_event,
                        fullscreen_enabled,
                    )
                continue

            last_publish_time = now

            T_link7_to_base_frame = get_redis_transform(
                redis_client,
                LINK7_TRANSFORM_KEY
            )

            T_world_camera = T_link7_to_base_frame @ T_camera_to_link7

            results = model(color_image, verbose=False)

            pose = compute_mouse_pose_world(
                results=results,
                depth_frame=depth_frame,
                intrinsics=intrinsics,
                T_world_camera=T_world_camera,
                depth_sample_radius=args.depth_sample_radius,
                confidence_threshold=args.confidence,
                tailbase_kpt_index=args.tailbase_kpt_index,
            )

            seq += 1

            if pose is None:
                filtered_center, center_rejected, center_median = center_filter.update(None)
                filtered_tailbase, tailbase_rejected, tailbase_median = tailbase_filter.update(None)

                packet = MousePosePacket(
                    seq=seq,
                    timestamp=now,
                    center_world=filtered_center,
                    tailbase_world=filtered_tailbase,
                    center_to_tail_vector_world=None,
                    confidence=None,
                    center_pixel=None,
                    tailbase_pixel=None,
                    box_xyxy=None,
                    center_depth=None,
                    tailbase_depth=None,
                    center_outlier_rejected=center_rejected,
                    tailbase_outlier_rejected=tailbase_rejected,
                    center_filter_median=center_median,
                    tailbase_filter_median=tailbase_median,
                )
                print("No valid mouse pose")
            else:
                raw_center_world = pose["center_world"]
                raw_tailbase_world = pose["tailbase_world"]
                filtered_center, center_rejected, center_median = center_filter.update(
                    raw_center_world
                )
                filtered_tailbase, tailbase_rejected, tailbase_median = tailbase_filter.update(
                    raw_tailbase_world
                )
                filtered_vector = None

                if filtered_center is not None and filtered_tailbase is not None:
                    filtered_vector = (
                        np.array(filtered_tailbase) - np.array(filtered_center)
                    ).tolist()

                packet = MousePosePacket(
                    seq=seq,
                    timestamp=now,
                    center_world=filtered_center,
                    tailbase_world=filtered_tailbase,
                    center_to_tail_vector_world=filtered_vector,
                    confidence=pose["confidence"],
                    center_pixel=pose["center_pixel"],
                    tailbase_pixel=pose["tailbase_pixel"],
                    box_xyxy=pose["box_xyxy"],
                    center_depth=pose["center_depth"],
                    tailbase_depth=pose["tailbase_depth"],
                    raw_center_world=raw_center_world,
                    raw_tailbase_world=raw_tailbase_world,
                    center_outlier_rejected=center_rejected,
                    tailbase_outlier_rejected=tailbase_rejected,
                    center_filter_median=center_median,
                    tailbase_filter_median=tailbase_median,
                )

                print(
                    "center:",
                    python_list_string(packet.center_world),
                    "tail:",
                    python_list_string(packet.tailbase_world),
                    "vector:",
                    python_list_string(packet.center_to_tail_vector_world),
                )

            put_latest(packets, packet)

            if not args.no_display:
                annotated = draw_mouse_pose(color_image, pose)
                cv2.imshow(DISPLAY_WINDOW_NAME, annotated)
                fullscreen_enabled = handle_display_key(
                    stop_event,
                    fullscreen_enabled,
                )

    finally:
        stop_event.set()
        publisher_thread.join(timeout=1.0)
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
