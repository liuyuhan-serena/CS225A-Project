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

DESIRED_POSITION_KEY = "opensai::perception::desired_position"
DETECTED_POINTS_KEY = "opensai::perception::detected_world_points"
DETECTION_METADATA_KEY = "opensai::perception::detection_metadata"

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "best.pt")

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
class DetectionPacket:
    seq: int
    timestamp: float
    points: list
    raw_best_point: list | None
    best_point: list | None
    best_confidence: float | None
    boxes: list
    outlier_rejected: bool
    filter_median: list | None


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


def put_latest(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


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

        metadata = {
            "seq": packet.seq,
            "timestamp": packet.timestamp,
            "num_detections": len(packet.points),
            "raw_best_point": packet.raw_best_point,
            "best_confidence": packet.best_confidence,
            "boxes": packet.boxes,
            "outlier_rejected": packet.outlier_rejected,
            "filter_median": packet.filter_median,
            "desired_position_key": DESIRED_POSITION_KEY,
            "detected_points_key": DETECTED_POINTS_KEY,
        }

        pipe = redis_client.pipeline(transaction=True)
        pipe.set(DETECTED_POINTS_KEY, python_list_string(packet.points))
        pipe.set(DETECTION_METADATA_KEY, json.dumps(metadata, separators=(",", ":")))

        desired_position = packet.best_point if packet.best_point is not None else []
        pipe.set(DESIRED_POSITION_KEY, python_list_string(desired_position))

        pipe.execute()
        packets.task_done()


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


def compute_world_detections(
    results,
    depth_frame,
    intrinsics,
    T_world_camera,
    depth_sample_radius,
    confidence_threshold
):
    points = []
    boxes = []
    best_point = None
    best_confidence = None

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0

            if confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            depth = sample_depth(
                depth_frame,
                center_x,
                center_y,
                depth_sample_radius
            )

            if depth <= 0:
                continue

            X = (center_x - intrinsics.ppx) * depth / intrinsics.fx
            Y = (center_y - intrinsics.ppy) * depth / intrinsics.fy
            Z = depth

            p_cam = np.array([X, Y, Z, 1.0], dtype=float)
            p_world = T_world_camera @ p_cam
            point = [
                float(p_world[0]),
                float(p_world[1]),
                float(p_world[2])
            ]

            points.append(point)
            boxes.append({
                "xyxy": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "confidence": confidence,
                "depth": depth,
            })

            if best_confidence is None or confidence > best_confidence:
                best_confidence = confidence
                best_point = point

    return points, best_point, best_confidence, boxes


def draw_detections(image, boxes, points):
    annotated = image.copy()

    for box, point in zip(boxes, points):
        x1, y1, x2, y2 = box["xyxy"]
        center_x, center_y = box["center"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)

        label = f"{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.putText(
        annotated,
        f"Redis: {DESIRED_POSITION_KEY}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return annotated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream YOLO world-frame detections to Redis."
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--depth-sample-radius", type=int, default=2)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--outlier-distance", type=float, default=0.15)
    parser.add_argument("--reset-after-rejections", type=int, default=6)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--publish-hz", type=float, default=10.0)
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

    print("Loading YOLO model...")
    model = YOLO(args.model)
    print("Model loaded.")

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
    position_filter = MedianOutlierFilter(
        window_size=args.median_window,
        max_distance=args.outlier_distance,
        reset_after_rejections=args.reset_after_rejections
    )

    print("\nStreaming detections to Redis:")
    print(f"  desired position: {DESIRED_POSITION_KEY}")
    print(f"  all detections:   {DETECTED_POINTS_KEY}")
    print(f"  metadata:         {DETECTION_METADATA_KEY}")
    print("  q = quit")

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
                    cv2.imshow("Detection Stream", color_image)
                    if cv2.waitKey(1) == ord("q"):
                        stop_event.set()
                continue

            last_publish_time = now
            T_link7_to_base_frame = get_redis_transform(
                redis_client,
                LINK7_TRANSFORM_KEY
            )
            T_world_camera = T_link7_to_base_frame @ T_camera_to_link7

            results = model(color_image, verbose=False)
            points, raw_best_point, best_confidence, boxes = compute_world_detections(
                results,
                depth_frame,
                intrinsics,
                T_world_camera,
                args.depth_sample_radius,
                args.confidence
            )
            best_point, outlier_rejected, filter_median = position_filter.update(
                raw_best_point
            )

            seq += 1
            packet = DetectionPacket(
                seq=seq,
                timestamp=now,
                points=points,
                raw_best_point=raw_best_point,
                best_point=best_point,
                best_confidence=best_confidence,
                boxes=boxes,
                outlier_rejected=outlier_rejected,
                filter_median=filter_median
            )
            put_latest(packets, packet)

            if best_point is None:
                print(f"[{seq}] no valid detection")
            elif outlier_rejected:
                print(
                    f"[{seq}] rejected outlier, holding median = "
                    f"{python_list_string(best_point)}"
                )
            else:
                print(f"[{seq}] desired_position = {python_list_string(best_point)}")

            if not args.no_display:
                annotated = draw_detections(color_image, boxes, points)
                cv2.imshow("Detection Stream", annotated)

                if cv2.waitKey(1) == ord("q"):
                    stop_event.set()

    finally:
        stop_event.set()
        publisher_thread.join(timeout=1.0)
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
