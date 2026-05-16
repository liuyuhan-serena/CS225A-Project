import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "../models/best.pt"

model = YOLO(MODEL_PATH)

print("Model loaded.")


# ============================================================
# START REALSENSE D405
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

# RGB
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

# Depth
config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

profile = pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Get camera intrinsics
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
# MAIN LOOP
# ============================================================

print("\nControls:")
print("  SPACE = run detection + output center position")
print("  q     = quit")

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

        # ----------------------------------------------------
        # Display live image
        # ----------------------------------------------------
        display = color_image.copy()

        cv2.putText(
            display,
            "Press SPACE to detect",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("D405 RGB", display)

        key = cv2.waitKey(1)

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------
        if key == ord('q'):
            break

        # ----------------------------------------------------
        # RUN DETECTION
        # ----------------------------------------------------
        if key == 32:  # SPACE

            print("\nRunning detection...")

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
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

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
                    # Depth at center
                    # ----------------------------------------
                    depth = depth_frame.get_distance(
                        center_x,
                        center_y
                    )

                    # meters
                    Z = depth

                    # ----------------------------------------
                    # Pixel -> camera 3D
                    #
                    # X = (u-cx)*Z/fx
                    # Y = (v-cy)*Z/fy
                    # ----------------------------------------
                    X = (center_x - cx) * Z / fx
                    Y = (center_y - cy) * Z / fy

                    # ----------------------------------------
                    # Print result
                    # ----------------------------------------
                    print("\n===================================")
                    print("Detected object center:")
                    print(f"Pixel: ({center_x}, {center_y})")
                    print(f"Depth: {Z:.4f} m")

                    print("\n3D position in CAMERA frame:")
                    print(f"X = {X:.4f} m")
                    print(f"Y = {Y:.4f} m")
                    print(f"Z = {Z:.4f} m")
                    print("===================================")

                    # ----------------------------------------
                    # Draw box
                    # ----------------------------------------
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Draw center
                    cv2.circle(
                        annotated,
                        (center_x, center_y),
                        6,
                        (0, 0, 255),
                        -1
                    )

                    text = f"({X:.3f}, {Y:.3f}, {Z:.3f})"

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

            cv2.imshow("Detection Result", annotated)

finally:

    pipeline.stop()
    cv2.destroyAllWindows()