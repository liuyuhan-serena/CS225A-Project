import cv2
import numpy as np
import pyrealsense2 as rs
import pickle

# ============================================================
# CHARUCO BOARD PARAMETERS
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

board = cv2.aruco.CharucoBoard(
    (7, 5),
    0.04,
    0.03,
    aruco_dict
)

detector_params = cv2.aruco.DetectorParameters()

# ============================================================
# REALSENSE SETUP
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

# Allow auto exposure to stabilize
for _ in range(30):
    pipeline.wait_for_frames()

# ============================================================
# STORAGE
# ============================================================

all_charuco_corners = []
all_charuco_ids = []

image_size = None

print("\n===================================")
print("CHARUCO CAMERA CALIBRATION")
print("===================================")

print("\nControls:")
print("SPACE = capture frame")
print("Q = finish calibration\n")

print("Capture recommendations:")
print("- Tilt board strongly")
print("- Use image corners")
print("- Vary depth")
print("- Rotate board")
print("- Capture 25-40 frames\n")

# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Improves paper-board detection
        gray = cv2.equalizeHist(gray)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detector_params
        )

        display = image.copy()

        retval = 0

        print("markers:", 0 if ids is None else len(ids), "charuco:", retval)

        if ids is not None:
            print("Detected IDs:", ids.flatten())

        if ids is not None and len(ids) > 0:

            cv2.aruco.drawDetectedMarkers(
                display,
                corners,
                ids
            )

            retval, charuco_corners, charuco_ids = (
                cv2.aruco.interpolateCornersCharuco(
                    corners,
                    ids,
                    gray,
                    board
                )
            )

            # More strict threshold for robustness
            if retval > 15:

                cv2.aruco.drawDetectedCornersCharuco(
                    display,
                    charuco_corners,
                    charuco_ids
                )

                cv2.putText(
                    display,
                    f"GOOD DETECTION ({retval} corners)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    display,
                    f"Need more corners ({retval})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        cv2.putText(
            display,
            f"Captured Frames: {len(all_charuco_corners)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imshow("Charuco Calibration", display)

        key = cv2.waitKey(1)

        # ====================================================
        # CAPTURE FRAME
        # ====================================================

        if key == ord(' '):

            if ids is not None and retval > 6:

                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

                image_size = gray.shape[::-1]

                print(
                    f"Captured frame "
                    f"{len(all_charuco_corners)} "
                    f"({retval} corners)"
                )

            else:

                print("Not enough corners detected")

        # ====================================================
        # FINISH
        # ====================================================

        elif key == ord('q'):
            break

# ============================================================
# CLEANUP
# ============================================================

finally:

    pipeline.stop()
    cv2.destroyAllWindows()

# ============================================================
# VALIDATION
# ============================================================

if len(all_charuco_corners) < 10:

    print("\nERROR:")
    print("Need at least 10 valid captures")
    exit()

# ============================================================
# CALIBRATION
# ============================================================

print("\n===================================")
print("RUNNING CALIBRATION")
print("===================================")

retval, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None
    )
)

# ============================================================
# RESULTS
# ============================================================

print("\n===================================")
print("CALIBRATION RESULTS")
print("===================================")

print("\nReprojection Error:")
print(retval)

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)

# ============================================================
# FACTORY INTRINSICS
# ============================================================

intr = profile.get_stream(
    rs.stream.color
).as_video_stream_profile().get_intrinsics()

print("\n===================================")
print("FACTORY REALSENSE INTRINSICS")
print("===================================")

print(f"\nfx:  {intr.fx}")
print(f"fy:  {intr.fy}")
print(f"ppx: {intr.ppx}")
print(f"ppy: {intr.ppy}")

factory_dist = np.array(intr.coeffs)

factory_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])

print("\nFactory Camera Matrix:")
print(factory_matrix)

print("\nFactory Distortion:")
print(factory_dist)

# ============================================================
# COMPARISON
# ============================================================

print("\n===================================")
print("CALIBRATION DIFFERENCE")
print("===================================")

print("\nDelta fx:")
print(camera_matrix[0, 0] - intr.fx)

print("\nDelta fy:")
print(camera_matrix[1, 1] - intr.fy)

print("\nDelta cx:")
print(camera_matrix[0, 2] - intr.ppx)

print("\nDelta cy:")
print(camera_matrix[1, 2] - intr.ppy)

# ============================================================
# SAVE
# ============================================================

with open("camera_calibration.pkl", "wb") as f:

    pickle.dump({
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "reprojection_error": retval,
        "factory_matrix": factory_matrix,
        "factory_dist": factory_dist
    }, f)

print("\n===================================")
print("SAVED")
print("===================================")

print("\nSaved:")
print("camera_calibration.pkl")