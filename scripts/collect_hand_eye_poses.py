import cv2
import numpy as np
import pickle
import pyrealsense2 as rs
import flexivrdk  # pip install flexivrdk

# ── Robot + camera setup ───────────────────────────────────────────────────
robot = flexivrdk.Robot("Rizon4s-XXXXXX")   # your serial number
robot.enable()
while not robot.operational():
    pass

pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board       = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.03, aruco_dict)
cam_matrix  = None   # filled after first detection
dist_coeffs = None

R_gripper2base = []
t_gripper2base = []
R_target2cam   = []
t_target2cam   = []

print("Move robot to a pose, then press SPACE to capture. Need 15+ poses. ESC to finish.")

while True:
    frames = pipe.wait_for_frames()
    color  = np.asanyarray(frames.get_color_frame().get_data())
    gray   = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)

        if charuco_corners is not None and len(charuco_corners) >= 6:
            # Get intrinsics from RealSense (first frame only)
            if cam_matrix is None:
                intr = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
                cam_matrix  = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])
                dist_coeffs = np.array(intr.coeffs)

            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, cam_matrix, dist_coeffs,
                None, None)

            if ok:
                cv2.drawFrameAxes(color, cam_matrix, dist_coeffs, rvec, tvec, 0.05)
                cv2.putText(color, f"Captured: {len(R_gripper2base)}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Calibration", color)
    key = cv2.waitKey(1)

    if key == 32 and ids is not None and ok:  # SPACE
        # ── Robot flange pose → R, t ───────────────────────────────────────
        pose = robot.states().flange_pose  # [x,y,z, qw,qx,qy,qz]
        q    = np.array([pose[4], pose[5], pose[6], pose[3]])  # xyzw for scipy
        from scipy.spatial.transform import Rotation
        R_gb = Rotation.from_quat(q).as_matrix()
        t_gb = np.array([[pose[0]], [pose[1]], [pose[2]]])

        R_gripper2base.append(R_gb)
        t_gripper2base.append(t_gb)
        R_target2cam.append(cv2.Rodrigues(rvec)[0])
        t_target2cam.append(tvec)
        print(f"  Pose {len(R_gripper2base)} captured ✓")

    elif key == 27:  # ESC
        break

pipe.stop()
cv2.destroyAllWindows()

# ── Save raw data in case you need to re-run calibration ──────────────────
with open("calibration_data.pkl", "wb") as f:
    pickle.dump((R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
                 cam_matrix, dist_coeffs), f)
print("Data saved to calibration_data.pkl")