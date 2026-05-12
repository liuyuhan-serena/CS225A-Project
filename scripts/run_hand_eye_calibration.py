import cv2, pickle, numpy as np

with open("calibration_data.pkl", "rb") as f:
    R_g2b, t_g2b, R_t2c, t_t2c, K, dist = pickle.load(f)

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_g2b, t_g2b,
    R_t2c, t_t2c,
    method=cv2.CALIB_HAND_EYE_TSAI   # most robust for eye-in-hand
)

print("R_cam2gripper:\n", R_cam2gripper)
print("t_cam2gripper:\n", t_cam2gripper)

np.save("R_cam2gripper.npy", R_cam2gripper)
np.save("t_cam2gripper.npy", t_cam2gripper)
print("Saved R_cam2gripper.npy and t_cam2gripper.npy")