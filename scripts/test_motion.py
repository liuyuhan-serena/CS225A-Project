import flexivrdk
import time
import copy
import math

ROBOT_SN = "Rizon4s-063394"

SAFE_MOVE_DISTANCE = 0.005   # 5 mm

try:
    print("Connecting...")
    robot = flexivrdk.Robot(ROBOT_SN)

    print("Connected")

    # -------------------------------------------------
    # Safety checks
    # -------------------------------------------------

    if not robot.estop_released():
        raise RuntimeError("E-stop is pressed")

    if robot.fault():
        print("Fault detected, clearing...")
        robot.ClearFault()
        time.sleep(2)

    print("Enabling robot...")
    robot.Enable()

    while not robot.operational():
        print("Waiting for operational state...")
        time.sleep(0.1)

    print("Robot operational")

    # -------------------------------------------------
    # Read current TCP pose
    # -------------------------------------------------

    state = robot.states()

    current_pose = list(state.tcp_pose)

    print("\nCurrent TCP pose:")
    print(current_pose)

    # Format:
    # [x, y, z, qw, qx, qy, qz]

    # -------------------------------------------------
    # Build tiny safe motion
    # -------------------------------------------------

    target_pose = copy.deepcopy(current_pose)

    # Move UP 5 mm
    target_pose[2] += SAFE_MOVE_DISTANCE

    print("\nTarget pose:")
    print(target_pose)

    # -------------------------------------------------
    # Safety distance check
    # -------------------------------------------------

    dx = target_pose[0] - current_pose[0]
    dy = target_pose[1] - current_pose[1]
    dz = target_pose[2] - current_pose[2]

    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

    print(f"\nMotion distance: {dist:.4f} m")

    if dist > 0.02:
        raise RuntimeError("Refusing motion > 2 cm")

    # -------------------------------------------------
    # Execute motion
    # -------------------------------------------------

    print("\nExecuting slow MoveL...")

    robot.ExecutePrimitive(
        "MoveL",
        {
            "target": target_pose,
            "maxVel": 0.02,   # 2 cm/s
            "maxAcc": 0.05
        }
    )

    print("Motion command sent")

except Exception as e:
    print("\nERROR:")
    print(e)