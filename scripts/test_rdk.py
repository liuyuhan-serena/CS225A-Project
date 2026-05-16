import flexivrdk
import time

ROBOT_SN = "Rizon4s-063394"

try:
    print("Connecting to robot...")

    robot = flexivrdk.Robot(ROBOT_SN)

    print("Connected.")
    time.sleep(1)  # allow DDS state update

    state = robot.states()

    print("\nJoint positions (q):")
    print(state.q)

    print("\nTCP pose [x, y, z, qw, qx, qy, qz]:")
    print(state.tcp_pose)

except Exception as e:
    print("ERROR:")
    print(e)