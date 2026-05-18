/**
 * @file controller.cpp
 * @brief Dumb executor: tracks EE pose goals published to redis.
 *
 * SIMULATION SCAFFOLDING. Stands in for the Flexiv controller on the real
 * Rizon4s. Intentionally STATE-AGNOSTIC: it does NOT know about SEARCHING /
 * TRACKING. It only reads a desired EE position + orientation from redis and
 * executes them. All decisions live in the Python state machine.
 *
 *   Reads from redis:
 *     cs225a::project::ee::desired_pos   (Vector3d) - EE position goal
 *     cs225a::project::ee::desired_ori   (Matrix3d) - EE orientation goal
 *   Writes to redis:
 *     cs225a::project::ee::current_pos   (Vector3d) - current EE position
 */

#include <SaiModel.h>
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"

#include <iostream>
#include <string>
#include <algorithm>

using namespace std;
using namespace Eigen;
using namespace SaiPrimitives;

#include <signal.h>
bool runloop = false;
void sighandler(int) { runloop = false; }

#include "redis_keys.h"

// EE goal keys — written by the Python state machine, read here every loop.
static const string EE_DESIRED_POS_KEY = "cs225a::project::ee::desired_pos";
static const string EE_DESIRED_ORI_KEY = "cs225a::project::ee::desired_ori";
// EE state key — published here every loop, read by the Python state machine.
static const string EE_POS_KEY         = "cs225a::project::ee::current_pos";

int main()
{
    static const string robot_file = string(CS225A_URDF_FOLDER) + "/Rizon4s_Grav.urdf";

    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // Load robot, sync initial state
    auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
    robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
    robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
    robot->updateModel();

    const int dof = robot->dof();
    VectorXd command_torques = VectorXd::Zero(dof);
    MatrixXd N_prec = MatrixXd::Identity(dof, dof);

    // --------------------
    // Pose task on link7 (control point = camera mount, matches simviz extrinsics)
    const string control_link = "link7";
    const Vector3d control_point = Vector3d(0.074, -0.01, 0.136);
    Affine3d compliant_frame = Affine3d::Identity();
    compliant_frame.translation() = control_point;

    auto pose_task = std::make_shared<SaiPrimitives::MotionForceTask>(
        robot, control_link, compliant_frame);

    pose_task->setPosControlGains(400, 40, 0);
    pose_task->setOriControlGains(400, 40, 0);

    // Velocity saturation: linear low, angular high.
    // NOTE: Python also rate-limits the position goal (MAX_GOAL_SPEED), so this
    // mainly acts as a backstop. Tune if the arm can't keep up.
    pose_task->enableVelocitySaturation(0.10, 6.0);

    // --------------------
    // Joint task (weak nullspace posture; strong would fight the pose task)
    auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
    joint_task->setGains(20, 10, 0);
    joint_task->setGoalPosition(robot->q());

    // Cache initial EE pose as "home"
    Vector3d ee_pos_home = robot->position(control_link, control_point);
    Matrix3d ee_ori_home = robot->rotation(control_link);
    pose_task->setGoalPosition(ee_pos_home);
    pose_task->setGoalOrientation(ee_ori_home);

    cout << "EE home position: " << ee_pos_home.transpose() << endl;

    // === Seed redis keys before the loop ===
    // Goal keys: seed with home so the controller holds home until Python
    // starts publishing (the two processes have no guaranteed start order).
    // State key: seed so Python's wait_for_key() can detect us immediately.
    redis_client.setEigen(EE_DESIRED_POS_KEY, ee_pos_home);
    redis_client.setEigen(EE_DESIRED_ORI_KEY, ee_ori_home);
    redis_client.setEigen(EE_POS_KEY,         ee_pos_home);

    // --------------------
    // Control loop
    runloop = true;
    const double control_freq = 1000.0;
    SaiCommon::LoopTimer timer(control_freq, 1e6);

    long long counter = 0;

    while (runloop)
    {
        timer.waitForNextLoop();
        const double time = timer.elapsedSimTime();

        // 1. Read robot state
        robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
        robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
        robot->updateModel();

        // 2. Publish current EE position (Python needs it for waypoint-arrival
        //    checks and look-at computation)
        Vector3d p_ee = robot->position(control_link, control_point);
        redis_client.setEigen(EE_POS_KEY, p_ee);

        // 3. Read EE goals from redis (published by the Python state machine)
        Vector3d desired_pos = redis_client.getEigen(EE_DESIRED_POS_KEY);
        Matrix3d desired_ori = redis_client.getEigen(EE_DESIRED_ORI_KEY);

        // 4. Apply goals — controller is state-agnostic (SEARCHING vs TRACKING
        //    is decided entirely in Python; this loop never branches on it)
        pose_task->setGoalPosition(desired_pos);
        pose_task->setGoalOrientation(desired_ori);

        // 5. Task hierarchy: pose task > joint task (in nullspace)
        N_prec.setIdentity();
        pose_task->updateTaskModel(N_prec);
        joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

        // 6. Compute torques
        command_torques = pose_task->computeTorques() + joint_task->computeTorques();

        // 7. Torque clamp for safety (salvaged from friend's controller —
        //    pure execution-layer safety, legitimately the controller's job).
        //    If too restrictive, raise torque_limit.
        const double torque_limit = 100.0;
        for (int i = 0; i < dof; i++) {
            command_torques(i) = std::max(
                -torque_limit,
                std::min(torque_limit, command_torques(i)));
        }

        redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, command_torques);

        // 8. Debug print every 0.5 s
        if (counter % 500 == 0) {
            double pos_err = (desired_pos - p_ee).norm();
            cout << "[ctrl] t=" << time
                 << " | desired_pos=" << desired_pos.transpose()
                 << " | ee_pos=" << p_ee.transpose()
                 << " | pos_err=" << pos_err << endl;
        }

        counter++;
    }

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    // Reset torques on shutdown — robot floats
    redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, VectorXd::Zero(dof));

    return 0;
}




// /**
//  * @file controller.cpp
//  * @brief Gaze controller: camera looks at the red block while EE stays at home.
//  *
//  * Strategy:
//  *   - Position goal: hold "home" position, moderate linear velocity saturation
//  *   - Orientation goal: look-at the red block (mouse), high angular saturation
//  *   - Camera optical axis assumed to be +link7_Z (verified via ee_camera view)
//  */

// #include <SaiModel.h>
// #include "SaiPrimitives.h"
// #include "redis/RedisClient.h"
// #include "timer/LoopTimer.h"

// #include <iostream>
// #include <string>

// using namespace std;
// using namespace Eigen;
// using namespace SaiPrimitives;

// #include <signal.h>
// bool runloop = false;
// void sighandler(int) { runloop = false; }

// #include "redis_keys.h"

// // Look-at rotation: build R_world_link7 so link7's +Z axis points
// // from p_ee toward p_target. (link7 +Z = camera optical axis, verified empirically)
// Matrix3d computeLookAtRotation(const Vector3d& p_ee,
//                                const Vector3d& p_target,
//                                const Vector3d& world_up = Vector3d::UnitZ())
// {
//     // Desired camera optical axis direction in world frame.
//     // Empirically determined: camera optical axis = +link7_Z
//     Vector3d camera_axis = p_target - p_ee;
//     double d = camera_axis.norm();
//     if (d < 1e-6) {
//         return Matrix3d::Identity();
//     }
//     camera_axis /= d;

//     // We want: link7's +Z axis (in world) = camera_axis
//     Vector3d link7_z = camera_axis;

//     // Build orthonormal frame: link7_z, then orthogonalize world_up to find
//     // link7_y (or link7_x), then complete with cross product.
//     //
//     // Strategy: pick link7_x = (world_up × link7_z) normalized (perpendicular to up and z)
//     //           then link7_y = link7_z × link7_x (completes right-handed frame)
//     Vector3d link7_x = world_up.cross(link7_z);
//     double xn = link7_x.norm();
//     if (xn < 1e-3) {
//         // link7_z parallel to world_up — pick alternate
//         Vector3d alt_up = Vector3d::UnitX();
//         link7_x = alt_up.cross(link7_z);
//         xn = link7_x.norm();
//     }
//     link7_x /= xn;
//     Vector3d link7_y = link7_z.cross(link7_x);   // unit by construction (orthogonal of unit vectors)

//     Matrix3d R;
//     R.col(0) = link7_x;
//     R.col(1) = link7_y;
//     R.col(2) = link7_z;   // camera optical axis points at target
// 	return R;

//     // Re-orthonormalize via QR for guaranteed validity (avoids OTG complaint)
//     Eigen::HouseholderQR<Matrix3d> qr(R);
//     Matrix3d Q = qr.householderQ();
//     if (Q.determinant() < 0) {
//         Q.col(2) *= -1;
//     }
//     return Q;
// }

// int main()
// {
//     static const string robot_file = string(CS225A_URDF_FOLDER) + "/Rizon4s.urdf";

//     auto redis_client = SaiCommon::RedisClient();
//     redis_client.connect();

//     signal(SIGABRT, &sighandler);
//     signal(SIGTERM, &sighandler);
//     signal(SIGINT, &sighandler);

//     // Load robot, sync initial state
//     auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
//     robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
//     robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
//     robot->updateModel();

//     int dof = robot->dof();
//     VectorXd command_torques = VectorXd::Zero(dof);
//     MatrixXd N_prec = MatrixXd::Identity(dof, dof);

//     // Pose task on link7
//     const string control_link = "link7";
//     const Vector3d control_point = Vector3d(0.074, -0.01, 0.136); 
//     Affine3d compliant_frame = Affine3d::Identity();
//     compliant_frame.translation() = control_point;
//     auto pose_task = std::make_shared<SaiPrimitives::MotionForceTask>(
//         robot, control_link, compliant_frame);
//     pose_task->setPosControlGains(400, 40, 0);
//     pose_task->setOriControlGains(400, 40, 0);

// // === Velocity saturation: moderate linear + high angular ===
// //   linear_vel_sat  = 0.10 m/s = 10 cm/s  (well below 15 cm/s hardware limit)
// //   angular_vel_sat = 6.0 rad/s ≈ 340°/s  (fast rotation to track mouse)
// pose_task->enableVelocitySaturation(0.10, 6.0);

//     // Joint task (nullspace posture)
//     auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);
//     joint_task->setGains(400, 40, 0);

//     VectorXd q_desired(dof);
//     q_desired << robot->q();
//     joint_task->setGoalPosition(q_desired);

//     // Cache initial EE pose as "home"
//     Vector3d ee_pos_home = robot->position(control_link, control_point);
//     Matrix3d R_last_good = robot->rotation(control_link);
//     pose_task->setGoalPosition(ee_pos_home);
//     pose_task->setGoalOrientation(R_last_good);

//     cout << "EE home position: " << ee_pos_home.transpose() << endl;
	
// 	// === DIAGNOSTIC: print link7 axes in world frame at home pose ===
// 	{
// 		Matrix3d R_home = robot->rotation(control_link);
// 		cout << "=== link7 axes at home pose (world frame) ===" << endl;
// 		cout << "  link7 +X: " << R_home.col(0).transpose() << endl;
// 		cout << "  link7 +Y: " << R_home.col(1).transpose() << endl;
// 		cout << "  link7 +Z: " << R_home.col(2).transpose() << endl;
// 		cout << "  (For reference: world +Z is UP, world -Z points DOWN)" << endl;
// 		cout << "=================================================" << endl;
// 	}

//     runloop = true;
//     double control_freq = 1000;
//     SaiCommon::LoopTimer timer(control_freq, 1e6);

//     int debug_counter = 0;

//     while (runloop)
//     {
//         timer.waitForNextLoop();

//         // 1. Read robot state
//         robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
//         robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
//         robot->updateModel();

//         // 2. Read red block (mouse) position published by simviz
//         Vector3d mouse_pos_world = redis_client.getEigen("cs225a::project::mouse::pos_world");
//         std::string detected_str = redis_client.get("cs225a::project::mouse::detected");
//         // bool mouse_detected = (detected_str == "1");
// 		bool mouse_detected = true;

//         // 3. Position goal: hold home (low linear sat keeps it nearly still)
//         pose_task->setGoalPosition(ee_pos_home);

//         // 4. Orientation goal: look at red block
//         Vector3d p_ee = robot->position(control_link, control_point);
//         if (mouse_detected) {
//             Matrix3d R_desired = computeLookAtRotation(p_ee, mouse_pos_world);
//             pose_task->setGoalOrientation(R_desired);
//             R_last_good = R_desired;
//         } else {
//             pose_task->setGoalOrientation(R_last_good);
//         }

//         // 5. Task hierarchy: pose task > joint task (in nullspace)
//         N_prec.setIdentity();
//         pose_task->updateTaskModel(N_prec);
//         joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

//         // 6. Compute and send torques
//         command_torques = pose_task->computeTorques() + joint_task->computeTorques();
//         redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, command_torques);

//         // 7. Debug print every 0.5 s
// 		if (++debug_counter % 500 == 0 && mouse_detected) {
// 			Matrix3d R_current = robot->rotation(control_link);
// 			Vector3d camera_axis_world = R_current.col(2);    // camera axis = +link7_Z
// 			Vector3d look_dir = (mouse_pos_world - p_ee).normalized();
// 			double alignment = camera_axis_world.dot(look_dir);   // +1 = camera AT target ✅
			
// 			cout << "[gaze] mouse=" << mouse_pos_world.transpose()
// 				<< "  align=" << alignment << endl;
// 		}
// 	}

//     timer.stop();
//     cout << "\nControl loop timer stats:\n";
//     timer.printInfoPostRun();

//     // Reset torques on shutdown — robot floats
//     redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, VectorXd::Zero(dof));

//     return 0;
// }