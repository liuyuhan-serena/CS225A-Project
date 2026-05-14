/**
 * @file controller.cpp
 * @brief Controller file
 */

#include <SaiModel.h>
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace Eigen;
using namespace SaiPrimitives;

#include <signal.h>
bool runloop = false;
void sighandler(int) { runloop = false; }

#include "redis_keys.h"

int main()
{
    static const string robot_file = string(CS225A_URDF_FOLDER) + "/flexiv/flexiv.urdf";

    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
    robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
    robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
    robot->updateModel();

    const int dof = robot->dof();
    VectorXd command_torques = VectorXd::Zero(dof);
    MatrixXd N_prec = MatrixXd::Identity(dof, dof);

    // --------------------
    // Pose task
    const string control_link = "flange";
    const Vector3d control_point = Vector3d::Zero();
    Affine3d compliant_frame = Affine3d::Identity();

    auto pose_task = std::make_shared<SaiPrimitives::MotionForceTask>(
        robot,
        control_link,
        compliant_frame
    );

    pose_task->setPosControlGains(400, 40, 0);
    pose_task->setOriControlGains(100, 20, 0);

    pose_task->enableVelocitySaturation(0.35, 2.5);

    Vector3d ee_pos_initial = robot->position(control_link, control_point);
    Matrix3d ee_ori_initial = robot->rotation(control_link);

    cout << "Initial EE position: " << ee_pos_initial.transpose() << endl;

    // --------------------
    // Joint task
    auto joint_task = std::make_shared<SaiPrimitives::JointTask>(robot);

    // Very weak joint task. Strong joint task can fight the pose task.
    joint_task->setGains(20, 10, 0);
    joint_task->setGoalPosition(robot->q());

    // Smooth position tracking variables
    Vector3d ee_goal_filtered = ee_pos_initial;

    const double desired_distance = 0.45;  // meters

    const double max_goal_speed = 0.30;    // m/s

    Vector3d x_ref = ee_ori_initial.col(0);

    // --------------------
    // Control loop
	runloop = true;
    const double control_freq = 1000.0;
    const double dt_nominal = 1.0 / control_freq;

    SaiCommon::LoopTimer timer(control_freq, 1e6);

    long long counter = 0;

    while (runloop)
    {
        timer.waitForNextLoop();
        const double time = timer.elapsedSimTime();

        // Read robot state
        robot->setQ(redis_client.getEigen(JOINT_ANGLES_KEY));
        robot->setDq(redis_client.getEigen(JOINT_VELOCITIES_KEY));
        robot->updateModel();

        // Read ball position
        Vector3d ball_pos = redis_client.getEigen(BALL_POS_KEY);

        // Current EE pose
        Vector3d ee_pos = robot->position(control_link, control_point);
        Matrix3d ee_ori = robot->rotation(control_link);

        Vector3d object_to_ee = ee_pos - ball_pos;
        double dist = object_to_ee.norm();

        Vector3d object_to_camera_dir;
        if (dist > 1e-6) {
            object_to_camera_dir = object_to_ee.normalized();
        } else {
            object_to_camera_dir = Vector3d(0, 0, 1);
        }

        // Raw goal: camera stays desired_distance away from ball.
        //Vector3d ee_goal_raw = ball_pos + desired_distance * object_to_camera_dir;
		Vector3d ee_goal_raw = ball_pos + Vector3d(0,0,desired_distance);

        // Smooth / rate-limit the position goal.
        Vector3d delta_goal = ee_goal_raw - ee_goal_filtered;
        double max_step = max_goal_speed * dt_nominal;

        if (delta_goal.norm() > max_step) {
            delta_goal = max_step * delta_goal.normalized();
        }

        ee_goal_filtered += delta_goal;

        pose_task->setGoalPosition(ee_goal_filtered);

        // z_des points from end-effector to object.
        Vector3d z_des = ball_pos - ee_pos;

        if (z_des.norm() > 1e-6) {
            z_des.normalize();
        } else {
            z_des = ee_ori.col(2);
        }

        // Keep roll continuous:
        // Project previous x_ref onto the plane perpendicular to z_des.
        Vector3d x_des = x_ref - z_des * z_des.dot(x_ref);

        if (x_des.norm() < 1e-3) {
            // Fallback if x_ref is almost parallel to z_des.
            Vector3d fallback = Vector3d::UnitX();

            if (fabs(fallback.dot(z_des)) > 0.9) {
                fallback = Vector3d::UnitY();
            }

            x_des = fallback - z_des * z_des.dot(fallback);
        }

        x_des.normalize();

        // Complete right-handed frame.
        Vector3d y_des = z_des.cross(x_des);
        y_des.normalize();

        Matrix3d ee_goal_ori;
        ee_goal_ori.col(0) = x_des;
        ee_goal_ori.col(1) = y_des;
        ee_goal_ori.col(2) = z_des;

        pose_task->setGoalOrientation(ee_goal_ori);

        // Update x_ref for next loop to avoid roll flipping.
        x_ref = x_des;

        N_prec.setIdentity();
        pose_task->updateTaskModel(N_prec);
        joint_task->updateTaskModel(pose_task->getTaskAndPreviousNullspace());

        command_torques = pose_task->computeTorques() + joint_task->computeTorques();

        // Optional torque clamp for safety.
        // If this is too restrictive, increase torque_limit.
        const double torque_limit = 100.0;
        for (int i = 0; i < dof; i++) {
            command_torques(i) = std::max(
                -torque_limit,
                std::min(torque_limit, command_torques(i))
            );
        }

        redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, command_torques);

        // Print debug info once per second
        if (counter % 1000 == 0) {
            cout << "time: " << time
                 << " | dist: " << dist
                 << " | ee_pos: " << ee_pos.transpose()
                 << " | ball_pos: " << ball_pos.transpose()
                 << endl;
        }

        counter++;
    }

    timer.stop();

    cout << "\nSimulation loop timer stats:\n";
    timer.printInfoPostRun();

    redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, VectorXd::Zero(dof));

    return 0;
}