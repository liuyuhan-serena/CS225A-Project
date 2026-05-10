/**
 * @file simviz.cpp
 * @brief Simulation and visualization of Rizon4s robot chasing a mouse
 *
 * This version publishes:
 *   - T_world_cam (4x4 matrix) for CV team to convert detections to world frame
 *   - Fake CV (ground-truth mouse pos in world frame) for testing without real CV
 */

#include <math.h>
#include <signal.h>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <fstream>
#include <filesystem>
#include <vector>
#include <typeinfo>
#include <random>

#include "SaiGraphics.h"
#include "SaiModel.h"
#include "SaiSimulation.h"
#include "SaiPrimitives.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "logger/Logger.h"

bool fSimulationRunning = true;
void sighandler(int) { fSimulationRunning = false; }

#include "redis_keys.h"

using namespace Eigen;
using namespace std;

// mutex and globals
VectorXd ui_torques;
mutex mutex_torques, mutex_update;

// specify urdf and robots
static const string robot_name = "Rizon4s";
static const string camera_name = "camera_fixed";

// dynamic objects information
const vector<std::string> object_names = {"mouse"};
vector<Affine3d> object_poses;
vector<VectorXd> object_velocities;
const int n_objects = object_names.size();

// shared robot model (used by simulation thread for FK / camera transform)
std::shared_ptr<SaiModel::SaiModel> g_robot;

// camera extrinsic constant (link7 -> camera) — keep in sync with graphics setup in main()
Affine3d getCameraExtrinsics()
{
    Affine3d T = Affine3d::Identity();
    T.translation() << 0.074, -0.01, 0.136;
    T.linear() = AngleAxisd(M_PI / 2, Vector3d::UnitY()).toRotationMatrix();
    return T;
}

// simulation thread
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim);

int main()
{
    SaiModel::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);
    static const string robot_file = string(CS225A_URDF_FOLDER) + "/Rizon4s.urdf";
    static const string world_file = string(MY_PROJECT_FOLDER) + "/world_my_project.urdf";
    std::cout << "Loading URDF world model file: " << world_file << endl;

    // start redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // load graphics scene
    auto graphics = std::make_shared<SaiGraphics::SaiGraphics>(world_file, camera_name, false);

    // === Attach EE camera to link7 (William's d405 left-eye position) ===
    {
        Affine3d T_link7_to_camera = getCameraExtrinsics();
        graphics->attachCameraToRobotLink(
            "ee_camera",
            robot_name,
            "link7",
            T_link7_to_camera);
    }

    // load robot model (shared with simulation thread for FK)
    g_robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
    g_robot->updateModel();
    ui_torques = VectorXd::Zero(g_robot->dof());

    graphics->addUIForceInteraction(robot_name);

    // load simulation world
    auto sim = std::make_shared<SaiSimulation::SaiSimulation>(world_file, false);
    // === Set a sane initial pose (EE-down hover) ===
    VectorXd q_init(g_robot->dof());
<<<<<<< HEAD
    q_init = sim->getJointPositions(robot_name);
    g_robot->setQ(q_init);
    g_robot->updateModel();
=======
    q_init << 0.0, -30.0, 0.0, 90.0, 0.0, 60.0, 0.0; // same as controller's nullspace pref
    q_init *= M_PI / 180.0;
    // g_robot->setQ(q_init);
    // g_robot->updateModel();

    // sim->setJointPositions(robot_name, q_init);
    // sim->setJointVelocities(robot_name, VectorXd::Zero(g_robot->dof()));
>>>>>>> 5ad808aeb2f848c0c0c81a9c168e186d5e38029d

    // fill in object information
    for (int i = 0; i < n_objects; ++i)
    {
        object_poses.push_back(sim->getObjectPose(object_names[i]));
        object_velocities.push_back(sim->getObjectVelocity(object_names[i]));
    }

    // set simulation parameters
    sim->setCollisionRestitution(0.0);
    sim->setCoeffFrictionStatic(0.5);
    sim->setCoeffFrictionDynamic(0.5);

    /*------- Set up visualization -------*/
    redis_client.setEigen(JOINT_ANGLES_KEY, g_robot->q());
    redis_client.setEigen(JOINT_VELOCITIES_KEY, g_robot->dq());
    redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * g_robot->q());

    // start simulation thread
    thread sim_thread(simulation, sim);

    // while window is open:
    while (graphics->isWindowOpen() && fSimulationRunning)
    {
        graphics->updateRobotGraphics(robot_name, redis_client.getEigen(JOINT_ANGLES_KEY));
        {
            lock_guard<mutex> lock(mutex_update);
            for (int i = 0; i < n_objects; ++i)
            {
                graphics->updateObjectGraphics(object_names[i], object_poses[i]);
            }
        }

        graphics->renderGraphicsWorld();
        {
            lock_guard<mutex> lock(mutex_torques);
            ui_torques = graphics->getUITorques(robot_name);
        }
    }

    // stop simulation
    fSimulationRunning = false;
    sim_thread.join();

    return 0;
}

//------------------------------------------------------------------------------
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim)
{
    // create redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // create a timer
    double sim_freq = 2000;
    SaiCommon::LoopTimer timer(sim_freq);

    sim->setTimestep(1.0 / sim_freq);
    sim->enableGravityCompensation(true);
    sim->enableJointLimits(robot_name);

    // camera extrinsic (constant)
    const Affine3d T_link7_cam = getCameraExtrinsics();

    // RNG for fake CV noise
    std::default_random_engine rng(0);
    std::normal_distribution<double> noise(0.0, 0.005); // 5mm std-dev

    // toggle this to false once teammate's real CV is online
    constexpr bool FAKE_CV_ENABLED = true;

    while (fSimulationRunning)
    {
        timer.waitForNextLoop();

        VectorXd control_torques = redis_client.getEigen(JOINT_TORQUES_COMMANDED_KEY);
        {
            lock_guard<mutex> lock(mutex_torques);
            sim->setJointTorques(robot_name, control_torques + ui_torques);
        }

        // === Mouse trajectory: circle, start at 12 o'clock (+y) ===
        {
            lock_guard<mutex> lock(mutex_update);
            double t = sim->time();
            double radius = 0.15; // 15 cm
            double omega = 2.67;  // rad/s, ~40cm/s (lower for early debug, e.g. 0.5)
            double cx = 0.6, cy = 0.0, cz = 0.05;

            Affine3d mouse_pose = Affine3d::Identity();
            mouse_pose.translation() << cx + radius * std::cos(omega * t + M_PI_2),
                cy + radius * std::sin(omega * t + M_PI_2),
                cz;
            sim->setObjectPose("mouse", mouse_pose);
        }

        // === Publish T_world_cam (for CV team) + fake CV (for testing) ===
        {
            // current robot state -> FK
            VectorXd q_now = sim->getJointPositions(robot_name);
            g_robot->setQ(q_now);
            g_robot->updateModel();
            Affine3d T_world_link7 = g_robot->transform("link7");
            Affine3d T_world_cam = T_world_link7 * T_link7_cam;

            // Publish 4x4 transform for CV team
            Matrix4d T_world_cam_mat = T_world_cam.matrix();
            redis_client.setEigen("cs225a::project::camera::T_world_cam", T_world_cam_mat);

            if (FAKE_CV_ENABLED)
            {
                // ground-truth mouse in world
                Affine3d T_world_mouse = sim->getObjectPose("mouse");
                Vector3d p_world_mouse = T_world_mouse.translation();

                // FOV check still needs camera-frame position
                Vector3d p_cam_mouse = T_world_cam.inverse() * p_world_mouse;

                // publish ground truth (no round-trip — avoid coupling fake CV to EE motion)
                Vector3d p_world_published = p_world_mouse;

                // add noise (5mm std-dev)
                p_world_published.x() += noise(rng);
                p_world_published.y() += noise(rng);
                p_world_published.z() += noise(rng);

                // FOV check: only "detect" if mouse is in front of camera
                bool in_view = (p_cam_mouse.z() > 0.05) && (p_cam_mouse.z() < 1.5);

                redis_client.setEigen("cs225a::project::mouse::pos_world", p_world_published);
                redis_client.set("cs225a::project::mouse::detected", in_view ? "1" : "0");
                redis_client.set("cs225a::project::mouse::timestamp", std::to_string(sim->time()));
            }
        }

        sim->integrate();
        redis_client.setEigen(JOINT_ANGLES_KEY, sim->getJointPositions(robot_name));
        redis_client.setEigen(JOINT_VELOCITIES_KEY, sim->getJointVelocities(robot_name));

        // update object information
        {
            lock_guard<mutex> lock(mutex_update);
            for (int i = 0; i < n_objects; ++i)
            {
                object_poses[i] = sim->getObjectPose(object_names[i]);
                object_velocities[i] = sim->getObjectVelocity(object_names[i]);
            }
        }
    }
    timer.stop();
    cout << "\nSimulation loop timer stats:\n";
    timer.printInfoPostRun();
}

// /**
//  * @file simviz.cpp
//  * @brief Simulation and visualization of panda robot with 1 DOF gripper
//  *
//  */

// #include <math.h>
// #include <signal.h>
// #include <iostream>
// #include <mutex>
// #include <string>
// #include <thread>
// #include <fstream>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <typeinfo>
// #include <random>

// #include "SaiGraphics.h"
// #include "SaiModel.h"
// #include "SaiSimulation.h"
// #include "SaiPrimitives.h"
// #include "redis/RedisClient.h"
// #include "timer/LoopTimer.h"
// #include "logger/Logger.h"

// bool fSimulationRunning = true;
// void sighandler(int) { fSimulationRunning = false; }

// #include "redis_keys.h"

// using namespace Eigen;
// using namespace std;

// // mutex and globals
// VectorXd ui_torques;
// mutex mutex_torques, mutex_update;

// // specify urdf and robots
// static const string robot_name = "Rizon4s";
// static const string camera_name = "camera_fixed";

// // dynamic objects information
// const vector<std::string> object_names = {"mouse"};
// vector<Affine3d> object_poses;
// vector<VectorXd> object_velocities;
// const int n_objects = object_names.size();

// // simulation thread
// void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim);

// int main()
// {
// 	SaiModel::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);
// 	static const string robot_file = string(CS225A_URDF_FOLDER) + "/Rizon4s.urdf";
// 	static const string world_file = string(MY_PROJECT_FOLDER) + "/world_my_project.urdf";
// 	std::cout << "Loading URDF world model file: " << world_file << endl;

// 	// start redis client
// 	auto redis_client = SaiCommon::RedisClient();
// 	redis_client.connect();

// 	// set up signal handler
// 	signal(SIGABRT, &sighandler);
// 	signal(SIGTERM, &sighandler);
// 	signal(SIGINT, &sighandler);

// 	// load graphics scene
// 	auto graphics = std::make_shared<SaiGraphics::SaiGraphics>(world_file, camera_name, false);

// // === Attach EE camera to link7 (using William's d405 left-eye position) ===
// 	{
// 			Eigen::Affine3d T_link7_to_camera = Eigen::Affine3d::Identity();

// 			// Translation: From Rizon4s.urdf left-eye ball positon (link7 frame)
// 			T_link7_to_camera.translation() << 0.074, -0.01, 0.136;

// 			// Rotation: needed tuning
// 			T_link7_to_camera.linear() =
// 					Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitY()).toRotationMatrix();

// 			graphics->attachCameraToRobotLink(
// 					"ee_camera",
// 					robot_name,
// 					"link7",
// 					T_link7_to_camera);
// 	}

// 	// load robots
// 	auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
// 	// robot->setQ();
// 	// robot->setDq();
// 	robot->updateModel();
// 	ui_torques = VectorXd::Zero(robot->dof());

// 	graphics->addUIForceInteraction(robot_name);

// 	// load simulation world
// 	auto sim = std::make_shared<SaiSimulation::SaiSimulation>(world_file, false);
// 	sim->setJointPositions(robot_name, robot->q());
// 	sim->setJointVelocities(robot_name, robot->dq());

// 	// fill in object information
// 	for (int i = 0; i < n_objects; ++i)
// 	{
// 		object_poses.push_back(sim->getObjectPose(object_names[i]));
// 		object_velocities.push_back(sim->getObjectVelocity(object_names[i]));
// 	}

// 	// set simulation parameters
// 	sim->setCollisionRestitution(0.5);
// 	sim->setCoeffFrictionStatic(0.0);
// 	sim->setCoeffFrictionDynamic(0.0);

// 	/*------- Set up visualization -------*/
// 	// init redis client values
// 	redis_client.setEigen(JOINT_ANGLES_KEY, robot->q());
// 	redis_client.setEigen(JOINT_VELOCITIES_KEY, robot->dq());
// 	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * robot->q());

// 	// start simulation thread
// 	thread sim_thread(simulation, sim);

// 	// while window is open:
// 	while (graphics->isWindowOpen() && fSimulationRunning)
// 	{
// 		graphics->updateRobotGraphics(robot_name, redis_client.getEigen(JOINT_ANGLES_KEY));
// 		{
// 			lock_guard<mutex> lock(mutex_update);
// 			for (int i = 0; i < n_objects; ++i)
// 			{
// 				graphics->updateObjectGraphics(object_names[i], object_poses[i]);
// 			}
// 		}

// 		graphics->renderGraphicsWorld();
// 		{
// 			lock_guard<mutex> lock(mutex_torques);
// 			ui_torques = graphics->getUITorques(robot_name);
// 		}
// 	}

// 	// stop simulation
// 	fSimulationRunning = false;
// 	sim_thread.join();

// 	return 0;
// }

// //------------------------------------------------------------------------------
// void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim)
// {
// 	// fSimulationRunning = true;

// 	// create redis client
// 	auto redis_client = SaiCommon::RedisClient();
// 	redis_client.connect();

// 	// create a timer
// 	double sim_freq = 2000;
// 	SaiCommon::LoopTimer timer(sim_freq);

// 	sim->setTimestep(1.0 / sim_freq);
// 	sim->enableGravityCompensation(true);
// 	sim->enableJointLimits(robot_name);

// 	while (fSimulationRunning)
// 	{
// 		timer.waitForNextLoop();

// 		VectorXd control_torques = redis_client.getEigen(JOINT_TORQUES_COMMANDED_KEY);
// 		{
// 			lock_guard<mutex> lock(mutex_torques);
// 			sim->setJointTorques(robot_name, control_torques + ui_torques);
// 		}

// 		// === Mouse trajectory: drive the box along a circle ===
//                 {
//                         lock_guard<mutex> lock(mutex_update);
//                         double t = sim->time();
//                         double radius = 0.15;     // 15 cm
//                         double omega  = 2.67;      // rad/s, ~40cm/s
//                         double cx = 0.6, cy = 0.0, cz = 0.05;  // circle center

//                         Eigen::Affine3d mouse_pose = Eigen::Affine3d::Identity();
//                         mouse_pose.translation() << cx + radius * std::cos(omega * t + M_PI_2),
//                                                     cy + radius * std::sin(omega * t + M_PI_2),
//                                                     cz;
//                         sim->setObjectPose("mouse", mouse_pose);
//                 }

// 		sim->integrate();
// 		redis_client.setEigen(JOINT_ANGLES_KEY, sim->getJointPositions(robot_name));
// 		redis_client.setEigen(JOINT_VELOCITIES_KEY, sim->getJointVelocities(robot_name));

// 		// update object information
// 		{
// 			lock_guard<mutex> lock(mutex_update);
// 			for (int i = 0; i < n_objects; ++i)
// 			{
// 				object_poses[i] = sim->getObjectPose(object_names[i]);
// 				object_velocities[i] = sim->getObjectVelocity(object_names[i]);
// 			}
// 		}
// 	}
// 	timer.stop();
// 	cout << "\nSimulation loop timer stats:\n";
// 	timer.printInfoPostRun();
// }