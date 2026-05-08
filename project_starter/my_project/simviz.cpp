/**
 * @file simviz.cpp
 * @brief Simulation and visualization of panda robot with 1 DOF gripper
 *
 */

#include <math.h>
#include <signal.h>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <fstream>
#include <filesystem>
#include <iostream>
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
static const string robot_name = "flexiv";
static const string camera_name = "camera_fixed";

// dynamic objects information
const vector<std::string> object_names = {"mouse"};
vector<Affine3d> object_poses;
vector<VectorXd> object_velocities;
const int n_objects = object_names.size();

// simulation thread
void simulation(std::shared_ptr<SaiSimulation::SaiSimulation> sim);

int main()
{
	SaiModel::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);
	static const string robot_file = string(CS225A_URDF_FOLDER) + "/flexiv/flexiv.urdf";
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

	// === Attach EE camera to link7, but specify pose relative to flange ===
	{
			// Step 1: Measure these from the flange frame
			Eigen::Affine3d T_flange_to_camera = Eigen::Affine3d::Identity();
			T_flange_to_camera.translation() << -0.10, 0.00, -0.10;   // Measured from flange
			T_flange_to_camera.linear() = 
					Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();

			// Step 2: From URDF: flange is at (0,0,0.081) in link7 frame, rotated 180° about Z
			Eigen::Affine3d T_link7_to_flange = Eigen::Affine3d::Identity();
			T_link7_to_flange.translation() << 0.0, 0.0, 0.081;
			T_link7_to_flange.linear() = 
					Eigen::AngleAxisd(-M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();

			// Step 3: Compose
			Eigen::Affine3d T_link7_to_camera = T_link7_to_flange * T_flange_to_camera;

			graphics->attachCameraToRobotLink(
					"ee_camera",
					robot_name,
					"link7",
					T_link7_to_camera);
	}
	
	// load robots
	auto robot = std::make_shared<SaiModel::SaiModel>(robot_file, false);
	// robot->setQ();
	// robot->setDq();
	robot->updateModel();
	ui_torques = VectorXd::Zero(robot->dof());

	graphics->addUIForceInteraction(robot_name);

	// load simulation world
	auto sim = std::make_shared<SaiSimulation::SaiSimulation>(world_file, false);
	sim->setJointPositions(robot_name, robot->q());
	sim->setJointVelocities(robot_name, robot->dq());

	// fill in object information
	for (int i = 0; i < n_objects; ++i)
	{
		object_poses.push_back(sim->getObjectPose(object_names[i]));
		object_velocities.push_back(sim->getObjectVelocity(object_names[i]));
	}

	// set simulation parameters
	sim->setCollisionRestitution(0.5);
	sim->setCoeffFrictionStatic(0.0);
	sim->setCoeffFrictionDynamic(0.0);

	/*------- Set up visualization -------*/
	// init redis client values
	redis_client.setEigen(JOINT_ANGLES_KEY, robot->q());
	redis_client.setEigen(JOINT_VELOCITIES_KEY, robot->dq());
	redis_client.setEigen(JOINT_TORQUES_COMMANDED_KEY, 0 * robot->q());

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
	// fSimulationRunning = true;

	// create redis client
	auto redis_client = SaiCommon::RedisClient();
	redis_client.connect();

	// create a timer
	double sim_freq = 2000;
	SaiCommon::LoopTimer timer(sim_freq);

	sim->setTimestep(1.0 / sim_freq);
	sim->enableGravityCompensation(true);
	sim->enableJointLimits(robot_name);

	while (fSimulationRunning)
	{
		timer.waitForNextLoop();

		VectorXd control_torques = redis_client.getEigen(JOINT_TORQUES_COMMANDED_KEY);
		{
			lock_guard<mutex> lock(mutex_torques);
			sim->setJointTorques(robot_name, control_torques + ui_torques);
		}

		// === Mouse trajectory: drive the box along a circle ===
                {
                        lock_guard<mutex> lock(mutex_update);
                        double t = sim->time();
                        double radius = 0.15;     // 15 cm
                        double omega  = 2.67;      // rad/s, ~40cm/s
                        double cx = 0.6, cy = 0.0, cz = 0.05;  // circle center

                        Eigen::Affine3d mouse_pose = Eigen::Affine3d::Identity();
                        mouse_pose.translation() << cx + radius * std::cos(omega * t),
                                                    cy + radius * std::sin(omega * t),
                                                    cz;
                        sim->setObjectPose("mouse", mouse_pose);
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