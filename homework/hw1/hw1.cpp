// some standard library includes
#include <math.h>

#include <iostream>
#include <mutex>
#include <string>
#include <thread>

// sai main libraries includes
#include "SaiModel.h"
#include "SaiCommon.h"

// sai utilities from sai-common
#include "timer/LoopTimer.h"
#include "redis/RedisClient.h"

// redis keys
#include "redis_keys.h"

// for handling ctrl+c and interruptions properly
#include <signal.h>
bool runloop = true;
void sighandler(int) { runloop = false; }

// namespaces for compactness of code
using namespace std;
using namespace Eigen;

// config file names and object names
const string robot_file = "${CS225A_URDF_FOLDER}/panda/panda_arm_controller.urdf";

int main(int argc, char **argv)
{
    SaiModel::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);
    SaiModel::URDF_FOLDERS["HW1_FOLDER"] = std::string(HW_FOLDER) + "/hw1";

    // check for command line arguments
    if (argc != 2)
    {
        cout << "Incorrect number of command line arguments" << endl;
        cout << "Expected usage: ./{HW_NUMBER} {QUESTION_NUMBER}" << endl;
        return 1;
    }
    // convert char to int, check for correct controller number input
    string arg = argv[1];
    int controller_number;
    try
    {
        size_t pos;
        controller_number = stoi(arg, &pos);
        if (pos < arg.size())
        {
            cerr << "Trailing characters after number: " << arg << '\n';
            return 1;
        }
        else if (controller_number < 1 || controller_number > 5)
        {
            cout << "Incorrect controller number" << endl;
            return 1;
        }
    }
    catch (invalid_argument const &ex)
    {
        cerr << "Invalid number: " << arg << '\n';
        return 1;
    }
    catch (out_of_range const &ex)
    {
        cerr << "Number out of range: " << arg << '\n';
        return 1;
    }

    // set up signal handler
    signal(SIGABRT, &sighandler);
    signal(SIGTERM, &sighandler);
    signal(SIGINT, &sighandler);

    // load robots
    auto robot = new SaiModel::SaiModel(robot_file);

    // prepare controller
    int dof = robot->dof();
    VectorXd control_torques = VectorXd::Zero(dof);

    // flag for enabling gravity compensation
    bool gravity_comp_enabled = false;

    // start redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // setup send and receive groups
    VectorXd robot_q = VectorXd::Zero(dof);
    VectorXd robot_dq = VectorXd::Zero(dof);
    redis_client.addToReceiveGroup(JOINT_ANGLES_KEY, robot_q);
    redis_client.addToReceiveGroup(JOINT_VELOCITIES_KEY, robot_dq);

    redis_client.addToSendGroup(JOINT_TORQUES_COMMANDED_KEY, control_torques);
    redis_client.addToSendGroup(GRAVITY_COMP_ENABLED_KEY, gravity_comp_enabled);

    redis_client.receiveAllFromGroup();
    redis_client.sendAllFromGroup();

    // update robot model from simulation configuration
    robot->setQ(robot_q);
    robot->setDq(robot_dq);
    robot->updateModel();

    // record initial configuration
    VectorXd initial_q = robot->q();
    VectorXd q_desired = VectorXd::Zero(dof);

    string file_path = std::string(HW_FOLDER) + "/hw1/logs/hw1_q" + std::to_string(controller_number);
    SaiCommon::Logger logger(file_path, false);
    logger.addToLog(robot_q, "q");
    logger.addToLog(q_desired, "q_des");

    logger.start(100); // log at 100 Hz

    // create a loop timer
    const double control_freq = 1000;
    SaiCommon::LoopTimer timer(control_freq);

    while (runloop)
    {
        // wait for next scheduled loop
        timer.waitForNextLoop();
        double time = timer.elapsedTime();

        // read robot state from redis
        redis_client.receiveAllFromGroup();
        robot->setQ(robot_q);
        robot->setDq(robot_dq);
        robot->updateModel();

        // **********************
        // WRITE YOUR CODE AFTER
        // **********************

        // ---------------------------  question 1 ---------------------------------------
        if (controller_number == 1)
        {

            double kp = 400.0; 
            double kv = 50.0; 

            q_desired = initial_q; 
            q_desired(0) = M_PI / 2.0;      // 90 deg
            q_desired(1) = -M_PI / 4.0;     // -45 deg
            q_desired(2) = 0.0;
            q_desired(3) = -125.0 * M_PI / 180.0;
            q_desired(4) = 0.0;
            q_desired(5) = 80.0 * M_PI / 180.0;
            q_desired(6) = 0.0;

            control_torques = -kp * (robot_q - q_desired) - kv * robot_dq;
        }

        // ---------------------------  question 2 ---------------------------------------
        else if (controller_number == 2)
        {
            double kp = 400.0;
            double kv = 50.0;  

            q_desired = VectorXd(7);
            q_desired << 90.0, -45.0, 0.0, -125.0, 0.0, 80.0, 0.0;
            q_desired *= M_PI / 180.0;

            VectorXd g = robot->jointGravityVector();
            control_torques = -kp * (robot_q - q_desired) - kv * robot_dq + g;
        }

        // ---------------------------  question 3 ---------------------------------------
        else if (controller_number == 3)
        {
            double kp = 400.0;
            double kv = 50.0;   

            q_desired = VectorXd(7);
            q_desired << 90.0, -45.0, 0.0, -125.0, 0.0, 80.0, 0.0;
            q_desired *= M_PI / 180.0;

            MatrixXd A = robot->M();
            VectorXd g = robot->jointGravityVector();

            VectorXd command = -kp * (robot_q - q_desired) - kv * robot_dq;
            control_torques = A * command + g;
        }

        // ---------------------------  question 4 ---------------------------------------
        else if (controller_number == 4)
        {
            double kp = 400.0;
            double kv = 40.0;   

            q_desired = VectorXd(7);
            q_desired << 90.0, -45.0, 0.0, -125.0, 0.0, 80.0, 0.0;
            q_desired *= M_PI / 180.0;

            MatrixXd A = robot->M();
            VectorXd g = robot->jointGravityVector();
            VectorXd b = robot->coriolisForce();   

            VectorXd command = -kp * (robot_q - q_desired) - kv * robot_dq;
            control_torques = A * command + b + g;
        }

        // ---------------------------  question 5 ---------------------------------------
        else if (controller_number == 5)
        {
            double kp = 400.0;
            double kv = 40.0;  

            q_desired = VectorXd(7);
            q_desired << 90.0, -45.0, 0.0, -125.0, 0.0, 80.0, 0.0;
            q_desired *= M_PI / 180.0;

            MatrixXd A = robot->M();
            VectorXd g = robot->jointGravityVector();
            VectorXd b = robot->coriolisForce();   

            VectorXd command = -kp * (robot_q - q_desired) - kv * robot_dq;
            control_torques = A * command + b + g;
        }

        // **********************
        // WRITE YOUR CODE BEFORE
        // **********************

        // send to redis
        redis_client.setInt("sai::simviz::gravity_comp_enabled", 0);
        redis_client.sendAllFromGroup();
    }

    control_torques.setZero();
    redis_client.sendAllFromGroup();

    logger.stop();

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    return 0;
}
