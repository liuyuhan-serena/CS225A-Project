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
const string robot_file = "${CS225A_URDF_FOLDER}/panda/panda_arm.urdf";

int main(int argc, char** argv) {
    SaiModel::URDF_FOLDERS["CS225A_URDF_FOLDER"] = string(CS225A_URDF_FOLDER);

    // check for command line arguments
    if (argc != 2) {
        cout << "Incorrect number of command line arguments" << endl;
        cout << "Expected usage: ./{HW_NUMBER} {QUESTION_NUMBER}" << endl;
        return 1;
    }
    // convert char to int, check for correct controller number input
    string arg = argv[1];
    int controller_number;
    try {
        size_t pos;
        controller_number = stoi(arg, &pos);
        if (pos < arg.size()) {
            cerr << "Trailing characters after number: " << arg << '\n';
            return 1;
        }
        else if (controller_number < 1 || controller_number > 4) {
            cout << "Incorrect controller number" << endl;
            return 1;
        }
    } catch (invalid_argument const &ex) {
        cerr << "Invalid number: " << arg << '\n';
        return 1;
    } catch (out_of_range const &ex) {
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
	const string link_name = "link7";
	const Vector3d pos_in_link = Vector3d(0, 0, 0.10);
	VectorXd control_torques = VectorXd::Zero(dof);

	// model quantities for operational space control
	MatrixXd Jv = MatrixXd::Zero(3,dof);
	MatrixXd Lambda = MatrixXd::Zero(3,3);
	MatrixXd J_bar = MatrixXd::Zero(dof,3);
	MatrixXd N = MatrixXd::Zero(dof,dof);

	Jv = robot->Jv(link_name, pos_in_link);
	Lambda = robot->taskInertiaMatrix(Jv);
	J_bar = robot->dynConsistentInverseJacobian(Jv);
	N = robot->nullspaceMatrix(Jv);

    // flag for enabling gravity compensation
    bool gravity_comp_enabled = false;

    // start redis client
    auto redis_client = SaiCommon::RedisClient();
    redis_client.connect();

    // setup send and receive groups
    VectorXd robot_q = redis_client.getEigen(JOINT_ANGLES_KEY);
    VectorXd robot_dq = redis_client.getEigen(JOINT_VELOCITIES_KEY);
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
    Vector3d x = robot->position(link_name, pos_in_link);
    Vector3d x_des;
    Vector3d dx = robot->linearVelocity(link_name, pos_in_link);
    x_des << 0.3, 0.1, 0.5;
    
    string file_path = std::string(HW_FOLDER) + "/hw2/logs/hw2_q" + std::to_string(controller_number);
    SaiCommon::Logger logger(file_path, false);

    if (controller_number == 1) {
        logger.addToLog(robot_q, "q");
        logger.addToLog(q_desired, "q_des");
    }
    else if (controller_number == 2 || controller_number == 3 || controller_number == 4) {
        logger.addToLog(x, "x");
        logger.addToLog(x_des, "x_des");
        logger.addToLog(robot_q, "q");   
    }

    logger.start(100);
    // create a loop timer
    const double control_freq = 1000;
    SaiCommon::LoopTimer timer(control_freq);
    
    while (runloop) {
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
        if(controller_number == 1) {
            MatrixXd Kp = MatrixXd::Zero(dof, dof);
              MatrixXd Kv = MatrixXd::Zero(dof, dof);

              for (int i = 0; i < 6; i++) {
                Kp(i,i) = 400.0;
                Kv(i,i) = 50.0;
            }

            Kp(6,6) = 50.0;
            Kv(6,6) = -0.28;   

            q_desired = initial_q;
            q_desired(6) = 0.1;   // q7d = 0.1 rad

            VectorXd g = robot->jointGravityVector();
            VectorXd b = robot->coriolisForce();   

            control_torques = -Kp * (robot_q - q_desired) - Kv * robot_dq + b + g;
        
        }

        // ---------------------------  question 2 ---------------------------------------
        else if(controller_number == 2) {
            double kp = 200.0;
            double kv = 30.0;   
            double kvj = 12.0;

            x = robot->position(link_name, pos_in_link);
            dx = robot->linearVelocity(link_name, pos_in_link);

            Jv = robot->Jv(link_name, pos_in_link);
            Lambda = robot->taskInertiaMatrix(Jv);

            // task-space control force
            Vector3d F = Lambda * (-kp * (x - x_des) - kv * dx);

            // gravity compensation in joint space
            VectorXd g = robot->jointGravityVector();
            N = robot->nullspaceMatrix(Jv);
            MatrixXd M = robot->M();
            // map task-space force to joint torques
            // 2a. control_torques = Jv.transpose() * F + g
            // 2c. control_torques = Jv.transpose() * F + g - kvj * robot_dq;
            control_torques = Jv.transpose() * F + g + (N.transpose() * M * (-kvj * robot_dq));
        }

        // ---------------------------  question 3 ---------------------------------------
        else if(controller_number == 3) {
            double kp = 200.0;
            double kv = 30.0;
            double kvj = 12.0;

            x_des << 0.3, 0.1, 0.5;
            // current task-space position and velocity
            x = robot->position(link_name, pos_in_link);
            dx = robot->linearVelocity(link_name, pos_in_link);

            Jv = robot->Jv(link_name, pos_in_link);

            Lambda = robot->taskInertiaMatrix(Jv);

            // dynamically consistent inverse Jacobian
            J_bar = robot->dynConsistentInverseJacobian(Jv);

            N = robot->nullspaceMatrix(Jv);

            MatrixXd M = robot->M();
            VectorXd g = robot->jointGravityVector();

            // operational space gravity vector
            Vector3d p = J_bar.transpose() * g;
            Vector3d F = Lambda * (kp * (x_des - x) - kv * dx) + p;

            MatrixXd Kvj = kvj * MatrixXd::Identity(dof, dof);

            control_torques =
            Jv.transpose() * F - N.transpose() * M * Kvj * robot_dq;
        }

        // ---------------------------  question 4 ---------------------------------------
        else if(controller_number == 4) {

            double kp = 200.0;
            double kpj = 50.0;
            double kv = 30.0;
            double kvj = 12.0;

            x_des << 0.3 + 0.1 * sin(M_PI * time),
             0.1 + 0.1 * cos(M_PI * time),
             0.5;

            // current task-space position and velocity
            x = robot->position(link_name, pos_in_link);
            dx = robot->linearVelocity(link_name, pos_in_link);

            Jv = robot->Jv(link_name, pos_in_link);

            Lambda = robot->taskInertiaMatrix(Jv);
            
            // dynamically consistent inverse Jacobian
            J_bar = robot->dynConsistentInverseJacobian(Jv);

            N = robot->nullspaceMatrix(Jv);

            MatrixXd M = robot->M();
            VectorXd g = robot->jointGravityVector();

            // operational space gravity vector
            Vector3d p = J_bar.transpose() * g;
            // 4.1 Vector3d F = Lambda * (kp * (x_des - x) - kv * dx) + p;
            // 4.2Vector3d F = (kp * (x_des - x) - kv * dx) + p;
            // 4.3Vector3d F = Lambda * (kp * (x_des - x) - kv * dx) + p;
            Vector3d F = Lambda * (kp * (x_des - x) - kv * dx)；
     
            VectorXd q_posture_des = VectorXd::Zero(dof);

            MatrixXd Kpj = kpj * MatrixXd::Identity(dof, dof);
            MatrixXd Kvj = kvj * MatrixXd::Identity(dof, dof);

            VectorXd posture_command =
            -Kpj * (robot_q - q_posture_des) - Kvj * robot_dq;

            //4.1, 4.2 
            // control_torques = Jv.transpose() * F - N.transpose() * M * Kvj * robot_dq;
            //4.3 control_torques = Jv.transpose() * F + N.transpose() * M * posture_command;
            control_torques = Jv.transpose() * F + N.transpose() * M * posture_command + g;
        }

        // **********************
        // WRITE YOUR CODE BEFORE
        // **********************

        // send to redis
        redis_client.sendAllFromGroup();
    }

    control_torques.setZero();
    gravity_comp_enabled = true;
    redis_client.sendAllFromGroup();

    logger.stop();

    timer.stop();
    cout << "\nControl loop timer stats:\n";
    timer.printInfoPostRun();

    return 0;
}
