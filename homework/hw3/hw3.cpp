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
	const Vector3d pos_in_link = Vector3d(0, 0, 0.15);
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

    // create a loop timer
    const double control_freq = 1000;
    SaiCommon::LoopTimer timer(control_freq);

    // desired posture
    VectorXd q_desired = VectorXd::Zero(dof);

    // task variables for logging
    Vector3d x = Vector3d::Zero();
    Vector3d x_des = Vector3d::Zero();
    Vector3d dx = Vector3d::Zero();

    // controller variables
    Vector3d F = Vector3d::Zero();
    VectorXd g = VectorXd::Zero(dof);

    string file_path = std::string(HW_FOLDER) + "/hw3/logs/hw3_q" + std::to_string(controller_number);
    SaiCommon::Logger logger(file_path, false);

    if (controller_number == 1 || controller_number == 2 || controller_number == 4) {
        logger.addToLog(x, "x");
        logger.addToLog(x_des, "x_des");
        logger.addToLog(robot_q, "q");
        logger.addToLog(q_desired, "q_des");
        logger.addToLog(dx, "dx");
    }
    else if (controller_number == 3) {
        logger.addToLog(x, "x"); 
        logger.addToLog(x_des, "x_des"); 
        logger.addToLog(robot_q, "delta_phi");   
    }

    logger.start(100);


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

            Vector3d dx_des = Vector3d::Zero();
            Vector3d ddx_des = Vector3d::Zero();

            x = robot->position(link_name, pos_in_link);
            Jv = robot->Jv(link_name, pos_in_link);
            dx = Jv * robot_dq;

            Lambda = robot->taskInertiaMatrix(Jv);
            J_bar = robot->dynConsistentInverseJacobian(Jv);

            g = robot->jointGravityVector();
        // ---------------------------  question 1 ---------------------------------------
        if(controller_number == 1) {
            double kp = 100.0;
            double kv = 20.0;
            double kpj = 50.0;
            double kvj = 14.0;
            x_des << 0.3 + 0.1 * sin(M_PI * time),
                     0.1 + 0.1 * cos(M_PI * time),
                     0.5;

            dx_des << 0.1 * M_PI * cos(M_PI * time),
                     -0.1 * M_PI * sin(M_PI * time),
                     0.0;

            ddx_des << -0.1 * M_PI * M_PI * sin(M_PI * time),
                     -0.1 * M_PI * M_PI * cos(M_PI * time),
                     0.0;

            N = MatrixXd::Identity(dof, dof) - J_bar * Jv;

            // a. 
            F = Lambda * (-kp * (x - x_des) - kv * dx);
            //F = Lambda * (ddx_des - kp * (x - x_des) - kv * (dx - dx_des));

           control_torques =
             Jv.transpose() * F
              + N.transpose() * (-kpj * (robot_q - q_desired) - kvj * robot_dq)
              + g;
        }

        // ---------------------------  question 2 ---------------------------------------
        else if(controller_number == 2) {
            double kp = 100.0;
            double kv = 20.0;
            double kpj = 50.0;
            double kvj = 14.0;
            double k_mid = 25.0;
            double k_damp = 14.0;
            
           //2.d VectorXd x_desired(3);
            // 2.e x_des << -0.1, 0.15, 0.2;
            x_des << -0.65, -0.45, 0.7;
            dx_des.setZero();

            VectorXd q_mid(dof);
            q_mid << 0.0, 0.0, 0.0, -100.0 * M_PI / 180.0,
            0.0, 105.0 * M_PI / 180.0, 0.0;

            VectorXd Gamma_mid = -2.0 * k_mid * (robot_q - q_mid);
            VectorXd Gamma_damp = -k_damp * robot_dq;

            F = Lambda * (-kp * (x - x_des) - kv * (dx - dx_des));
            N = MatrixXd::Identity(dof, dof) - J_bar * Jv;
            //2.d control_torques = Jv.transpose() * F + N.transpose() * (-kvj * robot_dq) + g;
            // 2.f control_torques = Jv.transpose() * F + N.transpose() * Gamma_mid
            //+ N.transpose() * Gamma_damp + g;
            control_torques = Jv.transpose() * F + Gamma_mid
                        + N.transpose() * Gamma_damp + g;  
        } 
        // ---------------------------  question 3 ---------------------------------------
        else if(controller_number == 3) {
            gravity_comp_enabled = true;

            double kp_pos = 10;
            double kv_pos = 7;

            double kp_ori = 10;
            double kv_ori = 7;

            double kvj = 14.0;

            MatrixXd J0 = robot->J(link_name, pos_in_link);

            MatrixXd Lambda0 = robot->taskInertiaMatrix(J0);
            MatrixXd N0 = robot->nullspaceMatrix(J0);

            x = robot->position(link_name, pos_in_link);
            VectorXd vel_6d = J0 * robot_dq;

            Vector3d dx = vel_6d.head(3);
            Vector3d omega = vel_6d.tail(3);

            x_des << 0.6, 0.3, 0.5;

            Matrix3d R = robot->rotation(link_name);

            Matrix3d R_desired;
            R_desired << cos(M_PI / 3.0), 0.0, sin(M_PI / 3.0),
                        0.0,             1.0, 0.0,
                        -sin(M_PI / 3.0), 0.0, cos(M_PI / 3.0);

            Matrix3d R_err = R_desired * R.transpose();
            AngleAxisd aa(R_err);

            Vector3d delta_phi = aa.angle() * aa.axis();

            VectorXd command(6);
            command.head(3) = kp_pos * (x_des - x) - kv_pos * dx;
            command.tail(3) = kp_ori * (delta_phi) - kv_ori * omega;

            VectorXd F = Lambda0 * command;

            VectorXd Gamma_damp = -kvj * robot_dq;

            control_torques = J0.transpose() * F + N0.transpose() * Gamma_damp;
        }
        // ---------------------------  question 4 ---------------------------------------
        else if(controller_number == 4) {

            gravity_comp_enabled = true;

            double kp = 200.0;
            double kv = 2.0 * sqrt(kp);   // critical damping, about 28.3

            double kpj = 50.0;
            double kvj = 14.0;
            double Vmax = 0.1;

            Jv = robot->Jv(link_name, pos_in_link);
            Lambda = robot->taskInertiaMatrix(Jv);
            N = robot->nullspaceMatrix(Jv);
            J_bar = robot->dynConsistentInverseJacobian(Jv);
            x = robot->position(link_name, pos_in_link);
            Vector3d dx = Jv * robot_dq;
            //dx_log = dx;

            x_des << 0.6, 0.3, 0.4;

            q_desired = VectorXd::Zero(dof);

            MatrixXd M = robot->M();
            dx_des = (kp / kv) * (x_des - x);
            double dx_des_norm = dx_des.norm();
            double nu = 1.0;
            if (dx_des_norm > 1e-6) {
                double ratio = Vmax / dx_des_norm;

                if (fabs(ratio) <= 1.0) {
                    nu = ratio;
                } else {
                    nu = (ratio > 0.0) ? 1.0 : -1.0;
                }
             }

            // 4.a Vector3d F = Lambda * (kp * (x_des - x) - kv * dx);
            F = Lambda * (-kv * (dx - nu * dx_des));
            VectorXd posture_term =
                M * (-kpj * (robot_q - q_desired) - kvj * robot_dq);

            control_torques =
                Jv.transpose() * F
                + N.transpose() * posture_term;
            
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
