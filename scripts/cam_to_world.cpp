// =============================================================================
//  cam_to_world.cpp
//
//  PURPOSE:  Takes a 3-D point in the RealSense camera frame and continuously
//            outputs its position in the robot world frame.
//
//  INPUT:    x, y, z  in camera frame  (metres)
//            – fed live from a RealSense D400 depth frame + YOLO pixel coords
//
//  OUTPUT:   x, y, z  in world frame   (metres)
//            – ready to pass directly into your Flexiv control algorithm
//
//  HARDWARE: Intel RealSense D435/D435i  (or any D400-series)
//            Flexiv Rizon 4s  (RDK v1.9+)
//            Camera mounted on end-effector (eye-in-hand)
//
//  DEPS:     flexiv_rdk   (https://github.com/flexivrobotics/flexiv_rdk)
//            librealsense2
//            Eigen 3.4+
//
//  BUILD:
//    mkdir build && cd build
//    cmake .. -DCMAKE_BUILD_TYPE=Release \
//             -DCMAKE_PREFIX_PATH="~/rdk_install"
//    make -j$(nproc)
//
//  RUN:
//    ./cam_to_world <robot_serial>        e.g.  Rizon4s-123456
// =============================================================================

#include <flexiv/rdk/robot.hpp>
#include <flexiv/rdk/utility.hpp>

#include <librealsense2/rs.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <thread>

// ── Graceful shutdown ────────────────────────────────────────────────────────
static std::atomic<bool> g_running{true};
static void on_signal(int) { g_running = false; }

// ── Helpers ──────────────────────────────────────────────────────────────────

// Build a 4×4 homogeneous transform from Flexiv's 7-element pose array:
//   [x, y, z, qw, qx, qy, qz]
inline Eigen::Matrix4d pose_to_T(const std::array<double, 7>& p)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    // Flexiv stores quaternion as [qw, qx, qy, qz]
    Eigen::Quaterniond q(p[3], p[4], p[5], p[6]);
    T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    T.block<3, 1>(0, 3) = Eigen::Vector3d(p[0], p[1], p[2]);
    return T;
}

// Apply a 4×4 transform to a 3-D point
inline Eigen::Vector3d transform_point(const Eigen::Matrix4d& T,
                                        const Eigen::Vector3d& p)
{
    Eigen::Vector4d ph(p.x(), p.y(), p.z(), 1.0);
    return (T * ph).head<3>();
}

// ─────────────────────────────────────────────────────────────────────────────
//  CamToWorldConverter
//
//  Single class that owns:
//    – RealSense pipeline      (grabs depth + colour)
//    – Flexiv robot handle     (reads flange pose = T_flange_in_world)
//    – Fixed T_cam_in_flange   (set once from hand-eye calibration)
//
//  Public API:
//    convert(u, v, confidence)  →  WorldPoint
//    run_loop()                 →  continuous print loop (for testing)
// ─────────────────────────────────────────────────────────────────────────────
struct WorldPoint {
    Eigen::Vector3d xyz;        // metres, world frame
    float           confidence; // YOLO confidence passed through
    uint64_t        timestamp_ns;
};

class CamToWorldConverter {
public:
    // ── Construction ─────────────────────────────────────────────────────────
    explicit CamToWorldConverter(const std::string& robot_sn)
        : robot_(robot_sn)
    {
        // Wait until robot is operational
        std::cout << "[Init] Connecting to Flexiv " << robot_sn << " ...\n";
        if (robot_.fault()) {
            std::cout << "[Init] Clearing robot fault...\n";
            if (!robot_.ClearFault())
                throw std::runtime_error("Could not clear robot fault.");
        }
        robot_.Enable();
        while (!robot_.operational())
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "[Init] Robot operational.\n";

        // ── RealSense pipeline ────────────────────────────────────────────────
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16,  30);
        rs_pipe_.start(cfg);

        // Warm up: discard first few frames (auto-exposure settling)
        for (int i = 0; i < 30; ++i) rs_pipe_.wait_for_frames();

        // Cache intrinsics from the depth stream
        auto profile   = rs_pipe_.get_active_profile();
        auto depth_str = profile.get_stream(RS2_STREAM_DEPTH)
                             .as<rs2::video_stream_profile>();
        intr_ = depth_str.get_intrinsics();

        // Depth scale: RealSense raw value → metres
        rs2::device dev       = profile.get_device();
        auto         sensor   = dev.first<rs2::depth_sensor>();
        depth_scale_          = sensor.get_depth_scale();

        std::cout << "[Init] RealSense ready. Depth scale = "
                  << depth_scale_ << " m/unit\n";

        // ── Default hand-eye calibration ─────────────────────────────────────
        // Replace with your cv::calibrateHandEye() result.
        // Camera is mounted on the flange, tilted -30° around X, offset forward.
        set_cam_in_flange_euler("XYZ",
            -30.0, 0.0, 0.0,             // rotation  (degrees)
            {0.05, 0.0, 0.02}            // translation [m]
        );
    }

    // ── Set hand-eye calibration from Euler angles (degrees) + translation ───
    void set_cam_in_flange_euler(const std::string& order,
                                  double a1, double a2, double a3,
                                  Eigen::Vector3d t)
    {
        auto axis_vec = [](char c) -> Eigen::Vector3d {
            if (c == 'X' || c == 'x') return Eigen::Vector3d::UnitX();
            if (c == 'Y' || c == 'y') return Eigen::Vector3d::UnitY();
            return Eigen::Vector3d::UnitZ();
        };
        constexpr double DEG = M_PI / 180.0;
        Eigen::Matrix3d R =
            (Eigen::AngleAxisd(a1 * DEG, axis_vec(order[0])) *
             Eigen::AngleAxisd(a2 * DEG, axis_vec(order[1])) *
             Eigen::AngleAxisd(a3 * DEG, axis_vec(order[2]))).toRotationMatrix();

        T_cam_in_flange_.setIdentity();
        T_cam_in_flange_.block<3, 3>(0, 0) = R;
        T_cam_in_flange_.block<3, 1>(0, 3) = t;
        std::cout << "[Calibration] T_cam_in_flange set.\n";
    }

    // ── Set hand-eye calibration directly from a rotation matrix ─────────────
    void set_cam_in_flange(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
    {
        T_cam_in_flange_.setIdentity();
        T_cam_in_flange_.block<3, 3>(0, 0) = R;
        T_cam_in_flange_.block<3, 1>(0, 3) = t;
    }

    // ── Core API: take a YOLO detection, return world-frame position ──────────
    //
    //   u, v        – pixel coords of mouse bbox centre (from YOLO)
    //   confidence  – YOLO confidence score
    //   out         – filled on success
    //   returns     – true if depth was valid and FK data is fresh
    //
    bool convert(float u, float v, float confidence, WorldPoint& out)
    {
        // 1. Grab a RealSense frameset (non-blocking poll; returns false if none)
        rs2::frameset frames;
        if (!rs_pipe_.poll_for_frames(&frames)) return false;

        rs2::depth_frame depth = frames.get_depth_frame();
        if (!depth) return false;

        uint64_t ts = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());

        // 2. Sample depth at (u, v) with 3×3 median filter for robustness
        float depth_m = sample_depth_median(depth, static_cast<int>(u),
                                             static_cast<int>(v));
        if (depth_m < 0.1f || depth_m > 4.0f)
            return false;  // out of reliable range

        // 3. Unproject pixel → camera-frame 3-D using RealSense intrinsics
        float pixel[2]  = {u, v};
        float point3d[3] = {};
        rs2_deproject_pixel_to_point(point3d, &intr_, pixel, depth_m);
        Eigen::Vector3d P_cam(point3d[0], point3d[1], point3d[2]);

        // 4. Get current flange pose from Flexiv (T_flange_in_world)
        //    robot.states().flange_pose = [x, y, z, qw, qx, qy, qz]
        auto flange_pose = robot_.states().flange_pose;
        Eigen::Matrix4d T_flange_in_world = pose_to_T(flange_pose);

        // 5. Full chain:
        //    P_world = T_flange_in_world * T_cam_in_flange * P_cam
        Eigen::Matrix4d T_cam_in_world = T_flange_in_world * T_cam_in_flange_;
        out.xyz          = transform_point(T_cam_in_world, P_cam);
        out.confidence   = confidence;
        out.timestamp_ns = ts;
        return true;
    }

    // ── Convenience: grab a live colour frame as a cv::Mat-compatible buffer ──
    //    Returns false if no frame is available this poll.
    bool grab_color_frame(std::vector<uint8_t>& bgr_out,
                           int& width_out, int& height_out)
    {
        rs2::frameset frames;
        if (!rs_pipe_.poll_for_frames(&frames)) return false;
        rs2::video_frame color = frames.get_color_frame();
        if (!color) return false;
        width_out  = color.get_width();
        height_out = color.get_height();
        const uint8_t* data = reinterpret_cast<const uint8_t*>(color.get_data());
        bgr_out.assign(data, data + width_out * height_out * 3);
        return true;
    }

    // ── Demo loop: accepts pixel coords from stdin, prints world coords ───────
    void run_loop()
    {
        std::cout << "\n[Loop] Enter YOLO detections as:  u v confidence\n"
                  << "       (or Ctrl-C to quit)\n\n";

        while (g_running) {
            float u, v, conf;
            if (!(std::cin >> u >> v >> conf)) break;

            WorldPoint wp;
            if (convert(u, v, conf, wp)) {
                std::printf(
                    "[World] x=%+.4f  y=%+.4f  z=%+.4f  [m]   conf=%.3f\n",
                    wp.xyz.x(), wp.xyz.y(), wp.xyz.z(), wp.confidence);
            } else {
                std::puts("[World] No valid depth or FK data at this frame.");
            }
        }
    }

    // ── Destructor ────────────────────────────────────────────────────────────
    ~CamToWorldConverter() { rs_pipe_.stop(); }

private:
    // 3×3 median depth sample centred on (u, v) for sub-pixel noise rejection
    float sample_depth_median(const rs2::depth_frame& depth, int u, int v)
    {
        int   w  = depth.get_width();
        int   h  = depth.get_height();
        float ds = depth_scale_;

        std::array<float, 9> samples{};
        int  n = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int x = std::clamp(u + dx, 0, w - 1);
                int y = std::clamp(v + dy, 0, h - 1);
                uint16_t raw = *reinterpret_cast<const uint16_t*>(
                    static_cast<const uint8_t*>(depth.get_data())
                    + y * depth.get_stride_in_bytes()
                    + x * sizeof(uint16_t));
                if (raw > 0) samples[n++] = raw * ds;
            }
        }
        if (n == 0) return 0.0f;
        std::sort(samples.begin(), samples.begin() + n);
        return samples[n / 2];
    }

    flexiv::rdk::Robot  robot_;
    rs2::pipeline       rs_pipe_;
    rs2_intrinsics      intr_{};
    float               depth_scale_{0.001f};
    Eigen::Matrix4d     T_cam_in_flange_ = Eigen::Matrix4d::Identity();
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <robot_serial_number>\n"
                  << "  e.g. " << argv[0] << " Rizon4s-123456\n";
        return 1;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    try {
        CamToWorldConverter converter(argv[1]);

        // ── OPTIONAL: override calibration with your actual result ─────────
        //
        // If you have run cv::calibrateHandEye() (OpenCV) and obtained
        // the rotation (R) and translation (t) from the camera frame into
        // the flange frame, then replace the example values below with
        // your calibration result and uncomment the call to apply them.
        //
        // Example (REPLACE these numbers with your calibration output):
        // // Rotation matrix from camera frame into flange frame (3×3)
        // Eigen::Matrix3d R_cam_in_flange;
        // R_cam_in_flange << 1.0, 0.0, 0.0,
        //                     0.0, 1.0, 0.0,
        //                     0.0, 0.0, 1.0;
        // // Translation vector (meters)
        // Eigen::Vector3d t_cam_in_flange(0.05, 0.0, 0.02);
        // // IMPORTANT: Replace the above with values from cv::calibrateHandEye()
        // // converter.set_cam_in_flange(R_cam_in_flange, t_cam_in_flange);
        //
        // Alternatively, you may provide a simple calibration file as argv[2]
        // containing 12 whitespace-separated numbers in this order:
        //   r11 r12 r13 r21 r22 r23 r31 r32 r33 tx ty tz
        // Example run with file:
        //   ./cam_to_world Rizon4s-XXXXXX config/camera_calib.txt
        // The program will attempt to read argv[2] below and apply it.

        if (argc >= 3) {
            std::ifstream f(argv[2]);
            if (f) {
                std::array<double, 12> vals{};
                bool ok = true;
                for (int i = 0; i < 12; ++i) {
                    if (!(f >> vals[i])) { ok = false; break; }
                }
                if (ok) {
                    Eigen::Matrix3d R;
                    R << vals[0], vals[1], vals[2],
                         vals[3], vals[4], vals[5],
                         vals[6], vals[7], vals[8];
                    Eigen::Vector3d t(vals[9], vals[10], vals[11]);
                    converter.set_cam_in_flange(R, t);
                    std::cout << "[Calibration] Loaded from " << argv[2] << "\n";
                } else {
                    std::cerr << "[WARN] Calibration file format invalid: expected 12 numbers.\n";
                }
            } else {
                std::cerr << "[WARN] Calibration file '" << argv[2]
                          << "' not found or unreadable. Using default calibration.\n";
            }
        }

        // ── Interactive demo loop ──────────────────────────────────────────
        // In production, replace run_loop() with your YOLO integration:
        //
        //   while (running) {
        //       auto det = yolo.detect(frame);          // your detection
        //       WorldPoint wp;
        //       if (converter.convert(det.u, det.v, det.confidence, wp)) {
        //           controls.update(wp.xyz);            // feed controller
        //       }
        //   }
        converter.run_loop();

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
