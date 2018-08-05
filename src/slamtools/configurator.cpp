#include <slamtools/configurator.h>

using namespace Eigen;

std::shared_ptr<Configurator> Configurator::default_config() {
    static std::shared_ptr<Configurator> s_default_config = std::make_shared<Configurator>();
    return s_default_config;
}

Configurator::~Configurator() = default;

Matrix3d Configurator::camera_intrinsic() const {
    return (Matrix3d() << 549.0, 0.0, 320.0, 0.0, 549.0, 240.0, 0.0, 0.0, 1.0).finished();
}

double Configurator::keypoint_pixel_error() const {
    return 0.7;
}

Quaterniond Configurator::camera_to_center_rotation() const {
    Quaterniond q;
    q = (Matrix3d() << -Vector3d::UnitY(), -Vector3d::UnitX(), -Vector3d::UnitZ()).finished();
    return q;
}

Vector3d Configurator::camera_to_center_translation() const {
    return {0.0, 0.065, 0.0};
}

Quaterniond Configurator::imu_to_center_rotation() const {
    return Quaterniond::Identity();
}

Vector3d Configurator::imu_to_center_translation() const {
    return Vector3d::Zero();
}

Matrix3d Configurator::imu_gyro_white_noise() const {
    return 4e-4 * Matrix3d::Identity();
}

Matrix3d Configurator::imu_accel_white_noise() const {
    return 2.5e-3 * Matrix3d::Identity();
}

Matrix3d Configurator::imu_gyro_random_walk() const {
    return 1.6e-9 * Matrix3d::Identity();
}

Matrix3d Configurator::imu_accel_random_walk() const {
    return 4e-6 * Matrix3d::Identity();
}

int Configurator::random() const {
    return 648; // <-- should be a random seed.
}

size_t Configurator::max_keypoint_detection() const {
    return 150;
}

size_t Configurator::max_init_raw_frames() const {
    return (init_map_frames() - 1) * 3 + 1;
}

size_t Configurator::min_init_raw_frames() const {
    return (init_map_frames() - 1) * 2 + 1;
}

size_t Configurator::min_raw_matches() const {
    return 50;
}

double Configurator::min_raw_parallax() const {
    return 10;
}

size_t Configurator::min_raw_triangulation() const {
    return 20;
}

size_t Configurator::init_map_frames() const {
    return 8;
}

size_t Configurator::min_init_map_landmarks() const {
    return 30;
}

bool Configurator::init_refine_imu() const {
    return true;
}

size_t Configurator::solver_iteration_limit() const {
    return 10;
}

double Configurator::solver_time_limit() const {
    return 1.0e6;
}

size_t Configurator::tracking_window_size() const {
    return init_map_frames();
}

bool Configurator::predict_keypoints() const {
    return true;
}
