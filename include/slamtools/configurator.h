#pragma once

#include <slamtools/common.h>

class Configurator {
  public:
    static std::shared_ptr<Configurator> default_config();
    virtual ~Configurator();
    virtual Eigen::Matrix3d camera_intrinsic() const;
    virtual double keypoint_pixel_error() const;
    virtual Eigen::Quaterniond camera_to_center_rotation() const;
    virtual Eigen::Vector3d camera_to_center_translation() const;
    virtual Eigen::Quaterniond imu_to_center_rotation() const;
    virtual Eigen::Vector3d imu_to_center_translation() const;
    virtual Eigen::Matrix3d imu_gyro_white_noise() const;
    virtual Eigen::Matrix3d imu_accel_white_noise() const;
    virtual Eigen::Matrix3d imu_gyro_random_walk() const;
    virtual Eigen::Matrix3d imu_accel_random_walk() const;
    virtual int random() const;
    virtual size_t max_keypoint_detection() const;
    virtual size_t max_init_raw_frames() const;
    virtual size_t min_init_raw_frames() const;
    virtual size_t min_raw_matches() const;
    virtual double min_raw_parallax() const;
    virtual size_t min_raw_triangulation() const;
    virtual size_t init_map_frames() const;
    virtual size_t min_init_map_landmarks() const;
    virtual bool init_refine_imu() const;
    virtual size_t solver_iteration_limit() const;
    virtual double solver_time_limit() const;
    virtual size_t tracking_window_size() const;
    virtual bool predict_keypoints() const;
};
