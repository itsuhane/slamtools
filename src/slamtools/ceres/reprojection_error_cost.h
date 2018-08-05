#pragma once

#include <slamtools/common.h>
#include <ceres/ceres.h>
#include <slamtools/state.h>
#include <slamtools/factor.h>
#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/lie_algebra.h>

class ReprojectionErrorCost : public Factor::FactorCostFunction, public ceres::SizedCostFunction<2, 4, 3, 3> {
  public:
    ReprojectionErrorCost(const Frame *frame, size_t keypoint_id) :
        frame(frame), keypoint_id(keypoint_id) {
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<const Eigen::Quaterniond> q_center(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> p_center(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> x(parameters[2]);

        const Eigen::Vector2d &z = frame->get_keypoint(keypoint_id);
        const ExtrinsicParams &camera = frame->camera;
        const Eigen::Matrix2d &sqrt_inv_cov = frame->sqrt_inv_cov;

        //Eigen::Vector3d y = camera.q_cs.conjugate() * q_center.conjugate() * (x - p_center) - camera.q_cs.conjugate() * camera.p_cs;
        Eigen::Vector3d y_center = q_center.conjugate() * (x - p_center);
        Eigen::Vector3d y = camera.q_cs.conjugate() * y_center - camera.q_cs.conjugate() * camera.p_cs;
        Eigen::Map<Eigen::Vector2d> r(residuals);
        r = y.hnormalized() - z;

        if (jacobians) {
            Eigen::Matrix<double, 2, 3> dr_dy;
            dr_dy << 1.0 / y.z(), 0.0, -y.x() / (y.z() * y.z()),
                0.0, 1.0 / y.z(), -y.y() / (y.z() * y.z());
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> dr_dq(jacobians[0]);
                dr_dq.block<2, 3>(0, 0) = dr_dy * camera.q_cs.conjugate().matrix() * hat(y_center);
                dr_dq.col(3).setZero();
                dr_dq = sqrt_inv_cov * dr_dq;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> dr_dp(jacobians[1]);
                dr_dp = -dr_dy * (q_center * camera.q_cs).conjugate().matrix();
                dr_dp = sqrt_inv_cov * dr_dp;
            }
            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> dr_dx(jacobians[2]);
                dr_dx = dr_dy * (q_center * camera.q_cs).conjugate().matrix();
                dr_dx = sqrt_inv_cov * dr_dx;
            }
        }

        r = sqrt_inv_cov * r;

        return true;
    };

  private:
    const Frame *frame;
    const size_t keypoint_id;
};

class PoseOnlyReprojectionErrorCost : public ceres::SizedCostFunction<2, 4, 3> {
  public:
    PoseOnlyReprojectionErrorCost(const Frame *frame, size_t keypoint_id) :
        error(frame, keypoint_id), x(frame->get_track(keypoint_id)->landmark.x) {
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        std::array<const double *, 3> params = {
            parameters[0],
            parameters[1],
            x.data()};
        if (jacobians) {
            std::array<double *, 3> jacobs = {
                jacobians[0],
                jacobians[1],
                nullptr};
            return error.Evaluate(params.data(), residuals, jacobs.data());
        } else {
            return error.Evaluate(params.data(), residuals, nullptr);
        }
    }

  private:
    ReprojectionErrorCost error;
    const Eigen::Vector3d &x;
};
