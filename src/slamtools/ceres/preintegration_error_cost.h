#pragma once

#include <slamtools/common.h>
#include <ceres/ceres.h>
#include <slamtools/state.h>
#include <slamtools/factor.h>
#include <slamtools/preintegrator.h>
#include <slamtools/frame.h>
#include <slamtools/lie_algebra.h>

class PreIntegrationErrorCost : public Factor::FactorCostFunction, public ceres::SizedCostFunction<15, 4, 3, 3, 3, 3, 4, 3, 3, 3, 3> {
  public:
    PreIntegrationErrorCost(const Frame *frame_i, const Frame *frame_j) :
        frame_i(frame_i), frame_j(frame_j) {
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        static const Eigen::Vector3d gravity = {0.0, 0.0, -GRAVITY_NOMINAL};
        Eigen::Map<const Eigen::Quaterniond> q_center_i(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> p_center_i(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> v_i(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> bg_i(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> ba_i(parameters[4]);

        Eigen::Map<const Eigen::Quaterniond> q_center_j(parameters[5]);
        Eigen::Map<const Eigen::Vector3d> p_center_j(parameters[6]);
        Eigen::Map<const Eigen::Vector3d> v_j(parameters[7]);
        Eigen::Map<const Eigen::Vector3d> bg_j(parameters[8]);
        Eigen::Map<const Eigen::Vector3d> ba_j(parameters[9]);

        const PreIntegrator &pre = frame_j->preintegration;
        const ExtrinsicParams &imu_i = frame_i->imu;
        const ExtrinsicParams &imu_j = frame_j->imu;
        const Eigen::Vector3d &ba_i_0 = frame_i->motion.ba;
        const Eigen::Vector3d &bg_i_0 = frame_i->motion.bg;

        const Eigen::Quaterniond q_i = q_center_i * imu_i.q_cs;
        const Eigen::Vector3d p_i = p_center_i + q_center_i * imu_i.p_cs;
        const Eigen::Quaterniond q_j = q_center_j * imu_j.q_cs;
        const Eigen::Vector3d p_j = p_center_j + q_center_j * imu_j.p_cs;

        const double &dt = pre.delta.t;
        const Eigen::Quaterniond &dq = pre.delta.q;
        const Eigen::Vector3d &dp = pre.delta.p;
        const Eigen::Vector3d &dv = pre.delta.v;
        const Eigen::Vector3d dbg = bg_i - bg_i_0;
        const Eigen::Vector3d dba = ba_i - ba_i_0;

        const Eigen::Matrix3d &dq_dbg = pre.jacobian.dq_dbg;
        const Eigen::Matrix3d &dp_dbg = pre.jacobian.dp_dbg;
        const Eigen::Matrix3d &dp_dba = pre.jacobian.dp_dba;
        const Eigen::Matrix3d &dv_dbg = pre.jacobian.dv_dbg;
        const Eigen::Matrix3d &dv_dba = pre.jacobian.dv_dba;

        Eigen::Map<Eigen::Matrix<double, 15, 1>> r(residuals);
        r.segment<3>(ES_Q) = logmap((dq * expmap(dq_dbg * dbg)).conjugate() * q_i.conjugate() * q_j);
        r.segment<3>(ES_P) = q_i.conjugate() * (p_j - p_i - dt * v_i - 0.5 * dt * dt * gravity) - (dp + dp_dbg * dbg + dp_dba * dba);
        r.segment<3>(ES_V) = q_i.conjugate() * (v_j - v_i - dt * gravity) - (dv + dv_dbg * dbg + dv_dba * dba);
        r.segment<3>(ES_BG) = bg_j - bg_i;
        r.segment<3>(ES_BA) = ba_j - ba_i;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> dr_dq_i(jacobians[0]);
                dr_dq_i.setZero();
                dr_dq_i.block<3, 3>(ES_Q, 0) = -right_jacobian(r.segment<3>(ES_Q)).inverse() * q_j.conjugate().matrix() * q_center_i.matrix();
                dr_dq_i.block<3, 3>(ES_P, 0) = imu_i.q_cs.conjugate().matrix() * hat(q_center_i.conjugate() * (p_j - p_center_i - dt * v_i - 0.5 * dt * dt * gravity));
                dr_dq_i.block<3, 3>(ES_V, 0) = imu_i.q_cs.conjugate().matrix() * hat(q_center_i.conjugate() * (v_j - v_i - dt * gravity));
                dr_dq_i = pre.delta.sqrt_inv_cov * dr_dq_i;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dp_i(jacobians[1]);
                dr_dp_i.setZero();
                dr_dp_i.block<3, 3>(ES_P, 0) = -q_i.conjugate().matrix();
                dr_dp_i = pre.delta.sqrt_inv_cov * dr_dp_i;
            }
            if (jacobians[2]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dv_i(jacobians[2]);
                dr_dv_i.setZero();
                dr_dv_i.block<3, 3>(ES_P, 0) = -dt * q_i.conjugate().matrix();
                dr_dv_i.block<3, 3>(ES_V, 0) = -q_i.conjugate().matrix();
                dr_dv_i = pre.delta.sqrt_inv_cov * dr_dv_i;
            }
            if (jacobians[3]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dbg_i(jacobians[3]);
                dr_dbg_i.setZero();
                dr_dbg_i.block<3, 3>(ES_Q, 0) = -right_jacobian(r.segment<3>(ES_Q)).inverse() * expmap(r.segment<3>(ES_Q)).conjugate().matrix() * right_jacobian(dq_dbg * dbg) * dq_dbg;
                dr_dbg_i.block<3, 3>(ES_P, 0) = -dp_dbg;
                dr_dbg_i.block<3, 3>(ES_V, 0) = -dv_dbg;
                dr_dbg_i.block<3, 3>(ES_BG, 0) = -Eigen::Matrix3d::Identity();
                dr_dbg_i = pre.delta.sqrt_inv_cov * dr_dbg_i;
            }
            if (jacobians[4]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dba_i(jacobians[4]);
                dr_dba_i.setZero();
                dr_dba_i.block<3, 3>(ES_P, 0) = -dp_dba;
                dr_dba_i.block<3, 3>(ES_V, 0) = -dv_dba;
                dr_dba_i.block<3, 3>(ES_BA, 0) = -Eigen::Matrix3d::Identity();
                dr_dba_i = pre.delta.sqrt_inv_cov * dr_dba_i;
            }
            if (jacobians[5]) {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> dr_dq_j(jacobians[5]);
                dr_dq_j.setZero();
                dr_dq_j.block<3, 3>(ES_Q, 0) = right_jacobian(r.segment<3>(ES_Q)).inverse() * imu_j.q_cs.conjugate().matrix();
                dr_dq_j.block<3, 3>(ES_P, 0) = -q_i.conjugate().matrix() * q_center_j.matrix() * hat(imu_j.p_cs);
                dr_dq_j = pre.delta.sqrt_inv_cov * dr_dq_j;
            }
            if (jacobians[6]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dp_j(jacobians[6]);
                dr_dp_j.setZero();
                dr_dp_j.block<3, 3>(ES_P, 0) = q_i.conjugate().matrix();
                dr_dp_j = pre.delta.sqrt_inv_cov * dr_dp_j;
            }
            if (jacobians[7]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dv_j(jacobians[7]);
                dr_dv_j.setZero();
                dr_dv_j.block<3, 3>(ES_V, 0) = q_i.conjugate().matrix();
                dr_dv_j = pre.delta.sqrt_inv_cov * dr_dv_j;
            }
            if (jacobians[8]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dbg_j(jacobians[8]);
                dr_dbg_j.setZero();
                dr_dbg_j.block<3, 3>(ES_BG, 0).setIdentity();
                dr_dbg_j = pre.delta.sqrt_inv_cov * dr_dbg_j;
            }
            if (jacobians[9]) {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dba_j(jacobians[9]);
                dr_dba_j.setZero();
                dr_dba_j.block<3, 3>(ES_BA, 0).setIdentity();
                dr_dba_j = pre.delta.sqrt_inv_cov * dr_dba_j;
            }
        }

        r = pre.delta.sqrt_inv_cov * r;

        return true;
    }

  private:
    const Frame *frame_i;
    const Frame *frame_j;
};

class PreIntegrationPriorCost : public ceres::SizedCostFunction<15, 4, 3, 3, 3, 3> {
  public:
    PreIntegrationPriorCost(const Frame *frame_i, const Frame *frame_j) :
        error(frame_i, frame_j), frame_i(frame_i) {
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        std::array<const double *, 10> params = {
            frame_i->pose.q.coeffs().data(),
            frame_i->pose.p.data(),
            frame_i->motion.v.data(),
            frame_i->motion.bg.data(),
            frame_i->motion.ba.data(),
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4]};
        if (jacobians) {
            std::array<double *, 10> jacobs = {
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                jacobians[0],
                jacobians[1],
                jacobians[2],
                jacobians[3],
                jacobians[4]};
            return error.Evaluate(params.data(), residuals, jacobs.data());
        } else {
            return error.Evaluate(params.data(), residuals, nullptr);
        }
    }

  private:
    PreIntegrationErrorCost error;
    const Frame *frame_i;
};
