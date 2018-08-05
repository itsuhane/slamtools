#pragma once

#include <slamtools/common.h>
#include <ceres/ceres.h>
#include <slamtools/lie_algebra.h>

struct LieAlgebraEigenQuaternionParamatrization : public ceres::LocalParameterization {
    virtual bool Plus(const double *q, const double *dq, double *q_plus_dq) const override {
        Eigen::Map<Eigen::Quaterniond> result(q_plus_dq);
        result = (Eigen::Map<const Eigen::Quaterniond>(q) * expmap(Eigen::Map<const Eigen::Vector3d>(dq))).normalized();
        return true;
    }
    virtual bool ComputeJacobian(const double *, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);
        J.setIdentity(); // the composited jacobian is computed in PreIntegrationError::Evaluate(), we simply forward it.
        return true;
    }
    virtual int GlobalSize() const override {
        return 4;
    }
    virtual int LocalSize() const override {
        return 3;
    }
};
