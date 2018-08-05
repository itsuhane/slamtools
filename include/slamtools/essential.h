#pragma once

#include <slamtools/common.h>

void decompose_essential(const Eigen::Matrix3d &E, Eigen::Matrix3d &R1, Eigen::Matrix3d &R2, Eigen::Vector3d &T);

// p2^T * E * p1 = 0
std::vector<Eigen::Matrix3d> solve_essential_5pt(const std::array<Eigen::Vector2d, 5> &points1, const std::array<Eigen::Vector2d, 5> &points2);

// d(p2, E * p1)
inline double essential_geometric_error(const Eigen::Matrix3d &E, const Eigen::Vector2d &p1, const Eigen::Vector2d &p2) {
    Eigen::Vector3d Ep1 = E * p1.homogeneous();
    double r = p2.homogeneous().transpose() * Ep1;
    return r * r / Ep1.segment<2>(0).squaredNorm();
}
