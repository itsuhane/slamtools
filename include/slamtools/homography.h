#pragma once

#include <slamtools/common.h>

bool decompose_homography(const Eigen::Matrix3d &H, Eigen::Matrix3d &R1, Eigen::Matrix3d &R2, Eigen::Vector3d &T1, Eigen::Vector3d &T2, Eigen::Vector3d &n1, Eigen::Vector3d &n2);

// p2 = H * p1
Eigen::Matrix3d solve_homography_4pt(const std::array<Eigen::Vector2d, 4> &points1, const std::array<Eigen::Vector2d, 4> &points2);

// p2 = H * p1
Eigen::Matrix3d solve_homography(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2);

// d(p2, H * p1)
inline double homography_geometric_error(const Eigen::Matrix3d &H, const Eigen::Vector2d &p1, const Eigen::Vector2d &p2) {
    return (p2 - (H * p1.homogeneous()).hnormalized()).squaredNorm();
}
