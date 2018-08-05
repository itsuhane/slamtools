#pragma once

#include <slamtools/common.h>

inline Eigen::Vector2d apply_k(const Eigen::Vector2d &p, const Eigen::Matrix3d &K) {
    return {p(0) * K(0, 0) + K(0, 2), p(1) * K(1, 1) + K(1, 2)};
}

inline Eigen::Vector2d remove_k(const Eigen::Vector2d &p, const Eigen::Matrix3d &K) {
    return {(p(0) - K(0, 2)) / K(0, 0), (p(1) - K(1, 2)) / K(1, 1)};
}

Eigen::Matrix3d find_essential_matrix(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, double threshold = 1.0, double confidence = 0.999, size_t max_iteration = 1000, int seed = 0);
Eigen::Matrix3d find_homography_matrix(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, double threshold = 1.0, double confidence = 0.999, size_t max_iteration = 1000, int seed = 0);
Eigen::Matrix3d find_essential_matrix(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, std::vector<char> &inlier_mask, double threshold = 1.0, double confidence = 0.999, size_t max_iteration = 1000, int seed = 0);
Eigen::Matrix3d find_homography_matrix(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, std::vector<char> &inlier_mask, double threshold = 1.0, double confidence = 0.999, size_t max_iteration = 1000, int seed = 0);

Eigen::Vector4d triangulate_point(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2);
Eigen::Vector4d triangulate_point(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points);

bool triangulate_point_checked(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2, Eigen::Vector3d &p);
bool triangulate_point_checked(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points, Eigen::Vector3d &p);
bool triangulate_point_scored(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2, Eigen::Vector3d &p, double &score);
bool triangulate_point_scored(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points, Eigen::Vector3d &p, double &score);

size_t triangulate_from_rt(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const Eigen::Matrix3d &R, const Eigen::Vector3d &T, std::vector<Eigen::Vector3d> &result_points, std::vector<char> &result_status);
size_t triangulate_from_rt(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const std::vector<Eigen::Matrix3d> &Rs, const std::vector<Eigen::Vector3d> &Ts, std::vector<Eigen::Vector3d> &result_points, Eigen::Matrix3d &result_R, Eigen::Vector3d &result_T, std::vector<char> &result_status);

size_t triangulate_from_rt_scored(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const Eigen::Matrix3d &R, const Eigen::Vector3d &T, std::vector<Eigen::Vector3d> &result_points, std::vector<char> &result_status, double &score);
size_t triangulate_from_rt_scored(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const std::vector<Eigen::Matrix3d> &Rs, const std::vector<Eigen::Vector3d> &Ts, size_t count_threshold, std::vector<Eigen::Vector3d> &result_points, Eigen::Matrix3d &result_R, Eigen::Vector3d &result_T, std::vector<char> &result_status);

size_t triangulate_from_essential(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const Eigen::Matrix3d &E, std::vector<Eigen::Vector3d> &result_points, std::vector<char> &result_status, Eigen::Matrix3d &R, Eigen::Vector3d &T);
size_t triangulate_from_homography(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2, const Eigen::Matrix3d &H, std::vector<Eigen::Vector3d> &result_points, std::vector<char> &result_status, Eigen::Matrix3d &R, Eigen::Vector3d &T);

inline Eigen::Vector4d triangulate_point(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2) {
    Eigen::Matrix4d A;
    A.row(0) = point1(0) * P1.row(2) - P1.row(0);
    A.row(1) = point1(1) * P1.row(2) - P1.row(1);
    A.row(2) = point2(0) * P2.row(2) - P2.row(0);
    A.row(3) = point2(1) * P2.row(2) - P2.row(1);
    return A.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);
}

inline Eigen::Vector4d triangulate_point(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points) {
    Eigen::Matrix<double, Eigen::Dynamic, 4> A(points.size() * 2, 4);
    for (size_t i = 0; i < points.size(); ++i) {
        A.row(i * 2 + 0) = points[i](0) * Ps[i].row(2) - Ps[i].row(0);
        A.row(i * 2 + 1) = points[i](1) * Ps[i].row(2) - Ps[i].row(1);
    }
    return A.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);
}

inline bool triangulate_point_checked(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2, Eigen::Vector3d &p) {
    double score;
    return triangulate_point_scored(P1, P2, point1, point2, p, score);
}

inline bool triangulate_point_checked(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points, Eigen::Vector3d &p) {
    double score;
    return triangulate_point_scored(Ps, points, p, score);
}

inline bool triangulate_point_scored(const Eigen::Matrix<double, 3, 4> &P1, const Eigen::Matrix<double, 3, 4> &P2, const Eigen::Vector2d &point1, const Eigen::Vector2d &point2, Eigen::Vector3d &p, double &score) {
    Eigen::Vector4d q = triangulate_point(P1, P2, point1, point2);
    score = 0;

    Eigen::Vector3d q1 = P1 * q;
    Eigen::Vector3d q2 = P2 * q;

    if (q1[2] * q[3] > 0 && q2[2] * q[3] > 0) {
        if (q1[2] / q[3] < 100 && q2[2] / q[3] < 100) {
            p = q.hnormalized();
            score = 0.5 * ((q1.hnormalized() - point1).squaredNorm() + (q2.hnormalized() - point2).squaredNorm());
            return true;
        }
    }

    return false;
}

inline bool triangulate_point_scored(const std::vector<Eigen::Matrix<double, 3, 4>> &Ps, const std::vector<Eigen::Vector2d> &points, Eigen::Vector3d &p, double &score) {
    if (Ps.size() < 2) return false;
    Eigen::Vector4d q = triangulate_point(Ps, points);
    score = 0;

    for (size_t i = 0; i < points.size(); ++i) {
        Eigen::Vector3d qi = Ps[i] * q;
        if (!(qi[2] * q[3] > 0)) {
            return false;
        }
        if (!(qi[2] / q[3] < 100)) {
            return false;
        }
        score += (qi.hnormalized() - points[i]).squaredNorm();
    }
    score /= points.size();
    p = q.hnormalized();
    return true;
}
