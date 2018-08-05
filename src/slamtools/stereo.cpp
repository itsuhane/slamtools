#include <slamtools/stereo.h>
#include <array>
#include <slamtools/ransac.h>
#include <slamtools/homography.h>
#include <slamtools/essential.h>
#include <slamtools/lie_algebra.h>

using namespace Eigen;

Matrix3d find_essential_matrix(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, double threshold, double confidence, size_t max_iteration, int seed) {
    std::vector<char> _;
    return find_essential_matrix(points1, points2, _, threshold, confidence, max_iteration, seed);
}

Matrix3d find_homography_matrix(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, double threshold, double confidence, size_t max_iteration, int seed) {
    std::vector<char> _;
    return find_homography_matrix(points1, points2, _, threshold, confidence, max_iteration, seed);
}

Matrix3d find_essential_matrix(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, std::vector<char> &inlier_mask, double threshold, double confidence, size_t max_iteration, int seed) {
    struct EssentialSolver {
        std::vector<Matrix3d> operator()(const std::array<Vector2d, 5> &samples1, const std::array<Vector2d, 5> &samples2) const {
            return solve_essential_5pt(samples1, samples2);
        }
    };
    struct EssentialEvaluator {
        const Matrix3d &E;
        const Matrix3d Et;
        EssentialEvaluator(const Matrix3d &E) :
            E(E), Et(E.transpose()) {
        }
        double operator()(const Vector2d &p1, const Vector2d &p2) const {
            return essential_geometric_error(E, p1, p2) + essential_geometric_error(Et, p2, p1);
        }
    };
    static const double t1 = 3.84;
    Ransac<5, Matrix3d, EssentialSolver, EssentialEvaluator> ransac(2.0 * t1 * threshold * threshold, confidence, max_iteration, seed);
    return ransac.solve(inlier_mask, points1, points2);
}

Matrix3d find_homography_matrix(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, std::vector<char> &inlier_mask, double threshold, double confidence, size_t max_iteration, int seed) {
    struct HomographySolver {
        Matrix3d operator()(const std::array<Vector2d, 4> &samples1, const std::array<Vector2d, 4> &samples2) const {
            return solve_homography_4pt(samples1, samples2);
        }
    };
    struct HomographyEvaluator {
        const Matrix3d &H;
        const Matrix3d Hinv;
        HomographyEvaluator(const Matrix3d &H) :
            H(H), Hinv(H.inverse()) {
        }
        double operator()(const Vector2d &p1, const Vector2d &p2) const {
            return homography_geometric_error(H, p1, p2) + homography_geometric_error(Hinv, p2, p1);
        }
    };
    static const double t2 = 5.99;
    Ransac<4, Matrix3d, HomographySolver, HomographyEvaluator> ransac(2.0 * t2 * threshold * threshold, confidence, max_iteration, seed);
    return ransac.solve(inlier_mask, points1, points2);
}

size_t triangulate_from_rt(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, const Matrix3d &R, const Vector3d &T, std::vector<Vector3d> &result_points, std::vector<char> &result_status) {
    size_t count = 0;
    result_points.resize(points1.size());
    result_status.resize(points1.size());

    Matrix<double, 3, 4> P1, P2;
    P1.setIdentity();
    P2 << R, T;

    for (size_t i = 0; i < points1.size(); ++i) {
        if (triangulate_point_checked(P1, P2, points1[i], points2[i], result_points[i])) {
            result_status[i] = 1;
            count++;
        } else {
            result_status[i] = 0;
        }
    }
    return count;
}

size_t triangulate_from_rt(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, const std::vector<Matrix3d> &Rs, const std::vector<Vector3d> &Ts, std::vector<Vector3d> &result_points, Matrix3d &result_R, Vector3d &result_T, std::vector<char> &result_status) {
    std::vector<std::vector<Vector3d>> points(Rs.size());
    std::vector<std::vector<char>> status(Rs.size());
    std::vector<size_t> counts(Rs.size());

    size_t best_i = 0;
    for (size_t i = 0; i < Rs.size(); ++i) {
        counts[i] = triangulate_from_rt(points1, points2, Rs[i], Ts[i], points[i], status[i]);
        if (counts[i] > counts[best_i]) {
            best_i = i;
        }
    }

    result_R = Rs[best_i];
    result_T = Ts[best_i];
    result_points.swap(points[best_i]);
    result_status.swap(status[best_i]);

    return counts[best_i];
}

size_t triangulate_from_rt_scored(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, const Matrix3d &R, const Vector3d &T, std::vector<Vector3d> &result_points, std::vector<char> &result_status, double &score) {
    size_t count = 0;
    result_points.resize(points1.size());
    result_status.resize(points1.size());

    Matrix<double, 3, 4> P1, P2;
    P1.setIdentity();
    P2 << R, T;

    score = 0;

    for (size_t i = 0; i < points1.size(); ++i) {
        double current_score;
        if (triangulate_point_scored(P1, P2, points1[i], points2[i], result_points[i], current_score)) {
            result_status[i] = 1;
            score += current_score;
            count++;
        } else {
            result_status[i] = 0;
        }
    }

    score /= (double)std::max(count, size_t(1));
    return count;
}

size_t triangulate_from_rt_scored(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, const std::vector<Matrix3d> &Rs, const std::vector<Vector3d> &Ts, size_t count_threshold, std::vector<Vector3d> &result_points, Matrix3d &result_R, Vector3d &result_T, std::vector<char> &result_status) {
    std::vector<std::vector<Vector3d>> points(Rs.size());
    std::vector<std::vector<char>> status(Rs.size());
    std::vector<size_t> counts(Rs.size());
    std::vector<double> scores(Rs.size());

    size_t best_i = 0;
    for (size_t i = 0; i < Rs.size(); ++i) {
        counts[i] = triangulate_from_rt_scored(points1, points2, Rs[i], Ts[i], points[i], status[i], scores[i]);
        if (counts[i] > count_threshold && scores[i] < scores[best_i]) {
            best_i = i;
        } else if (counts[i] > counts[best_i]) {
            best_i = i;
        }
    }

    result_R = Rs[best_i];
    result_T = Ts[best_i];
    result_points.swap(points[best_i]);
    result_status.swap(status[best_i]);

    return counts[best_i];
}

size_t triangulate_from_essential(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, const Matrix3d &E, std::vector<Vector3d> &result_points, std::vector<char> &result_status, Matrix3d &R, Vector3d &T) {
    std::array<Matrix3d, 4> Rs;
    std::array<Vector3d, 4> Ts;
    std::array<std::vector<Vector3d>, 4> results;
    std::array<std::vector<char>, 4> stats;
    std::array<size_t, 4> counts;

    decompose_essential(E, Rs[0], Rs[2], Ts[0]);
    Rs[1] = Rs[0];
    Rs[3] = Rs[2];
    Ts[2] = Ts[0];
    Ts[1] = Ts[3] = -Ts[0];

    size_t best_i = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        counts[i] = triangulate_from_rt(points1, points2, Rs[i], Ts[i], results[i], stats[i]);
        if (counts[i] > counts[best_i]) {
            best_i = i;
        }
    }

    R = Rs[best_i];
    T = Ts[best_i];
    result_points.swap(results[best_i]);
    result_status.swap(stats[best_i]);

    return counts[best_i];
}

size_t triangulate_from_homography(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2, const Matrix3d &H, std::vector<Vector3d> &result_points, std::vector<char> &result_status, Matrix3d &R, Vector3d &T) {
    std::array<Matrix3d, 4> Rs;
    std::array<Vector3d, 4> Ts;
    std::array<std::vector<Vector3d>, 4> results;
    std::array<std::vector<char>, 4> stats;
    std::array<size_t, 4> counts;

    Vector3d n1, n2;
    decompose_homography(H, Rs[0], Rs[2], Ts[0], Ts[2], n1, n2);

    Rs[1] = Rs[0];
    Rs[3] = Rs[2];
    Ts[1] = -Ts[0];
    Ts[3] = -Ts[2];

    size_t best_i = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        counts[i] = triangulate_from_rt(points1, points2, Rs[i], Ts[i], results[i], stats[i]);
        if (counts[i] > counts[best_i]) {
            best_i = i;
        }
    }

    R = Rs[best_i];
    T = Ts[best_i];
    result_points.swap(results[best_i]);
    result_status.swap(stats[best_i]);

    return counts[best_i];
}
