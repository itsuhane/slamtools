#include <slamtools/homography.h>

using namespace Eigen;

bool decompose_homography(const Matrix3d &H, Matrix3d &R1, Matrix3d &R2, Vector3d &T1, Vector3d &T2, Vector3d &n1, Vector3d &n2) {
    Matrix3d Hn = H / H.jacobiSvd().singularValues()(1);
    Matrix3d S = Hn.transpose() * Hn - Matrix3d::Identity();

    bool is_pure_rotation = true;
    for (int i = 0; i < 3 && is_pure_rotation; ++i) {
        for (int j = 0; j < 3 && is_pure_rotation; ++j) {
            if (abs(S(i, j)) > 1e-3) {
                is_pure_rotation = false;
            }
        }
    }

    if (is_pure_rotation) {
        // Pure rotation
        JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
        R1 = svd.matrixU() * Matrix3d::Identity() * svd.matrixV().transpose();
        if (R1.determinant() < 0) {
            R1 = -R1;
        }
        R2 = R1;
        //R1 = R2 = Hn;
        T1 = T2 = Vector3d::Zero();
        n1 = n2 = Vector3d::Zero();
    } else {
        double Ms00 = S(1, 2) * S(1, 2) - S(1, 1) * S(2, 2);
        double Ms11 = S(0, 2) * S(0, 2) - S(0, 0) * S(2, 2);
        double Ms22 = S(0, 1) * S(0, 1) - S(0, 0) * S(1, 1);

        double sqrtMs00 = sqrt(Ms00);
        double sqrtMs11 = sqrt(Ms11);
        double sqrtMs22 = sqrt(Ms22);

        double nu = 2.0 * sqrt(1 + S.trace() - Ms00 - Ms11 - Ms22);
        double tenormsq = 2 + S.trace() - nu;

        Vector3d tstar1, tstar2;

        if (S(0, 0) > S(1, 1) && S(0, 0) > S(2, 2)) {
            double epslMs12 = (((S(0, 1) * S(0, 2) - S(0, 0) * S(1, 2)) < 0) ? -1 : 1);
            n1 << S(0, 0), S(0, 1) + sqrtMs22, S(0, 2) + epslMs12 * sqrtMs11;
            n2 << S(0, 0), S(0, 1) - sqrtMs22, S(0, 2) - epslMs12 * sqrtMs11;
            tstar1 = n1.norm() * n2 / S(0, 0);
            tstar2 = n2.norm() * n1 / S(0, 0);
        } else if (S(1, 1) > S(0, 0) && S(1, 1) > S(2, 2)) {
            double epslMs02 = (((S(1, 1) * S(0, 2) - S(0, 1) * S(1, 2)) < 0) ? -1 : 1);
            n1 << S(0, 1) + sqrtMs22, S(1, 1), S(1, 2) - epslMs02 * sqrtMs00;
            n2 << S(0, 1) - sqrtMs22, S(1, 1), S(1, 2) + epslMs02 * sqrtMs00;
            tstar2 = n2.norm() * n1 / S(1, 1);
            tstar1 = n1.norm() * n2 / S(1, 1);
        } else {
            double epslMs01 = (((S(1, 2) * S(0, 2) - S(0, 1) * S(2, 2)) < 0) ? -1 : 1);
            n1 << S(0, 2) + epslMs01 * sqrtMs11, S(1, 2) + sqrtMs00, S(2, 2);
            n2 << S(0, 2) - epslMs01 * sqrtMs11, S(1, 2) - sqrtMs00, S(2, 2);
            tstar1 = n1.norm() * n2 / S(2, 2);
            tstar2 = n2.norm() * n1 / S(2, 2);
        }
        n1.normalize();
        n2.normalize();
        tstar1 -= tenormsq * n1;
        tstar2 -= tenormsq * n2;
        R1 = Hn * (Matrix3d::Identity() - (tstar1 / nu) * n1.transpose());
        R2 = Hn * (Matrix3d::Identity() - (tstar2 / nu) * n2.transpose());
        tstar1 *= 0.5;
        tstar2 *= 0.5;
        T1 = R1 * tstar1;
        T2 = R2 * tstar2;
    }
    return !is_pure_rotation;
}

inline Matrix3d to_matrix(const Matrix<double, 9, 1> &vec) {
    return (Matrix3d() << vec.segment<3>(0), vec.segment<3>(3), vec.segment<3>(6)).finished();
}

inline Matrix3d solve_homography_normalized(const std::array<Vector2d, 4> &pa, const std::array<Vector2d, 4> &pb) {
    Matrix<double, 8, 9> A = Matrix<double, 8, 9>::Zero();

    for (size_t i = 0; i < 4; ++i) {
        const Vector2d &a = pa[i];
        const Vector2d &b = pb[i];
        A(i * 2, 1) = -a(0);
        A(i * 2, 2) = a(0) * b(1);
        A(i * 2, 4) = -a(1);
        A(i * 2, 5) = a(1) * b(1);
        A(i * 2, 7) = -1;
        A(i * 2, 8) = b(1);
        A(i * 2 + 1, 0) = a(0);
        A(i * 2 + 1, 2) = -a(0) * b(0);
        A(i * 2 + 1, 3) = a(1);
        A(i * 2 + 1, 5) = -a(1) * b(0);
        A(i * 2 + 1, 6) = 1;
        A(i * 2 + 1, 8) = -b(0);
    }

    Matrix<double, 9, 1> h = A.jacobiSvd(ComputeFullV).matrixV().col(8);
    return to_matrix(h);
}

inline Matrix3d solve_homography_normalized(const std::vector<Vector2d> &pa, const std::vector<Vector2d> &pb) {
    Matrix<double, Dynamic, 9> A;
    A.resize(pa.size() * 2, 9);
    A.setZero();

    for (size_t i = 0; i < pa.size(); ++i) {
        const Vector2d &a = pa[i];
        const Vector2d &b = pb[i];
        A(i * 2, 1) = -a(0);
        A(i * 2, 2) = a(0) * b(1);
        A(i * 2, 4) = -a(1);
        A(i * 2, 5) = a(1) * b(1);
        A(i * 2, 7) = -1;
        A(i * 2, 8) = b(1);
        A(i * 2 + 1, 0) = a(0);
        A(i * 2 + 1, 2) = -a(0) * b(0);
        A(i * 2 + 1, 3) = a(1);
        A(i * 2 + 1, 5) = -a(1) * b(0);
        A(i * 2 + 1, 6) = 1;
        A(i * 2 + 1, 8) = -b(0);
    }

    Matrix<double, 9, 1> h = A.jacobiSvd(ComputeFullV).matrixV().col(8);
    return to_matrix(h);
}

Matrix3d solve_homography_4pt(const std::array<Vector2d, 4> &points1, const std::array<Vector2d, 4> &points2) {
    static const double sqrt2 = sqrt(2.0);

    Vector2d pa_mean = Vector2d::Zero();
    Vector2d pb_mean = Vector2d::Zero();
    for (size_t i = 0; i < 4; ++i) {
        pa_mean += points1[i];
        pb_mean += points2[i];
    }
    pa_mean /= 4;
    pb_mean /= 4;

    double sa = 0;
    double sb = 0;

    for (size_t i = 0; i < 4; ++i) {
        sa += (points1[i] - pa_mean).norm();
        sb += (points2[i] - pb_mean).norm();
    }

    sa = 1.0 / (sqrt2 * sa);
    sb = 1.0 / (sqrt2 * sb);

    std::array<Vector2d, 4> na;
    std::array<Vector2d, 4> nb;
    for (size_t i = 0; i < 4; ++i) {
        na[i] = (points1[i] - pa_mean) * sa;
        nb[i] = (points2[i] - pb_mean) * sb;
    }

    Matrix3d NH = solve_homography_normalized(na, nb);

    Matrix3d Na, Nb;
    Nb << 1 / sb, 0, pb_mean(0),
        0, 1 / sb, pb_mean(1),
        0, 0, 1;
    Na << sa, 0, -sa * pa_mean(0),
        0, sa, -sa * pa_mean(1),
        0, 0, 1;

    return Nb * NH * Na;
}

Matrix3d solve_homography(const std::vector<Vector2d> &points1, const std::vector<Vector2d> &points2) {
    static const double sqrt2 = sqrt(2.0);

    Vector2d pa_mean = Vector2d::Zero();
    Vector2d pb_mean = Vector2d::Zero();
    for (size_t i = 0; i < points1.size(); ++i) {
        pa_mean += points1[i];
        pb_mean += points2[i];
    }
    pa_mean /= 4;
    pb_mean /= 4;

    double sa = 0;
    double sb = 0;

    for (size_t i = 0; i < points1.size(); ++i) {
        sa += (points1[i] - pa_mean).norm();
        sb += (points2[i] - pb_mean).norm();
    }

    sa = 1.0 / (sqrt2 * sa);
    sb = 1.0 / (sqrt2 * sb);

    std::vector<Vector2d> na(points1.size());
    std::vector<Vector2d> nb(points1.size());
    for (size_t i = 0; i < points1.size(); ++i) {
        na[i] = (points1[i] - pa_mean) * sa;
        nb[i] = (points2[i] - pb_mean) * sb;
    }

    Matrix3d NH = solve_homography_normalized(na, nb);

    Matrix3d Na, Nb;
    Nb << 1 / sb, 0, pb_mean(0),
        0, 1 / sb, pb_mean(1),
        0, 0, 1;
    Na << sa, 0, -sa * pa_mean(0),
        0, sa, -sa * pa_mean(1),
        0, 0, 1;

    return Nb * NH * Na;
}
