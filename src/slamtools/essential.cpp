#include <slamtools/essential.h>

using namespace Eigen;

class Polynomial {
    public:
    // clang-format off
    enum GRevLexMonomials {
        XXX = 0, XXY = 1, XYY = 2, YYY = 3, XXZ = 4, XYZ = 5, YYZ = 6, XZZ = 7, YZZ = 8, ZZZ = 9,
        XX = 10, XY = 11, YY = 12, XZ = 13, YZ = 14, ZZ = 15, X = 16, Y = 17, Z = 18, I = 19
    };
    // clang-format on

    Matrix<double, 20, 1> v;

    Polynomial(const Matrix<double, 20, 1> &coeffcients) :
        v(coeffcients) {
    }

    public:
    Polynomial() :
        Polynomial(Matrix<double, 20, 1>::Zero()) {
    }

    Polynomial(double w) {
        v.setZero();
        v[I] = w;
    }

    void set_xyzw(double x, double y, double z, double w) {
        v.setZero();
        v[X] = x;
        v[Y] = y;
        v[Z] = z;
        v[I] = w;
    }

    Polynomial operator-() const {
        return Polynomial(-v);
    }

    Polynomial operator+(const Polynomial &b) const {
        return Polynomial(v + b.v);
    }

    Polynomial operator-(const Polynomial &b) const {
        return Polynomial(v - b.v);
    }

    Polynomial operator*(const Polynomial &b) const {
        Polynomial r;

        r.v[I] = v[I] * b.v[I];

        r.v[Z] = v[I] * b.v[Z] + v[Z] * b.v[I];
        r.v[Y] = v[I] * b.v[Y] + v[Y] * b.v[I];
        r.v[X] = v[I] * b.v[X] + v[X] * b.v[I];

        r.v[ZZ] = v[I] * b.v[ZZ] + v[Z] * b.v[Z] + v[ZZ] * b.v[I];
        r.v[YZ] = v[I] * b.v[YZ] + v[Z] * b.v[Y] + v[Y] * b.v[Z] + v[YZ] * b.v[I];
        r.v[XZ] = v[I] * b.v[XZ] + v[Z] * b.v[X] + v[X] * b.v[Z] + v[XZ] * b.v[I];
        r.v[YY] = v[I] * b.v[YY] + v[Y] * b.v[Y] + v[YY] * b.v[I];
        r.v[XY] = v[I] * b.v[XY] + v[Y] * b.v[X] + v[X] * b.v[Y] + v[XY] * b.v[I];
        r.v[XX] = v[I] * b.v[XX] + v[X] * b.v[X] + v[XX] * b.v[I];

        r.v[ZZZ] = v[I] * b.v[ZZZ] + v[Z] * b.v[ZZ] + v[ZZ] * b.v[Z] + v[ZZZ] * b.v[I];
        r.v[YZZ] = v[I] * b.v[YZZ] + v[Z] * b.v[YZ] + v[Y] * b.v[ZZ] + v[ZZ] * b.v[Y] + v[YZ] * b.v[Z] + v[YZZ] * b.v[I];
        r.v[XZZ] = v[I] * b.v[XZZ] + v[Z] * b.v[XZ] + v[X] * b.v[ZZ] + v[ZZ] * b.v[X] + v[XZ] * b.v[Z] + v[XZZ] * b.v[I];
        r.v[YYZ] = v[I] * b.v[YYZ] + v[Z] * b.v[YY] + v[Y] * b.v[YZ] + v[YZ] * b.v[Y] + v[YY] * b.v[Z] + v[YYZ] * b.v[I];
        r.v[XYZ] = v[I] * b.v[XYZ] + v[Z] * b.v[XY] + v[Y] * b.v[XZ] + v[X] * b.v[YZ] + v[YZ] * b.v[X] + v[XZ] * b.v[Y] + v[XY] * b.v[Z] + v[XYZ] * b.v[I];
        r.v[XXZ] = v[I] * b.v[XXZ] + v[Z] * b.v[XX] + v[X] * b.v[XZ] + v[XZ] * b.v[X] + v[XX] * b.v[Z] + v[XXZ] * b.v[I];
        r.v[YYY] = v[I] * b.v[YYY] + v[Y] * b.v[YY] + v[YY] * b.v[Y] + v[YYY] * b.v[I];
        r.v[XYY] = v[I] * b.v[XYY] + v[Y] * b.v[XY] + v[X] * b.v[YY] + v[YY] * b.v[X] + v[XY] * b.v[Y] + v[XYY] * b.v[I];
        r.v[XXY] = v[I] * b.v[XXY] + v[Y] * b.v[XX] + v[X] * b.v[XY] + v[XY] * b.v[X] + v[XX] * b.v[Y] + v[XXY] * b.v[I];
        r.v[XXX] = v[I] * b.v[XXX] + v[X] * b.v[XX] + v[XX] * b.v[X] + v[XXX] * b.v[I];

        return r;
    }

    const Matrix<double, 20, 1> &coeffcients() const {
        return v;
    }
};

Polynomial operator*(const double &scale, const Polynomial &poly) {
    return Polynomial(scale * poly.coeffcients());
}

inline Matrix3d to_matrix(const Matrix<double, 9, 1> &vec) {
    return (Matrix3d() << vec.segment<3>(0), vec.segment<3>(3), vec.segment<3>(6)).finished();
}

inline Matrix<double, 9, 4> generate_nullspace_basis(const std::array<Vector2d, 5> &points1, const std::array<Vector2d, 5> &points2) {
    Matrix<double, 5, 9> A;
    for (size_t i = 0; i < 5; ++i) {
        Matrix3d h = Vector3d(points1[i].homogeneous()) * points2[i].homogeneous().transpose();
        for (size_t j = 0; j < 3; ++j) {
            A.block<1, 3>(i, j * 3) = h.row(j);
        }
    }
    return A.jacobiSvd(ComputeFullV).matrixV().block<9, 4>(0, 5);
}

inline Matrix<double, 10, 20> generate_polynomials(const Matrix<double, 9, 4> &basis) {
    typedef Matrix<Polynomial, 3, 3> matrix_poly;
    Matrix3d Ex = to_matrix(basis.col(0));
    Matrix3d Ey = to_matrix(basis.col(1));
    Matrix3d Ez = to_matrix(basis.col(2));
    Matrix3d Ew = to_matrix(basis.col(3));

    matrix_poly Epoly;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            Epoly(i, j).set_xyzw(Ex(i, j), Ey(i, j), Ez(i, j), Ew(i, j));
        }
    }

    Matrix<double, 10, 20> polynomials;

    matrix_poly EEt = Epoly * Epoly.transpose();
    matrix_poly singular_value_constraints = (EEt * Epoly) - (0.5 * EEt.trace()) * Epoly;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            polynomials.row(i * 3 + j) = singular_value_constraints(i, j).coeffcients();
        }
    }

    Polynomial detE = Epoly.determinant();
    polynomials.row(9) = detE.coeffcients();

    return polynomials;
}

inline Matrix<double, 10, 10> generate_action_matrix(Matrix<double, 10, 20> &polynomials) {
    std::array<size_t, 10> perm;
    for (size_t i = 0; i < 10; ++i) {
        perm[i] = i;
    }
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = i + 1; j < 10; ++j) {
            if (abs(polynomials(perm[i], i)) < abs(polynomials(perm[j], i))) {
                std::swap(perm[i], perm[j]);
            }
        }
        if (polynomials(perm[i], i) == 0) continue;
        polynomials.row(perm[i]) /= polynomials(perm[i], i);
        for (size_t j = i + 1; j < 10; ++j) {
            polynomials.row(perm[j]) -= polynomials.row(perm[i]) * polynomials(perm[j], i);
        }
    }
    for (size_t i = 9; i > 0; --i) {
        for (size_t j = 0; j < i; ++j) {
            polynomials.row(perm[j]) -= polynomials.row(perm[i]) * polynomials(perm[j], i);
        }
    }

    Matrix<double, 10, 10> action;
    action.row(0) = -polynomials.block<1, 10>(perm[Polynomial::XXX], Polynomial::XX);
    action.row(1) = -polynomials.block<1, 10>(perm[Polynomial::XXY], Polynomial::XX);
    action.row(2) = -polynomials.block<1, 10>(perm[Polynomial::XYY], Polynomial::XX);
    action.row(3) = -polynomials.block<1, 10>(perm[Polynomial::XXZ], Polynomial::XX);
    action.row(4) = -polynomials.block<1, 10>(perm[Polynomial::XYZ], Polynomial::XX);
    action.row(5) = -polynomials.block<1, 10>(perm[Polynomial::XZZ], Polynomial::XX);
    action.row(6) = Matrix<double, 10, 1>::Unit(Polynomial::XX - Polynomial::XX).transpose();
    action.row(7) = Matrix<double, 10, 1>::Unit(Polynomial::XY - Polynomial::XX).transpose();
    action.row(8) = Matrix<double, 10, 1>::Unit(Polynomial::XZ - Polynomial::XX).transpose();
    action.row(9) = Matrix<double, 10, 1>::Unit(Polynomial::X - Polynomial::XX).transpose();

    return action;
}

inline std::vector<Vector3d> solve_grobner_system(const Matrix<double, 10, 10> &action) {
    EigenSolver<Matrix<double, 10, 10>> eigen(action, true);
    Matrix<std::complex<double>, 10, 1> xs = eigen.eigenvalues();

    std::vector<Vector3d> results;
    for (size_t i = 0; i < 10; ++i) {
        if (abs(xs[i].imag()) < 1.0e-10) {
            Matrix<double, 10, 1> h = eigen.eigenvectors().col(i).real();
            double xw = h(Polynomial::X - Polynomial::XX);
            double yw = h(Polynomial::Y - Polynomial::XX);
            double zw = h(Polynomial::Z - Polynomial::XX);
            double w = h(Polynomial::I - Polynomial::XX);
            results.emplace_back(xw / w, yw / w, zw / w);
        }
    }
    return results;
}

void decompose_essential(const Matrix3d &E, Matrix3d &R1, Matrix3d &R2, Vector3d &T) {
#ifdef ESSENTIAL_DECOMPOSE_HORN
    Matrix3d EET = E * E.transpose();
    double halfTrace = 0.5 * EET.trace();
    Vector3d b;

    Vector3d e0e1 = E.col(0).cross(E.col(1));
    Vector3d e1e2 = E.col(1).cross(E.col(2));
    Vector3d e2e0 = E.col(2).cross(E.col(0));

#    if ESSENTIAL_DECOMPOSE_HORN == 1
    if (e0e1.norm() > e1e2.norm() && e0e1.norm() > e2e0.norm()) {
        b = e0e1.normalized() * sqrt(halfTrace);
    } else if (e1e2.norm() > e0e1.norm() && e1e2.norm() > e2e0.norm()) {
        b = e1e2.normalized() * sqrt(halfTrace);
    } else {
        b = e2e0.normalized() * sqrt(halfTrace);
    }
#    else
    Matrix3d bbT = halfTrace * Matrix3d::Identity() - EET;
    Vector3d bbT_diag = bbT.diagonal();
    if (bbT_diag(0) > bbt_diag(1) && bbT_diag(0) > bbT_diag(2)) {
        b = bbT.row(0) / sqrt(bbT_diag(0));
    } else if (bbT_diag(1) > bbT_diag(0) && bbT_diag(1) > bbT_diag(2)) {
        b = bbT.row(1) / sqrt(bbT_diag(1));
    } else {
        b = bbT.row(2) / sqrt(bbT_diag(2));
    }
#    endif

    Matrix3d cofactorsT;
    cofactorsT.col(0) = e1e2;
    cofactorsT.col(1) = e2e0;
    cofactorsT.col(2) = e0e1;

    Matrix3d skew_b;
    skew_b << 0, -b.z(), b.y(),
        b.z(), 0, -b.x(),
        -b.y(), b.x(), 0;
    Matrix3d bxE = skew_b * E;

    double bTb = b.dot(b);
    R1 = (cofactorsT - bxE) / bTb;
    R2 = (cofactorsT + bxE) / bTb;
    T = b;
#else
    JacobiSVD<Matrix3d> svd(E, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d VT = svd.matrixV().transpose();
    if (U.determinant() < 0) {
        U = -U;
    }
    if (VT.determinant() < 0) {
        VT = -VT;
    }
    Matrix3d W;
    W << 0, 1, 0,
        -1, 0, 0,
        0, 0, 1;
    R1 = U * W * VT;
    R2 = U * W.transpose() * VT;
    T = U.col(2);
#endif
}

std::vector<Matrix3d> solve_essential_5pt(const std::array<Vector2d, 5> &points1, const std::array<Vector2d, 5> &points2) {
    Matrix<double, 9, 4> basis = generate_nullspace_basis(points1, points2);
    Matrix<double, 10, 20> polynomials = generate_polynomials(basis);
    Matrix<double, 10, 10> action = generate_action_matrix(polynomials);
    std::vector<Vector3d> solutions = solve_grobner_system(action);
    std::vector<Matrix3d> results(solutions.size());
    for (size_t i = 0; i < solutions.size(); ++i) {
        results[i] = to_matrix(basis * solutions[i].homogeneous());
    }
    return results;
}
