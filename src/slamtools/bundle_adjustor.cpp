#include <slamtools/bundle_adjustor.h>
#include <ceres/ceres.h>
#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/sliding_window.h>
#include <slamtools/stereo.h>
#include <slamtools/factor.h>
#include <slamtools/ceres/lie_algebra_eigen_quaternion_parameterization.h>
#include <slamtools/ceres/reprojection_error_cost.h>
#include <slamtools/ceres/preintegration_error_cost.h>

using namespace Eigen;
using namespace ceres;

struct BundleAdjustor::BundleAdjustorSolver {
    BundleAdjustorSolver() {
        cauchy_loss = std::make_unique<CauchyLoss>(1.0);
        quaternion_parameterization = std::make_unique<LieAlgebraEigenQuaternionParamatrization>();
    }
    std::unique_ptr<LossFunction> cauchy_loss;
    std::unique_ptr<LocalParameterization> quaternion_parameterization;
};

BundleAdjustor::BundleAdjustor() {
    solver = std::make_unique<BundleAdjustorSolver>();
}

BundleAdjustor::~BundleAdjustor() = default;

bool BundleAdjustor::solve(SlidingWindow *map, bool use_inertial, size_t max_iter, const double &max_time) {
    Problem::Options problem_options;
    problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
    problem_options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
    problem_options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;

    Problem problem(problem_options);

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        problem.AddParameterBlock(frame->pose.q.coeffs().data(), 4, solver->quaternion_parameterization.get());
        problem.AddParameterBlock(frame->pose.p.data(), 3);
        if (frame->pose.flag(PF_FIXED)) {
            problem.SetParameterBlockConstant(frame->pose.q.coeffs().data());
            problem.SetParameterBlockConstant(frame->pose.p.data());
        }
        if (use_inertial) {
            problem.AddParameterBlock(frame->motion.v.data(), 3);
            problem.AddParameterBlock(frame->motion.bg.data(), 3);
            problem.AddParameterBlock(frame->motion.ba.data(), 3);
        }
    }

    std::unordered_set<Track *> visited_tracks;
    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        for (size_t j = 0; j < frame->keypoint_num(); ++j) {
            Track *track = frame->get_track(j);
            if (!track) continue;
            if (!track->landmark.flag(LF_VALID)) continue;
            if (visited_tracks.count(track) > 0) continue;
            visited_tracks.insert(track);
            problem.AddParameterBlock(track->landmark.x.data(), 3);
        }
    }

    for (size_t i = 0; i < map->frame_num(); ++i) {
        Frame *frame = map->get_frame(i);
        for (size_t j = 0; j < frame->keypoint_num(); ++j) {
            Track *track = frame->get_track(j);
            if (!track) continue;
            if (!track->landmark.flag(LF_VALID)) continue;
            problem.AddResidualBlock(frame->get_reprojection_factor(j)->get_cost_function<ReprojectionErrorCost>(),
                                     solver->cauchy_loss.get(),
                                     frame->pose.q.coeffs().data(),
                                     frame->pose.p.data(),
                                     track->landmark.x.data());
        }
    }

    if (use_inertial) {
        for (size_t j = 1; j < map->frame_num(); ++j) {
            Frame *frame_i = map->get_frame(j - 1);
            Frame *frame_j = map->get_frame(j);
            if (frame_j->preintegration.integrate(frame_j->image->t, frame_i->motion.bg, frame_i->motion.ba, true, true)) {
                problem.AddResidualBlock(frame_j->get_preintegration_factor()->get_cost_function<PreIntegrationErrorCost>(),
                                         nullptr,
                                         frame_i->pose.q.coeffs().data(),
                                         frame_i->pose.p.data(),
                                         frame_i->motion.v.data(),
                                         frame_i->motion.bg.data(),
                                         frame_i->motion.ba.data(),
                                         frame_j->pose.q.coeffs().data(),
                                         frame_j->pose.p.data(),
                                         frame_j->motion.v.data(),
                                         frame_j->motion.bg.data(),
                                         frame_j->motion.ba.data());
            }
        }
    }

    Solver::Options solver_options;
    solver_options.linear_solver_type = SPARSE_SCHUR;
    solver_options.trust_region_strategy_type = DOGLEG;
    solver_options.use_explicit_schur_complement = true;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.logging_type = SILENT;
    solver_options.max_num_iterations = (int)max_iter;
    solver_options.max_solver_time_in_seconds = max_time;
    solver_options.num_threads = 1;
    solver_options.num_linear_solver_threads = 1;

    Solver::Summary solver_summary;
    ceres::Solve(solver_options, &problem, &solver_summary);

    for (size_t i = 0; i < map->track_num(); ++i) {
        Track *track = map->get_track(i);
        if (!track->landmark.flag(LF_VALID)) continue;
        const Vector3d &x = track->landmark.x;
        double quality = 0.0;
        double quality_num = 0.0;
        for (const auto &k : track->keypoint_map()) {
            Frame *frame = k.first;
            size_t keypoint_id = k.second;
            PoseState pose = frame->get_pose(frame->camera);
            Vector3d y = pose.q.conjugate() * (x - pose.p);
            if (y.z() <= 1.0e-3 || y.z() > 50) {
                track->landmark.flag(LF_VALID) = false;
                break;
            }
            quality += (apply_k(y.hnormalized(), frame->K) - apply_k(frame->get_keypoint(keypoint_id), frame->K)).norm();
            quality_num += 1.0;
        }
        if (!track->landmark.flag(LF_VALID)) continue;
        track->landmark.quality = quality / std::max(quality_num, 1.0);
    }

    return solver_summary.IsSolutionUsable();
}

struct LandmarkInfo {
    LandmarkInfo() {
        mat.setZero();
        vec.setZero();
    }
    Matrix3d mat;
    Vector3d vec;
    std::unordered_map<size_t, Matrix<double, 3, 6>> h;
};
