#include <slamtools/factor.h>
#include <slamtools/frame.h>
#include <slamtools/ceres/reprojection_error_cost.h>
#include <slamtools/ceres/preintegration_error_cost.h>

std::unique_ptr<Factor> Factor::create_reprojection_error(Frame *frame, size_t keypoint_id) {
    return std::make_unique<Factor>(
        std::make_unique<ReprojectionErrorCost>(frame, keypoint_id),
        factor_construct_t());
}

std::unique_ptr<Factor> Factor::create_preintegration_error(Frame *frame_i, Frame *frame_j) {
    return std::make_unique<Factor>(
        std::make_unique<PreIntegrationErrorCost>(frame_i, frame_j),
        factor_construct_t());
}

Factor::Factor(std::unique_ptr<FactorCostFunction> cost_function, const factor_construct_t &) :
    cost_function(std::move(cost_function)) {
}

Factor::~Factor() = default;
