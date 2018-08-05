#pragma once

#include <slamtools/common.h>

class Frame;

class Factor {
    Factor() = delete;
    struct factor_construct_t {};

  public:
    struct FactorCostFunction {
        virtual ~FactorCostFunction() = default;
    };

    static std::unique_ptr<Factor> create_reprojection_error(Frame *frame, size_t keypoint_id);
    static std::unique_ptr<Factor> create_preintegration_error(Frame *frame_i, Frame *frame_j);

    template <typename T>
    T *get_cost_function() {
        return static_cast<T *>(cost_function.get());
    }

    Factor(std::unique_ptr<FactorCostFunction> cost_function, const factor_construct_t &);
    virtual ~Factor();

  private:
    std::unique_ptr<FactorCostFunction> cost_function;
};
