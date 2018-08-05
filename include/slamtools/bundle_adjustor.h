#pragma once

#include <slamtools/common.h>

class SlidingWindow;
class Frame;

class BundleAdjustor {
    struct BundleAdjustorSolver; // pimpl

  public:
    BundleAdjustor();
    virtual ~BundleAdjustor();

    bool solve(SlidingWindow *map, bool use_inertial = true, size_t max_iter = 50, const double &max_time = 1.0e6);

  private:
    std::unique_ptr<BundleAdjustorSolver> solver;
};
