#pragma once

#include <slamtools/common.h>

class Frame;
class SlidingWindow;

void visual_inertial_pnp(SlidingWindow *map, Frame *frame, bool use_inertial = true, size_t max_iter = 50, const double &max_time = 1.0e6);
