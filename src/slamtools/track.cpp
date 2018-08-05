#include <slamtools/track.h>
#include <slamtools/frame.h>
#include <slamtools/sliding_window.h>
#include <slamtools/factor.h>
#include <slamtools/stereo.h>

using namespace Eigen;

Track::Track() = default;
Track::~Track() = default;

const Vector2d &Track::get_keypoint(Frame *frame) const {
    return frame->get_keypoint(keypoint_refs.at(frame));
}

void Track::add_keypoint(Frame *frame, size_t keypoint_id) {
    frame->tracks[keypoint_id] = this;
    frame->reprojection_factors[keypoint_id] = Factor::create_reprojection_error(frame, keypoint_id);
    keypoint_refs[frame] = keypoint_id;
}

void Track::remove_keypoint(Frame *frame, bool suicide_if_empty) {
    size_t keypoint_id = keypoint_refs.at(frame);
    frame->tracks[keypoint_id] = nullptr;
    frame->reprojection_factors[keypoint_id].reset();
    keypoint_refs.erase(frame);
    if (suicide_if_empty && keypoint_refs.size() == 0) {
        map->recycle_track(this);
    }
}

bool Track::triangulate() {
    std::vector<Matrix<double, 3, 4>> Ps;
    std::vector<Vector2d> ps;
    for (const auto &t : keypoint_map()) {
        Matrix<double, 3, 4> P;
        Matrix3d R;
        Vector3d T;
        PoseState pose = t.first->get_pose(t.first->camera);
        R = pose.q.conjugate().matrix();
        T = -(R * pose.p);
        P << R, T;
        Ps.push_back(P);
        ps.push_back(t.first->get_keypoint(t.second));
    }
    Vector3d p;
    if (triangulate_point_checked(Ps, ps, p)) {
        landmark.x = p;
        landmark.flag(LF_VALID) = true;
    } else {
        landmark.flag(LF_VALID) = false;
    }
    return landmark.flag(LF_VALID);
}
