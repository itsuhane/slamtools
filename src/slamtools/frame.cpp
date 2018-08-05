#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/sliding_window.h>
#include <slamtools/factor.h>
#include <slamtools/stereo.h>
#include <slamtools/configurator.h>

using namespace Eigen;

create_if_empty_t create_if_empty{};

Frame::Frame() :
    map(nullptr) {
}

Frame::~Frame() = default;

std::unique_ptr<Frame> Frame::clone() const {
    std::unique_ptr<Frame> frame = std::make_unique<Frame>();
    frame->K = K;
    frame->sqrt_inv_cov = sqrt_inv_cov;
    frame->image = image;
    frame->pose = pose;
    frame->motion = motion;
    frame->camera = camera;
    frame->imu = imu;
    frame->preintegration = preintegration;
    frame->external_gravity = external_gravity; // maybe inappropriate?
    frame->keypoints = keypoints;
    frame->segments = segments;
    frame->tracks = std::vector<Track *>(keypoints.size(), nullptr);
    frame->reprojection_factors = std::vector<std::unique_ptr<Factor>>(keypoints.size());
    frame->map = nullptr;
    return frame;
}

Track *Frame::get_track(size_t keypoint_id, const create_if_empty_t &) {
    if (tracks[keypoint_id] == nullptr) {
        assert(("for get_track(..., create_if_empty) to work, frame->map cannot be nullptr") && (map != nullptr));
        Track *track = map->create_track();
        track->add_keypoint(this, keypoint_id);
    }
    return tracks[keypoint_id];
}

void Frame::detect_keypoints(Configurator *config) {
    std::vector<Vector2d> pkeypoints(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        pkeypoints[i] = apply_k(keypoints[i], K);
    }
    image->detect_keypoints(pkeypoints, config->max_keypoint_detection());
    size_t old_keypoint_num = keypoints.size();
    keypoints.resize(pkeypoints.size());
    tracks.resize(pkeypoints.size(), nullptr);
    reprojection_factors.resize(pkeypoints.size());
    for (size_t i = old_keypoint_num; i < pkeypoints.size(); ++i) {
        keypoints[i] = remove_k(pkeypoints[i], K);
    }
}

void Frame::track_keypoints(Frame *next_frame, Configurator *config) {
    std::vector<Vector2d> curr_keypoints(keypoints.size());
    std::vector<Vector2d> next_keypoints;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        curr_keypoints[i] = apply_k(keypoints[i], K);
    }

    if (config->predict_keypoints()) {
        Quaterniond delta_key_q = (camera.q_cs.conjugate() * imu.q_cs * next_frame->preintegration.delta.q * next_frame->imu.q_cs.conjugate() * next_frame->camera.q_cs).conjugate();
        next_keypoints.resize(curr_keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            next_keypoints[i] = apply_k((delta_key_q * keypoints[i].homogeneous()).hnormalized(), next_frame->K);
        }
    }

    std::vector<char> status;
    image->track_keypoints(next_frame->image.get(), curr_keypoints, next_keypoints, status);

    for (size_t curr_keypoint_id = 0; curr_keypoint_id < curr_keypoints.size(); ++curr_keypoint_id) {
        if (status[curr_keypoint_id]) {
            size_t next_keypoint_id = next_frame->keypoints.size();
            next_frame->keypoints.emplace_back(remove_k(next_keypoints[curr_keypoint_id], next_frame->K));
            next_frame->tracks.emplace_back(nullptr);
            next_frame->reprojection_factors.emplace_back(nullptr);
            get_track(curr_keypoint_id, create_if_empty)->add_keypoint(next_frame, next_keypoint_id);
        }
    }
}

void Frame::detect_segments(size_t max_segments) {
    image->detect_segments(segments, max_segments);
    for (size_t i = 0; i < segments.size(); ++i) {
        std::get<0>(segments[i]) = remove_k(std::get<0>(segments[i]), K);
        std::get<1>(segments[i]) = remove_k(std::get<1>(segments[i]), K);
    }
}

PoseState Frame::get_pose(const ExtrinsicParams &sensor) const {
    PoseState result;
    result.q = pose.q * sensor.q_cs;
    result.p = pose.p + pose.q * sensor.p_cs;
    return result;
}

void Frame::set_pose(const ExtrinsicParams &sensor, const PoseState &pose) {
    this->pose.q = pose.q * sensor.q_cs.conjugate();
    this->pose.p = pose.p - this->pose.q * sensor.p_cs;
}
