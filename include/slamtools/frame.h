#pragma once

#include <slamtools/common.h>
#include <slamtools/state.h>
#include <slamtools/preintegrator.h>

class Track;
class SlidingWindow;
class Factor;
class Configurator;

struct create_if_empty_t {};
extern create_if_empty_t create_if_empty;

class Frame {
    friend class Track;
    friend class SlidingWindow;
    SlidingWindow *map;

  public:
    Frame();
    virtual ~Frame();

    std::unique_ptr<Frame> clone() const;

    size_t keypoint_num() const {
        return keypoints.size();
    }

    const Eigen::Vector2d &get_keypoint(size_t keypoint_id) const {
        return keypoints[keypoint_id];
    }

    Track *get_track(size_t keypoint_id) const {
        return tracks[keypoint_id];
    }

    Track *get_track(size_t keypoint_id, const create_if_empty_t &);

    Factor *get_reprojection_factor(size_t keypoint_id) {
        return reprojection_factors[keypoint_id].get();
    }

    Factor *get_preintegration_factor() {
        return preintegration_factor.get();
    }

    void detect_keypoints(Configurator *config);
    void track_keypoints(Frame *next_frame, Configurator *config);
    void detect_segments(size_t max_segments = 0);

    PoseState get_pose(const ExtrinsicParams &sensor) const;
    void set_pose(const ExtrinsicParams &sensor, const PoseState &pose);

    Eigen::Matrix3d K;
    Eigen::Matrix2d sqrt_inv_cov;
    std::shared_ptr<Image> image;

    PoseState pose;
    MotionState motion;
    ExtrinsicParams camera;
    ExtrinsicParams imu;

    PreIntegrator preintegration;

    Eigen::Vector3d external_gravity;
    std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d>> segments;

  private:
    std::vector<Eigen::Vector2d> keypoints;
    std::vector<Track *> tracks;
    std::vector<std::unique_ptr<Factor>> reprojection_factors;
    std::unique_ptr<Factor> preintegration_factor;
};

template <>
struct Comparator<Frame> {
    constexpr bool operator()(const Frame &frame_i, const Frame &frame_j) const {
        return Comparator<Image *>()(frame_i.image.get(), frame_j.image.get());
    }
};
