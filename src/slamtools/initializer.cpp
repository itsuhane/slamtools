#include <slamtools/initializer.h>
#include <slamtools/stereo.h>
#include <slamtools/lie_algebra.h>
#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/sliding_window.h>
#include <slamtools/pnp.h>
#include <slamtools/bundle_adjustor.h>
#include <slamtools/configurator.h>
#include <slamtools/homography.h>
#include <slamtools/essential.h>

using namespace Eigen;

Initializer::Initializer(std::shared_ptr<Configurator> config) :
    config(config) {
    raw = std::make_unique<SlidingWindow>();
}

Initializer::~Initializer() = default;

void Initializer::append_frame(std::unique_ptr<Frame> frame) {
    if (raw->frame_num() > 0) {
        Frame *last_frame = raw->get_frame(raw->frame_num() - 1);
        frame->preintegration.integrate(frame->image->t, last_frame->motion.bg, last_frame->motion.ba, true, false);
        last_frame->track_keypoints(frame.get(), config.get());
    }
    frame->detect_keypoints(config.get());

    raw->put_frame(std::move(frame));
    while (raw->frame_num() > config->max_init_raw_frames()) {
        raw->erase_frame(0);
    }
}


std::unique_ptr<SlidingWindow> Initializer::init_sfm() const {
    // [1] find a proper pair for initialization
    Frame *init_frame_i = nullptr;
    Frame *init_frame_j = raw->get_frame(raw->frame_num() - 1);
    size_t init_frame_i_id = nil();
    std::vector<Vector3d> init_points;
    std::vector<std::pair<size_t, size_t>> init_matches;
    std::vector<char> init_point_status;
    Matrix3d init_R;
    Vector3d init_T;

    Frame *frame_j = init_frame_j;
    std::vector<Vector2d> frame_i_keypoints;
    std::vector<Vector2d> frame_j_keypoints;
    for (size_t frame_i_id = 0; frame_i_id + config->min_init_raw_frames() < raw->frame_num(); ++frame_i_id) {
        double total_parallax = 0;
        int common_track_num = 0;
        Frame *frame_i = raw->get_frame(frame_i_id);
        frame_i_keypoints.clear();
        frame_j_keypoints.clear();
        init_matches.clear();
        for (size_t ki = 0; ki < frame_i->keypoint_num(); ++ki) {
            Track *track = frame_i->get_track(ki);
            if (!track) continue;
            size_t kj = track->get_keypoint_id(init_frame_j);
            if (kj == nil()) continue;
            frame_i_keypoints.push_back(frame_i->get_keypoint(ki));
            frame_j_keypoints.push_back(frame_j->get_keypoint(kj));
            init_matches.emplace_back(ki, kj);
            total_parallax += (apply_k(frame_i->get_keypoint(ki), frame_i->K) - apply_k(frame_j->get_keypoint(kj), frame_j->K)).norm();
            common_track_num++;
        }

        if (common_track_num < (int)config->min_raw_matches()) continue;
        total_parallax /= std::max(common_track_num, 1);
        if (total_parallax < config->min_raw_parallax()) continue;

        std::vector<Matrix3d> Rs;
        std::vector<Vector3d> Ts;

        Matrix3d RH1, RH2;
        Vector3d TH1, TH2, nH1, nH2;
        Matrix3d H = find_homography_matrix(frame_i_keypoints, frame_j_keypoints, 0.7 / frame_i->K(0, 0), 0.999, 1000, config->random());
        if (!decompose_homography(H, RH1, RH2, TH1, TH2, nH1, nH2)) {
            continue; // is pure rotation
        }
        TH1 = TH1.normalized();
        TH2 = TH2.normalized();
        Rs.insert(Rs.end(), {RH1, RH1, RH2, RH2});
        Ts.insert(Ts.end(), {TH1, -TH1, TH2, -TH2});

        Matrix3d RE1, RE2;
        Vector3d TE;
        Matrix3d E = find_essential_matrix(frame_i_keypoints, frame_j_keypoints, 0.7 / frame_i->K(0, 0), 0.999, 1000, config->random());
        decompose_essential(E, RE1, RE2, TE);
        TE = TE.normalized();
        Rs.insert(Rs.end(), {RE1, RE1, RE2, RE2});
        Ts.insert(Ts.end(), {TE, -TE, TE, -TE});

        size_t triangulated_num = triangulate_from_rt_scored(frame_i_keypoints, frame_j_keypoints, Rs, Ts, config->min_raw_triangulation(), init_points, init_R, init_T, init_point_status);

        if (triangulated_num < config->min_raw_triangulation()) {
            continue;
        }

        init_frame_i = frame_i;
        init_frame_i_id = frame_i_id;
        break;
    }

    if (!init_frame_i) return nullptr;

    // [2] create sfm map

    // [2.1] enumerate keyframe ids
    std::vector<size_t> init_keyframe_ids;
    size_t init_map_frames = config->init_map_frames();
    double keyframe_id_gap = (double)(raw->frame_num() - 1 - init_frame_i_id) / (double)(init_map_frames - 1);
    for (size_t i = 0; i < init_map_frames; ++i) {
        init_keyframe_ids.push_back((size_t)round(init_frame_i_id + keyframe_id_gap * i));
    }

    // [2.2] make a clone of submap using keyframe ids
    std::unique_ptr<SlidingWindow> map = std::make_unique<SlidingWindow>();
    for (size_t i = 0; i < init_keyframe_ids.size(); ++i) {
        map->put_frame(raw->get_frame(init_keyframe_ids[i])->clone());
    }
    for (size_t j = 1; j < init_keyframe_ids.size(); ++j) {
        Frame *old_frame_i = raw->get_frame(init_keyframe_ids[j - 1]);
        Frame *old_frame_j = raw->get_frame(init_keyframe_ids[j]);
        Frame *new_frame_i = map->get_frame(j - 1);
        Frame *new_frame_j = map->get_frame(j);
        for (size_t ki = 0; ki < old_frame_i->keypoint_num(); ++ki) {
            Track *track = old_frame_i->get_track(ki);
            if (track == nullptr) continue;
            size_t kj = track->get_keypoint_id(old_frame_j);
            if (kj == nil()) continue;
            new_frame_i->get_track(ki, create_if_empty)->add_keypoint(new_frame_j, kj);
        }
        new_frame_j->preintegration.data.clear();
        for (size_t f = init_keyframe_ids[j - 1]; f < init_keyframe_ids[j]; ++f) {
            Frame *old_frame = raw->get_frame(f + 1);
            std::vector<IMUData> &old_data = old_frame->preintegration.data;
            std::vector<IMUData> &new_data = new_frame_j->preintegration.data;
            new_data.insert(new_data.end(), old_data.begin(), old_data.end());
        }
    }

    Frame *new_init_frame_i = map->get_frame(0);
    Frame *new_init_frame_j = map->get_frame(map->frame_num() - 1);

    // [2.3] set init states
    PoseState pose;
    pose.q.setIdentity();
    pose.p.setZero();
    new_init_frame_i->set_pose(new_init_frame_i->camera, pose);
    pose.q = init_R.transpose();
    pose.p = -(init_R.transpose() * init_T);
    new_init_frame_j->set_pose(new_init_frame_j->camera, pose);

    for (size_t k = 0; k < init_points.size(); ++k) {
        if (init_point_status[k] == 0) continue;
        Track *track = new_init_frame_i->get_track(init_matches[k].first);
        track->landmark.x = init_points[k];
        track->landmark.flag(LF_VALID) = true;
    }

    // [2.4] solve other frames via pnp
    for (size_t j = 1; j + 1 < map->frame_num(); ++j) {
        Frame *frame_i = map->get_frame(j - 1);
        Frame *frame_j = map->get_frame(j);
        frame_j->set_pose(frame_j->camera, frame_i->get_pose(frame_i->camera));
        visual_inertial_pnp(map.get(), frame_j, false);
    }

    // [2.5] triangulate more points
    for (size_t i = 0; i < map->track_num(); ++i) {
        Track *track = map->get_track(i);
        if (track->landmark.flag(LF_VALID)) continue;
        track->triangulate();
    }

    // [3] sfm

    // [3.1] bundle adjustment
    map->get_frame(0)->pose.flag(PF_FIXED) = true;
    if (!BundleAdjustor().solve(map.get(), false, config->solver_iteration_limit() * 5, config->solver_time_limit())) {
        return nullptr;
    }

    // [3.2] cleanup invalid points
    map->prune_tracks([](const Track *track) {
        return !track->landmark.flag(LF_VALID) || track->landmark.quality > 1.0;
    });

    return map;
}
