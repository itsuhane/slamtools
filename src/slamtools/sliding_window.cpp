#include <slamtools/sliding_window.h>
#include <cstdio>
#include <unordered_set>
#include <unordered_map>
#include <slamtools/frame.h>
#include <slamtools/track.h>
#include <slamtools/factor.h>

struct SlidingWindow::construct_by_map_t {};

SlidingWindow::SlidingWindow() = default;
SlidingWindow::~SlidingWindow() = default;

void SlidingWindow::clear() {
    frames.clear();
    tracks.clear();
}

void SlidingWindow::put_frame(std::unique_ptr<Frame> frame, size_t pos) {
    frame->map = this;
    if (pos == nil()) {
        frames.emplace_back(std::move(frame));
        pos = frames.size() - 1;
    } else {
        frames.emplace(frames.begin() + pos, std::move(frame));
    }
    if (pos > 0) {
        Frame *frame_i = frames[pos - 1].get();
        Frame *frame_j = frames[pos].get();
        frame_j->preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
    }
    if (pos < frames.size() - 1) {
        Frame *frame_i = frames[pos].get();
        Frame *frame_j = frames[pos + 1].get();
        frame_j->preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
    }
}

void SlidingWindow::erase_frame(size_t id) {
    Frame *frame = frames[id].get();
    for (size_t i = 0; i < frame->keypoint_num(); ++i) {
        Track *track = frame->get_track(i);
        if (track != nullptr) {
            track->remove_keypoint(frame);
        }
    }
    frames.erase(frames.begin() + id);
    if (id > 0 && id < frames.size()) {
        Frame *frame_i = frames[id - 1].get();
        Frame *frame_j = frames[id].get();
        frame_j->preintegration_factor = Factor::create_preintegration_error(frame_i, frame_j);
    }
}

Track *SlidingWindow::create_track() {
    std::unique_ptr<Track> track = std::make_unique<Track>(construct_by_map_t());
    track->id_in_map = tracks.size();
    track->map = this;
    tracks.emplace_back(std::move(track));
    return tracks.back().get();
}

void SlidingWindow::erase_track(Track *track) {
    while (track->keypoint_map().size() > 0) {
        track->remove_keypoint(track->keypoint_map().begin()->first, false);
    }
    recycle_track(track);
}

void SlidingWindow::prune_tracks(const std::function<bool(const Track *)> &condition) {
    std::vector<Track *> tracks_to_prune;
    for (size_t i = 0; i < track_num(); ++i) {
        Track *track = get_track(i);
        if (condition(track)) {
            tracks_to_prune.push_back(track);
        }
    }
    for (Track *t : tracks_to_prune) {
        erase_track(t);
    }
}

void SlidingWindow::recycle_track(Track *track) {
    if (track->id_in_map != tracks.back()->id_in_map) {
        tracks[track->id_in_map].swap(tracks.back());
        tracks[track->id_in_map]->id_in_map = track->id_in_map;
    }
    tracks.pop_back();
}
