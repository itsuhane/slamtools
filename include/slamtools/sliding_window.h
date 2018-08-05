#pragma once

#include <slamtools/common.h>

class Frame;
class Track;
class Factor;

class SlidingWindow {
    friend class Track;
    struct construct_by_map_t;

  public:
    SlidingWindow();
    virtual ~SlidingWindow();

    void clear();

    size_t frame_num() const {
        return frames.size();
    }

    Frame *get_frame(size_t id) const {
        return frames[id].get();
    }

    void put_frame(std::unique_ptr<Frame> frame, size_t pos = nil());

    void erase_frame(size_t id);

    size_t track_num() const {
        return tracks.size();
    }

    Track *get_track(size_t id) const {
        return tracks[id].get();
    }

    Track *create_track();
    void erase_track(Track *track);

    void prune_tracks(const std::function<bool(const Track *)> &condition);

  private:
    void recycle_track(Track *track);

    std::deque<std::unique_ptr<Frame>> frames;
    std::vector<std::unique_ptr<Track>> tracks;
};
