#pragma once

#include <slamtools/common.h>
#include <slamtools/sliding_window.h>
#include <slamtools/state.h>
#include <slamtools/frame.h>

class Frame;
class SlidingWindow;

class Track {
    friend class SlidingWindow;
    size_t id_in_map;
    SlidingWindow *map;
    Track();

  public:
    Track(const SlidingWindow::construct_by_map_t &) :
        Track() {
    }

    virtual ~Track();

    const std::map<Frame *, size_t, Comparator<Frame *>> &keypoint_map() const {
        return keypoint_refs;
    }

    bool has_keypoint(Frame *frame) const {
        return keypoint_refs.count(frame) > 0;
    }

    size_t get_keypoint_id(Frame *frame) const {
        if (has_keypoint(frame)) {
            return keypoint_refs.at(frame);
        } else {
            return nil();
        }
    }

    const Eigen::Vector2d &get_keypoint(Frame *frame) const;

    void add_keypoint(Frame *frame, size_t keypoint_id);

    void remove_keypoint(Frame *frame, bool suicide_if_empty = true);

    bool triangulate();

    LandmarkState landmark;

  private:
    std::map<Frame *, size_t, Comparator<Frame *>> keypoint_refs;
};

// This comparator is incorrect in theory, consider if a and b are modified...
// Hence, don't use this in production.
template <>
struct Comparator<Track> {
    bool operator()(const Track &a, const Track &b) const {
        auto ia = a.keypoint_map().cbegin();
        auto ib = b.keypoint_map().cbegin();
        while (ia != a.keypoint_map().cend() && ib != b.keypoint_map().cend()) {
            bool eab = Comparator<Frame *>()(ia->first, ib->first);
            bool eba = Comparator<Frame *>()(ib->first, ia->first);
            if (eab || eba) {
                return eab;
            } else if (ia->second != ib->second) {
                return ia->second < ib->second;
            } else {
                ia++;
                ib++;
            }
        }
        return ia != a.keypoint_map().cend();
    }
};
