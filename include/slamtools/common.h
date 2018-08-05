#pragma once

#include <array>
#include <bitset>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <Eigen/Eigen>

#define GRAVITY_NOMINAL 9.80665

template <typename T>
struct Comparator; /*
    constexpr bool operator()(const T &a, const T &b) const;
*/

template <typename T>
struct Comparator<T *> {
    constexpr bool operator()(const T *a, const T *b) const {
        return Comparator<T>()(*a, *b);
    }
};

inline constexpr size_t nil() {
    return size_t(-1);
}

struct IMUData {
    double t;    // timestamp
    Eigen::Vector3d w; // gyro measurement
    Eigen::Vector3d a; // accelerometer measurement
};

struct Image {
    double t;
    Eigen::Vector3d g; // an external gravity measured by the device, it is only used for comparison or algorithms that uses external gravity.
    virtual ~Image() = default;
    virtual void detect_keypoints(std::vector<Eigen::Vector2d> &keypoints, size_t max_points = 0) const = 0;
    virtual void track_keypoints(const Image *next_image, const std::vector<Eigen::Vector2d> &curr_keypoints, std::vector<Eigen::Vector2d> &next_keypoints, std::vector<char> &result_status) const = 0;
    virtual void detect_segments(std::vector<std::tuple<Eigen::Vector2d, Eigen::Vector2d>> &segments, size_t max_segments = 0) const = 0;
};

template <>
struct Comparator<Image> {
    constexpr bool operator()(const Image &a, const Image &b) const {
        return a.t < b.t;
    }
};
