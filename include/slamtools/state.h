#pragma once

#include <slamtools/common.h>

enum ErrorStateLocation {
    ES_Q = 0,
    ES_P = 3,
    ES_V = 6,
    ES_BG = 9,
    ES_BA = 12,
    ES_SIZE = 15
};

enum PoseFlag {
    PF_FIXED = 0,
    PF_SIZE
};

enum LandmarkFlag {
    LF_VALID = 0,
    LF_SIZE
};

struct ExtrinsicParams {
    Eigen::Quaterniond q_cs;
    Eigen::Vector3d p_cs;
};

struct PoseState {
    PoseState() {
        q.setIdentity();
        p.setZero();
        flags.reset();
    }

    bool flag(PoseFlag f) const {
        return flags[f];
    }

    std::bitset<PF_SIZE>::reference flag(PoseFlag f) {
        return flags[f];
    }

    Eigen::Quaterniond q;
    Eigen::Vector3d p;

  private:
    std::bitset<PF_SIZE> flags;
};

struct MotionState {
    MotionState() {
        v.setZero();
        bg.setZero();
        ba.setZero();
    }

    Eigen::Vector3d v;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
};

struct LandmarkState {
    LandmarkState() {
        x.setZero();
        quality = 0;
        flags.reset();
    }

    Eigen::Vector3d x;
    double quality;

    bool flag(LandmarkFlag f) const {
        return flags[f];
    }

    std::bitset<LF_SIZE>::reference flag(LandmarkFlag f) {
        return flags[f];
    }

  private:
    std::bitset<LF_SIZE> flags;
};
