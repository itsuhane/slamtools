#pragma once

#include <slamtools/common.h>

class Frame;
class SlidingWindow;
class Configurator;

class Initializer {
  public:
    Initializer(std::shared_ptr<Configurator> config);
    virtual ~Initializer();

    virtual void append_frame(std::unique_ptr<Frame> frame);
    virtual std::unique_ptr<SlidingWindow> init() const = 0;

    virtual std::unique_ptr<SlidingWindow> init_sfm() const;

  protected:
    std::unique_ptr<SlidingWindow> raw;
    std::shared_ptr<Configurator> config;
};
