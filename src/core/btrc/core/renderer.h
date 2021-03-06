#pragma once

#include <atomic>
#include <future>

#include <btrc/core/camera.h>
#include <btrc/core/context.h>
#include <btrc/core/reporter.h>
#include <btrc/core/scene.h>

BTRC_BEGIN

class Renderer : public Object
{
public:

    struct RenderResult
    {
        cuda::Buffer<Vec4f> color;
        cuda::Buffer<Vec4f> albedo;
        cuda::Buffer<Vec4f> normal;
    };

    virtual ~Renderer() = default;

    virtual void set_scene(RC<Scene> scene) = 0;

    virtual void set_camera(RC<Camera> camera) = 0;

    virtual void set_film(int width, int height) = 0;

    virtual void set_reporter(RC<Reporter> reporter) = 0;

    virtual RenderResult render() = 0;

    void render_async();

    void stop_async();

    RenderResult wait_async();

    bool is_waitable() const;

    bool is_rendering() const;

protected:

    bool should_stop() const;

private:

    std::atomic<bool> stop_      = false;
    std::atomic<bool> rendering_ = false;
    bool waitable_               = false;

    std::future<RenderResult> async_future_;
};

BTRC_END
