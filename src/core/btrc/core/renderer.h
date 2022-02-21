#pragma once

#include <atomic>
#include <future>

#include <btrc/core/context.h>
#include <btrc/core/reporter.h>
#include <btrc/core/scene.h>
#include <btrc/utils/image.h>

BTRC_BEGIN

class Renderer : public Object
{
public:

    struct RenderResult
    {
        Image<Vec3f> value;
        Image<Vec3f> albedo;
        Image<Vec3f> normal;
    };

    virtual void set_compile_context(CompileContext &cc) = 0;

    virtual void set_scene(RC<const Scene> scene) = 0;

    virtual void set_camera(RC<const Camera> camera) = 0;

    virtual void set_film(int width, int height) = 0;

    virtual void set_reporter(RC<Reporter> reporter) = 0;

    virtual RenderResult render() const = 0;

    void render_async();

    void stop_async();

    RenderResult wait_async();

    bool is_waitable() const;

    bool is_rendering() const;

private:

    std::atomic<bool> stop_      = false;
    std::atomic<bool> rendering_ = false;
    bool waitable_               = false;

    std::future<RenderResult> async_future_;
};

BTRC_END
