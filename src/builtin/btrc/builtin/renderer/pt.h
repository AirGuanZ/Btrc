#pragma once

#include <btrc/core/film_filter.h>
#include <btrc/core/renderer.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class PathTracer : public Renderer, public Uncopyable
{
public:

    struct Params
    {
        int spp = 128;

        int min_depth = 5;
        int max_depth = 10;
        float rr_threshold = 0.1f;
        float rr_cont_prob = 0.5f;

        bool albedo = false;
        bool normal = false;
    };

    explicit PathTracer(optix::Context &optix_ctx);

    ~PathTracer() override;

    void set_params(const Params &params);

    void set_film_filter(RC<FilmFilter> filter);

    void set_scene(RC<Scene> scene);

    void set_camera(RC<Camera> camera);

    void set_film(int width, int height) override;

    void set_reporter(RC<Reporter> reporter) override;

    void commit() override;

    std::vector<RC<Object>> get_dependent_objects() override;

    RenderResult render() override;

private:

    void update_device_preview_data();

    void new_preview_image();

    struct Impl;

    Box<Impl> impl_;
};

class PathTracerCreator : public factory::Creator<Renderer>
{
public:

    std::string get_name() const override { return "pt"; }

    RC<Renderer> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
