#pragma once

#include <btrc/core/film_filter.h>
#include <btrc/core/renderer.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class WavefrontPathTracer : public Renderer, public Uncopyable
{
public:

    struct Params
    {
        bool tile = false;
        int spp = 128;

        int   min_depth    = 5;
        int   max_depth    = 100;
        float rr_threshold = 0.1f;
        float rr_cont_prob = 0.5f;

        int state_count = 1000000;

        bool albedo = false;
        bool normal = false;
    };

    explicit WavefrontPathTracer(optix::Context &optix_ctx);

    ~WavefrontPathTracer() override;

    void set_params(const Params &params);

    void set_film_filter(RC<FilmFilter> filter);

    void set_scene(RC<Scene> scene) override;

    void set_camera(RC<Camera> camera) override;

    void set_film(int width, int height) override;

    void set_reporter(RC<Reporter> reporter) override;

    std::vector<RC<Object>> get_dependent_objects() override;

    void recompile() override;

    RenderResult render() override;

private:

    void update_device_preview_data();

    void new_preview_image();

    struct Impl;

    Box<Impl> impl_;
};

class WavefrontPathTracerCreator : public factory::Creator<Renderer>
{
public:

    std::string get_name() const override { return "wfpt"; }

    RC<Renderer> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
