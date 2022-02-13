#pragma once

#include <btrc/core/renderer/renderer.h>

BTRC_CORE_BEGIN

class WavefrontPathTracer : public Renderer, public Uncopyable
{
public:

    struct Params
    {
        int spp    = 128;
        int width  = 512;
        int height = 512;

        int   min_depth    = 5;
        int   max_depth    = 100;
        float rr_threshold = 0.1f;
        float rr_cont_prob = 0.5f;

        int state_count = 1000000;

        bool albedo = false;
        bool normal = false;
    };

    WavefrontPathTracer();

    ~WavefrontPathTracer() override;

    void set_params(const Params &params);

    void set_scene(RC<const Scene> scene) override;

    RenderResult render() const override;

private:

    void build_pipeline();

    struct Impl;

    Box<Impl> impl_;
};

BTRC_CORE_END
