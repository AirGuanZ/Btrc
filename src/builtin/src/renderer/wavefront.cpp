#include <btrc/builtin/renderer/wavefront.h>

#include "./wavefront/generate.h"
#include "./wavefront/path_state.h"
#include "./wavefront/shade.h"
#include "./wavefront/shadow.h"
#include "./wavefront/sort.h"
#include "./wavefront/trace.h"

BTRC_BUILTIN_BEGIN

struct WavefrontPathTracer::Impl
{
    optix::Context *optix_ctx = nullptr;

    Params           params;
    RC<const Scene>  scene;
    RC<const Camera> camera;

    int width = 512;
    int height = 512;

    bool is_dirty = true;

    Film                 film;
    wfpt::PathState        path_state;
    wfpt::GeneratePipeline generate;
    wfpt::TracePipeline    trace;
    wfpt::SortPipeline     sort;
    wfpt::ShadePipeline    shade;
    wfpt::ShadowPipeline   shadow;
};

WavefrontPathTracer::WavefrontPathTracer(optix::Context &optix_ctx)
{
    impl_ = newBox<Impl>();
    impl_->optix_ctx = &optix_ctx;
}

WavefrontPathTracer::~WavefrontPathTracer()
{
    
}

void WavefrontPathTracer::set_params(const Params &params)
{
    impl_->params = params;
    impl_->is_dirty = true;
}

void WavefrontPathTracer::set_scene(RC<const Scene> scene)
{
    impl_->scene = scene;
    impl_->is_dirty = true;
}

void WavefrontPathTracer::set_camera(RC<const Camera> camera)
{
    impl_->camera = std::move(camera);
}

void WavefrontPathTracer::set_film(int width, int height)
{
    impl_->width = width;
    impl_->height = height;
    impl_->is_dirty = true;
}

Renderer::RenderResult WavefrontPathTracer::render() const
{
    if(impl_->is_dirty)
        build_pipeline();

    impl_->generate.clear();
    impl_->path_state.clear();

    auto &scene = *impl_->scene;
    auto &soa = impl_->path_state;
    auto &params = impl_->params;

    int active_state_count = 0;
    while(!impl_->generate.is_done() || active_state_count > 0)
    {
        const int new_state_count = impl_->generate.generate(
            active_state_count,
            wfpt::GeneratePipeline::SOAParams{
                .rng                  = soa.rng,
                .output_pixel_coord   = soa.pixel_coord,
                .output_ray_o_t0      = soa.o_t0,
                .output_ray_d_t1      = soa.d_t1,
                .output_ray_time_mask = soa.time_mask,
                .output_beta          = soa.beta,
                .output_beta_le       = soa.beta_le,
                .output_bsdf_pdf      = soa.bsdf_pdf,
                .output_depth         = soa.depth,
                .output_path_radiance = soa.path_radiance
            });
        active_state_count += new_state_count;

        impl_->trace.trace(
            scene.get_tlas(),
            active_state_count,
            wfpt::TracePipeline::SOAParams{
                .ray_o_t0      = soa.o_t0,
                .ray_d_t1      = soa.d_t1,
                .ray_time_mask = soa.time_mask,
                .inct_t        = soa.inct_t,
                .inct_uv_id    = soa.inct_uv_id,
                .state_index   = soa.active_state_indices
            });

        // sort

        const auto shade_counters = impl_->shade.shade(
            active_state_count,
            wfpt::ShadePipeline::SOAParams{
                .rng                         = soa.rng,
                .active_state_indices        = soa.active_state_indices,
                .path_radiance               = soa.path_radiance,
                .pixel_coord                 = soa.pixel_coord,
                .depth                       = soa.depth,
                .beta                        = soa.beta,
                .beta_le                     = soa.beta_le,
                .bsdf_pdf                    = soa.bsdf_pdf,
                .inct_t                      = soa.inct_t,
                .inct_uv_id                  = soa.inct_uv_id,
                .ray_o_t0                    = soa.o_t0,
                .ray_d_t1                    = soa.d_t1,
                .ray_time_mask               = soa.time_mask,
                .output_rng                  = soa.next_rng,
                .output_path_radiance        = soa.next_path_radiance,
                .output_pixel_coord          = soa.next_pixel_coord,
                .output_depth                = soa.next_depth,
                .output_beta                 = soa.next_beta,
                .output_shadow_pixel_coord   = soa.shadow_pixel_coord,
                .output_shadow_ray_o_t0      = soa.shadow_o_t0,
                .output_shadow_ray_d_t1      = soa.shadow_d_t1,
                .output_shadow_ray_time_mask = soa.shadow_time_mask,
                .output_shadow_beta_li       = soa.shadow_beta_li,
                .output_new_ray_o_t0         = soa.next_o_t0,
                .output_new_ray_d_t1         = soa.next_d_t1,
                .output_new_ray_time_mask    = soa.next_time_mask,
                .output_beta_le              = soa.next_beta_le,
                .output_bsdf_pdf             = soa.next_bsdf_pdf
            });

        active_state_count = shade_counters.active_state_counter;
        
        if(shade_counters.shadow_ray_counter)
        {
            impl_->shadow.test(
                scene.get_tlas(),
                shade_counters.shadow_ray_counter,
                wfpt::ShadowPipeline::SOAParams{
                    .pixel_coord   = soa.shadow_pixel_coord,
                    .ray_o_t0      = soa.shadow_o_t0,
                    .ray_d_t1      = soa.shadow_d_t1,
                    .ray_time_mask = soa.shadow_time_mask,
                    .beta_li       = soa.shadow_beta_li
                });
        }

        soa.next_iteration();
    }

    throw_on_error(cudaStreamSynchronize(nullptr));

    auto value = Image<Vec4f>(impl_->width, impl_->height);
    auto weight = Image<float>(impl_->width, impl_->height);
    auto albedo = params.albedo ? Image<Vec4f>(impl_->width, impl_->height) : Image<Vec4f>();
    auto normal = params.normal ? Image<Vec4f>(impl_->width, impl_->height) : Image<Vec4f>();

    impl_->film.get_float3_output(Film::OUTPUT_RADIANCE).to_cpu(&value(0, 0).x);
    impl_->film.get_float_output(Film::OUTPUT_WEIGHT).to_cpu(&weight(0, 0));
    if(params.albedo)
        impl_->film.get_float3_output(Film::OUTPUT_ALBEDO).to_cpu(&albedo(0, 0).x);
    if(params.normal)
        impl_->film.get_float3_output(Film::OUTPUT_NORMAL).to_cpu(&normal(0, 0).x);

    for(int i = 0; i < impl_->width * impl_->height; ++i)
    {
        float &f = weight.data()[i];
        if(f > 0)
            f = 1.0f / f;
    }

    RenderResult result;
    result.value = Image<Vec3f>(impl_->width, impl_->height);
    for(int i = 0; i < impl_->width * impl_->height; ++i)
    {
        const Vec4f &sum = value.data()[i];
        const float ratio = weight.data()[i];
        result.value.data()[i] = ratio * sum.xyz();
    }

    if(params.albedo)
    {
        result.albedo = Image<Vec3f>(impl_->width, impl_->height);
        for(int i = 0; i < impl_->width * impl_->height; ++i)
        {
            const Vec4f &sum = albedo.data()[i];
            const float ratio = weight.data()[i];
            result.albedo.data()[i] = ratio * sum.xyz();
        }
    }
    if(params.normal)
    {
        result.normal = Image<Vec3f>(impl_->width, impl_->height);
        for(int i = 0; i < impl_->width * impl_->height; ++i)
        {
            const Vec4f &sum = normal.data()[i];
            const float ratio = weight.data()[i];
            result.normal.data()[i] = 0.5f + 0.5f * ratio * sum.xyz();
        }
    }

    return result;
}

void WavefrontPathTracer::build_pipeline() const
{
    assert(impl_->is_dirty);
    BTRC_SCOPE_SUCCESS{ impl_->is_dirty = false; };

    auto &params = impl_->params;

    // film

    impl_->film = Film(impl_->width, impl_->height);
    impl_->film.add_output(Film::OUTPUT_RADIANCE, Film::Float3);
    impl_->film.add_output(Film::OUTPUT_WEIGHT, Film::Float);
    if(params.albedo)
        impl_->film.add_output(Film::OUTPUT_ALBEDO, Film::Float3);
    if(params.normal)
        impl_->film.add_output(Film::OUTPUT_NORMAL, Film::Float3);

    // pipelines

    impl_->generate = wfpt::GeneratePipeline(
        *impl_->camera,
        { impl_->width, impl_->height },
        params.spp, params.state_count);

    impl_->trace = wfpt::TracePipeline(
        *impl_->optix_ctx,
        impl_->scene->has_motion_blur(),
        impl_->scene->is_triangle_only(),
        2);

    impl_->sort = wfpt::SortPipeline();

    impl_->shade = wfpt::ShadePipeline(
        impl_->film, *impl_->scene, wfpt::ShadePipeline::ShadeParams{
            .min_depth = params.min_depth,
            .max_depth = params.max_depth,
            .rr_threshold = params.rr_threshold,
            .rr_cont_prob = params.rr_cont_prob
        });

    impl_->shadow = wfpt::ShadowPipeline(
        impl_->film, *impl_->optix_ctx,
        impl_->scene->has_motion_blur(),
        impl_->scene->is_triangle_only(),
        2);

    // path state

    impl_->path_state.initialize(params.state_count);
}

RC<Renderer> WavefrontPathTracerCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    WavefrontPathTracer::Params params;
    params.spp          = node->parse_child_or("spp", params.spp);
    params.min_depth    = node->parse_child_or("min_depth", params.min_depth);
    params.max_depth    = node->parse_child_or("max_depth", params.max_depth);
    params.rr_threshold = node->parse_child_or("rr_threshold", params.rr_threshold);
    params.rr_cont_prob = node->parse_child_or("rr_cont_prob", params.rr_cont_prob);
    params.state_count  = node->parse_child_or("state_count", params.state_count);
    params.albedo       = node->parse_child_or("albedo", params.albedo);
    params.normal       = node->parse_child_or("normal", params.normal);
    auto wfpt = newRC<WavefrontPathTracer>(context.get_optix_context());
    wfpt->set_params(params);
    return wfpt;
}

BTRC_BUILTIN_END
