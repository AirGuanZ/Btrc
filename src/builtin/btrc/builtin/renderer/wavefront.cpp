#include <btrc/builtin/film_filter/box.h>
#include <btrc/builtin/renderer/wavefront/generate.h>
#include <btrc/builtin/renderer/wavefront/medium.h>
#include <btrc/builtin/renderer/wavefront/preview.h>
#include <btrc/builtin/renderer/wavefront/shade.h>
#include <btrc/builtin/renderer/wavefront/shadow.h>
#include <btrc/builtin/renderer/wavefront/soa_buffer.h>
#include <btrc/builtin/renderer/wavefront/trace.h>
#include <btrc/builtin/renderer/wavefront.h>

BTRC_BUILTIN_BEGIN

struct WavefrontPathTracer::Impl
{
    optix::Context *optix_ctx = nullptr;

    Params         params;
    RC<FilmFilter> filter;
    RC<Scene>      scene;
    RC<Camera>     camera;
    RC<Reporter>   reporter;

    bool has_medium = false;

    int width = 512;
    int height = 512;

    Film film;
    
    RC<wfpt::RayBuffer>          ray_buffer;
    RC<wfpt::PathBuffer>         path_buffer;
    RC<wfpt::BSDFLeBuffer>       bsdf_le_buffer;
    RC<wfpt::IntersectionBuffer> inct_buffer;

    RC<wfpt::RayBuffer>          next_ray_buffer;
    RC<wfpt::PathBuffer>         next_path_buffer;
    RC<wfpt::BSDFLeBuffer>       next_bsdf_le_buffer;

    RC<wfpt::ShadowRayBuffer>     shadow_ray_buffer;
    RC<wfpt::ShadowSamplerBuffer> shadow_sampler_buffer;

    wfpt::GeneratePipeline      generate;
    wfpt::TracePipeline         trace;
    wfpt::MediumPipeline        medium;
    wfpt::ShadePipeline         shade;
    wfpt::ShadowPipeline        shadow;
    wfpt::PreviewImageGenerator preview;

    RC<cuda::Buffer<wfpt::StateCounters>> state_counters;

    cuda::Buffer<Vec4f> device_preview_image;
    cuda::Buffer<Vec4f> device_preview_normal;
    cuda::Buffer<Vec4f> device_preview_albedo;
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
}

void WavefrontPathTracer::set_film_filter(RC<FilmFilter> filter)
{
    impl_->filter = std::move(filter);
}

void WavefrontPathTracer::set_scene(RC<Scene> scene)
{
    impl_->scene = scene;
}

void WavefrontPathTracer::set_camera(RC<Camera> camera)
{
    impl_->camera = std::move(camera);
}

void WavefrontPathTracer::set_film(int width, int height)
{
    impl_->width = width;
    impl_->height = height;
}

void WavefrontPathTracer::set_reporter(RC<Reporter> reporter)
{
    impl_->reporter = std::move(reporter);
}

std::vector<RC<Object>> WavefrontPathTracer::get_dependent_objects()
{
    std::set<RC<Object>> scene_objects;
    impl_->scene->collect_objects(scene_objects);
    std::vector<RC<Object>> result = { scene_objects.begin(), scene_objects.end() };
    result.push_back(impl_->camera);
    result.push_back(impl_->filter);
    return result;
}

void WavefrontPathTracer::recompile()
{
    CompileContext cc;

    auto &params = impl_->params;

    impl_->has_medium = impl_->scene->has_medium();

    // film

    impl_->film = Film(impl_->width, impl_->height);
    impl_->film.add_output(Film::OUTPUT_RADIANCE, Film::Float3);
    impl_->film.add_output(Film::OUTPUT_WEIGHT, Film::Float);
    if(params.albedo)
        impl_->film.add_output(Film::OUTPUT_ALBEDO, Film::Float3);
    if(params.normal)
        impl_->film.add_output(Film::OUTPUT_NORMAL, Film::Float3);

    // counters

    impl_->state_counters = newRC<cuda::Buffer<wfpt::StateCounters>>(1);

    // pipelines

    const auto shade_params = wfpt::ShadeParams{
        .min_depth = params.min_depth,
        .max_depth = params.max_depth,
        .rr_threshold = params.rr_threshold,
        .rr_cont_prob = params.rr_cont_prob
    };

    impl_->generate = {};
    impl_->trace = {};
    impl_->medium = {};
    impl_->shade = {};
    impl_->shadow = {};

    const AABB3f world_bbox = union_aabb(impl_->camera->get_bounding_box(), impl_->scene->get_bbox());
    const float world_diagonal = 1.2f * length(world_bbox.upper - world_bbox.lower);

    {
        cuj::ScopedModule cuj_module;

        impl_->generate.record_device_code(cc, *impl_->scene, *impl_->camera, impl_->film, *impl_->filter);
        if(impl_->has_medium)
            impl_->medium.record_device_code(cc, impl_->film, *impl_->scene, shade_params, world_diagonal);
        impl_->shade.record_device_code(cc, impl_->film, *impl_->scene, shade_params, world_diagonal);

        cuj::PTXGenerator ptx_gen;
        ptx_gen.set_options(cuj::Options{
            .opt_level = cuj::OptimizationLevel::O3,
            .fast_math = true,
            .approx_math_func = true
        });
        ptx_gen.generate(cuj_module);

        auto &ptx = ptx_gen.get_ptx();
        auto cuda_module = newRC<cuda::Module>();
        cuda_module->load_ptx_from_memory(ptx.data(), ptx.size());
        cuda_module->link();

        impl_->generate.initialize(cuda_module, params.spp, params.state_count, { impl_->width, impl_->height });
        if(impl_->has_medium)
            impl_->medium.initialize(cuda_module, impl_->state_counters, *impl_->scene);
        impl_->shade.initialize(cuda_module, impl_->state_counters, *impl_->scene);
    }

    impl_->trace = wfpt::TracePipeline(
        *impl_->optix_ctx,
        impl_->scene->has_motion_blur(),
        impl_->scene->is_triangle_only(),
        2);

    impl_->shadow = wfpt::ShadowPipeline(
        *impl_->scene, impl_->film, *impl_->optix_ctx,
        impl_->scene->has_motion_blur(),
        impl_->scene->is_triangle_only(),
        2, world_diagonal);

    // path state

    impl_->ray_buffer     = newRC<wfpt::RayBuffer>(params.state_count);
    impl_->path_buffer    = newRC<wfpt::PathBuffer>(params.state_count);
    impl_->bsdf_le_buffer = newRC<wfpt::BSDFLeBuffer>(params.state_count);
    impl_->inct_buffer    = newRC<wfpt::IntersectionBuffer>(params.state_count);

    impl_->next_ray_buffer = newRC<wfpt::RayBuffer>(params.state_count);
    impl_->next_path_buffer = newRC<wfpt::PathBuffer>(params.state_count);
    impl_->next_bsdf_le_buffer = newRC<wfpt::BSDFLeBuffer>(params.state_count);
    
    impl_->shadow_ray_buffer = newRC<wfpt::ShadowRayBuffer>(params.state_count);
    impl_->shadow_sampler_buffer = newRC<wfpt::ShadowSamplerBuffer>(params.state_count);
}

Renderer::RenderResult WavefrontPathTracer::render()
{
    impl_->generate.clear();
    impl_->shadow_sampler_buffer->clear();

    impl_->film.clear_output(Film::OUTPUT_RADIANCE);
    impl_->film.clear_output(Film::OUTPUT_WEIGHT);
    if(impl_->film.has_output(Film::OUTPUT_ALBEDO))
        impl_->film.clear_output(Film::OUTPUT_ALBEDO);
    if(impl_->film.has_output(Film::OUTPUT_NORMAL))
        impl_->film.clear_output(Film::OUTPUT_NORMAL);

    auto &scene = *impl_->scene;
    auto &params = impl_->params;
    auto &reporter = *impl_->reporter;

    reporter.new_stage();

    const uint64_t total_path_count = static_cast<uint64_t>(params.spp) * impl_->width * impl_->height;
    uint64_t finished_path_count = 0;

    int active_state_count = 0;
    while(!impl_->generate.is_done() || active_state_count > 0)
    {
        const int64_t limited_state_count = reporter.need_fast_preview() ?
            (active_state_count ? active_state_count : impl_->width * impl_->height) :
            (std::numeric_limits<int64_t>::max)();

        const int new_state_count = impl_->generate.generate(
            active_state_count,
            wfpt::GeneratePipeline::SOAParams{
                .path    = *impl_->path_buffer,
                .ray     = *impl_->ray_buffer,
                .bsdf_le = *impl_->bsdf_le_buffer
            },
            limited_state_count);

        active_state_count += new_state_count;

        impl_->trace.trace(
            scene.get_tlas(),
            active_state_count,
            wfpt::TracePipeline::SOAParams{
                .ray  = *impl_->ray_buffer,
                .inct = *impl_->inct_buffer
            });

        impl_->state_counters->clear_bytes_async(0);

        if(impl_->has_medium)
        {
            impl_->medium.sample_scattering(
                active_state_count,
                wfpt::MediumPipeline::SOAParams{
                    .path           = *impl_->path_buffer,
                    .ray            = *impl_->ray_buffer,
                    .bsdf_le        = *impl_->bsdf_le_buffer,
                    .inct           = *impl_->inct_buffer,
                    .output_path    = *impl_->next_path_buffer,
                    .output_ray     = *impl_->next_ray_buffer,
                    .output_bsdf_le = *impl_->next_bsdf_le_buffer,
                    .shadow_ray     = *impl_->shadow_ray_buffer
                });
        }

        impl_->shade.shade(
            active_state_count,
            wfpt::ShadePipeline::SOAParams{
                .path           = *impl_->path_buffer,
                .ray            = *impl_->ray_buffer,
                .bsdf_le        = *impl_->bsdf_le_buffer,
                .inct           = *impl_->inct_buffer,
                .output_path    = *impl_->next_path_buffer,
                .output_ray     = *impl_->next_ray_buffer,
                .output_bsdf_le = *impl_->next_bsdf_le_buffer,
                .shadow_ray     = *impl_->shadow_ray_buffer
            });

        wfpt::StateCounters state_counters;
        impl_->state_counters->to_cpu(&state_counters);

        finished_path_count += active_state_count - state_counters.active_state_counter;
        reporter.progress(100.0f * finished_path_count / total_path_count);
        active_state_count = state_counters.active_state_counter;

        if(state_counters.shadow_ray_counter)
        {
            impl_->shadow.test(
                scene.get_tlas(),
                state_counters.shadow_ray_counter,
                wfpt::ShadowPipeline::SOAParams{
                    .shadow_ray    = *impl_->shadow_ray_buffer,
                    .sampler_state = *impl_->shadow_sampler_buffer
                });
        }

        if(reporter.need_preview())
            new_preview_image();

        std::swap(impl_->path_buffer, impl_->next_path_buffer);
        std::swap(impl_->ray_buffer, impl_->next_ray_buffer);
        std::swap(impl_->bsdf_le_buffer, impl_->next_bsdf_le_buffer);

        if(should_stop())
            break;
    }

    throw_on_error(cudaStreamSynchronize(nullptr));

    reporter.complete_stage();
    if(should_stop())
        return {};

    new_preview_image();

    update_device_preview_data();
    RenderResult result;
    result.color.swap(impl_->device_preview_image);
    result.albedo.swap(impl_->device_preview_albedo);
    result.normal.swap(impl_->device_preview_normal);
    return result;
}

void WavefrontPathTracer::update_device_preview_data()
{
    const size_t texel_count = impl_->width * impl_->height;
    if(impl_->device_preview_image.get_size() != texel_count)
        impl_->device_preview_image.initialize(texel_count);

    impl_->preview.generate(
        impl_->width, impl_->height,
        impl_->film.get_float3_output(Film::OUTPUT_RADIANCE).as<Vec4f>(),
        impl_->film.get_float_output(Film::OUTPUT_WEIGHT).get(),
        impl_->device_preview_image.get());

    if(impl_->params.albedo)
    {
        if(impl_->device_preview_albedo.get_size() != texel_count)
            impl_->device_preview_albedo.initialize(texel_count);

        impl_->preview.generate_albedo(
            impl_->width, impl_->height,
            impl_->film.get_float3_output(Film::OUTPUT_ALBEDO).as<Vec4f>(),
            impl_->film.get_float_output(Film::OUTPUT_WEIGHT).get(),
            impl_->device_preview_albedo.get());
    }

    if(impl_->params.normal)
    {
        if(impl_->device_preview_normal.get_size() != texel_count)
            impl_->device_preview_normal.initialize(texel_count);

        impl_->preview.generate_normal(
            impl_->width, impl_->height,
            impl_->film.get_float3_output(Film::OUTPUT_NORMAL).as<Vec4f>(),
            impl_->film.get_float_output(Film::OUTPUT_WEIGHT).get(),
            impl_->device_preview_normal.get());
    }
}

void WavefrontPathTracer::new_preview_image()
{
    update_device_preview_data();
    impl_->reporter->new_preview(
        impl_->device_preview_image.get(),
        impl_->device_preview_albedo.get(),
        impl_->device_preview_normal.get(),
        impl_->width,
        impl_->height);
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

    RC<FilmFilter> filter;
    if(auto n = node->find_child_node("filter"))
        filter = context.create<FilmFilter>(n);
    else
        filter = newRC<BoxFilter>();

    auto wfpt = newRC<WavefrontPathTracer>(context.get_optix_context());
    wfpt->set_params(params);
    wfpt->set_film_filter(std::move(filter));
    return wfpt;
}

BTRC_BUILTIN_END
