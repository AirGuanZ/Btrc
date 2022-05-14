#include <btrc/builtin/film_filter/box.h>
#include <btrc/builtin/renderer/pt/trace.h>
#include <btrc/builtin/renderer/pt.h>
#include <btrc/builtin/renderer/wavefront/preview.h>
#include <btrc/core/film.h>

BTRC_BUILTIN_BEGIN

namespace
{

    struct LaunchParams
    {
        int32_t finished_spp;
    };

    CUJ_CLASS(LaunchParams, finished_spp);

    using Pipeline = optix::MegaKernelPipeline<LaunchParams>;

} // namespace anonymous

struct PathTracer::Impl
{
    optix::Context *optix_ctx = nullptr;
    Params params;

    int width  = 1;
    int height = 1;

    RC<FilmFilter> filter;
    RC<Camera>     camera;
    RC<Scene>      scene;
    RC<Reporter>   reporter;

    Film film;
    Pipeline pipeline;
    wfpt::PreviewImageGenerator preview;

    cuda::Buffer<Vec4f> device_preview_image;
    cuda::Buffer<Vec4f> device_preview_normal;
    cuda::Buffer<Vec4f> device_preview_albedo;
};

PathTracer::PathTracer(optix::Context &optix_ctx)
{
    impl_ = newBox<Impl>();
    impl_->optix_ctx = &optix_ctx;
}

PathTracer::~PathTracer()
{
    assert(impl_);
}

void PathTracer::set_params(const Params &params)
{
    impl_->params = params;
}

void PathTracer::set_film_filter(RC<FilmFilter> filter)
{
    impl_->filter = std::move(filter);
}

void PathTracer::set_scene(RC<Scene> scene)
{
    impl_->scene = std::move(scene);
}

void PathTracer::set_camera(RC<Camera> camera)
{
    impl_->camera = std::move(camera);
}

void PathTracer::set_film(int width, int height)
{
    impl_->width = width;
    impl_->height = height;
}

void PathTracer::set_reporter(RC<Reporter> reporter)
{
    impl_->reporter = std::move(reporter);
}

std::vector<RC<Object>> PathTracer::get_dependent_objects()
{
    std::set<RC<Object>> scene_objects;
    impl_->scene->collect_objects(scene_objects);
    std::vector<RC<Object>> result = { scene_objects.begin(), scene_objects.end()};
    result.push_back(impl_->camera);
    result.push_back(impl_->filter);
    return result;
}

void PathTracer::recompile()
{
    CompileContext cc;

    auto &params = impl_->params;

    // film

    impl_->film = Film(impl_->width, impl_->height);
    impl_->film.add_output(Film::OUTPUT_RADIANCE, Film::Float3);
    impl_->film.add_output(Film::OUTPUT_WEIGHT, Film::Float);
    if(params.albedo)
        impl_->film.add_output(Film::OUTPUT_ALBEDO, Film::Float3);
    if(params.normal)
        impl_->film.add_output(Film::OUTPUT_NORMAL, Film::Float3);

    // pipeline

    auto raygen = [
        &cc,
        &params,
        &film = impl_->film,
        &filter = impl_->filter,
        &camera = impl_->camera,
        &scene = impl_->scene]
    (const Pipeline::RecordContext &ctx)
    {
        ref launch_params = ctx.launch_params.get_reference();
        u32 pixel_x = optix::get_launch_index_x();
        u32 pixel_y = optix::get_launch_index_y();

        pt::GlobalSampler sampler(film.size(), CVec2u(pixel_x, pixel_y), launch_params.finished_spp);

        // filter importance sampling

        var filter_sample = filter->sample(sampler);
        var pixel_xf = f32(pixel_x) + 0.5f + filter_sample.x;
        var pixel_yf = f32(pixel_y) + 0.5f + filter_sample.y;
        var film_x = pixel_xf / static_cast<float>(film.width());
        var film_y = pixel_yf / static_cast<float>(film.height());
        var time_sample = sampler.get1d();
        auto sample_we_result = camera->generate_ray(cc, CVec2f(film_x, film_y), time_sample);

        // trace

        pt::TraceUtils trace_utils;
        trace_utils.find_closest_intersection = [&](const CRay &r)
            { return ctx.find_closest_intersection(scene->get_tlas(), r); };
        trace_utils.has_intersection = [&](const CRay &r)
            { return ctx.has_intersection(scene->get_tlas(), r); };

        const pt::Params trace_params = {
            .min_depth = params.min_depth,
            .max_depth = params.max_depth,
            .rr_threshold = params.rr_threshold,
            .rr_cont_prob = params.rr_cont_prob,
            .albedo = params.albedo,
            .normal = params.normal
        };

        CRay trace_ray(sample_we_result.pos, sample_we_result.dir);
        auto trace_result = trace_path(cc, trace_utils, trace_params, *scene, trace_ray, sampler);

        var radiance = sample_we_result.throughput * trace_result.radiance;

        // write film

        std::vector<std::pair<std::string_view, Film::CValue>> splat_values;
        splat_values.push_back({ Film::OUTPUT_WEIGHT, Film::CValue(f32(1)) });
        splat_values.push_back({ Film::OUTPUT_RADIANCE, Film::CValue(radiance.to_rgb()) });
        if(params.normal)
            splat_values.push_back({ Film::OUTPUT_NORMAL, Film::CValue(trace_result.normal) });
        if(params.albedo)
            splat_values.push_back({ Film::OUTPUT_ALBEDO, Film::CValue(trace_result.albedo.to_rgb()) });

        film.splat(CVec2u(pixel_x, pixel_y), splat_values);
    };

    impl_->pipeline = Pipeline(*impl_->optix_ctx, raygen, Pipeline::Config{
        .traversal_depth = 2,
        .motion_blur     = impl_->scene->has_motion_blur(),
        .triangle_only   = impl_->scene->is_triangle_only()
    });
}

Renderer::RenderResult PathTracer::render()
{
    impl_->film.clear_output(Film::OUTPUT_RADIANCE);
    impl_->film.clear_output(Film::OUTPUT_WEIGHT);
    if(impl_->film.has_output(Film::OUTPUT_ALBEDO))
        impl_->film.clear_output(Film::OUTPUT_ALBEDO);
    if(impl_->film.has_output(Film::OUTPUT_NORMAL))
        impl_->film.clear_output(Film::OUTPUT_NORMAL);

    auto &reporter = *impl_->reporter;
    reporter.new_stage();

    for(int sample_index = 0; sample_index < impl_->params.spp; ++sample_index)
    {
        const LaunchParams launch_params = {
            .finished_spp = sample_index
        };
        impl_->pipeline.launch(launch_params, impl_->width, impl_->height, 1);

        reporter.progress(100.0f * (sample_index + 1.0f) / impl_->params.spp);

        if(reporter.need_preview())
            new_preview_image();

        if(should_stop())
            break;
    }

    throw_on_error(cudaStreamSynchronize(nullptr));

    reporter.complete_stage();
    if(should_stop())
        return {};

    new_preview_image();

    RenderResult result;
    result.color.swap(impl_->device_preview_image);
    result.albedo.swap(impl_->device_preview_albedo);
    result.normal.swap(impl_->device_preview_normal);
    return result;
}

void PathTracer::update_device_preview_data()
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

void PathTracer::new_preview_image()
{
    update_device_preview_data();
    impl_->reporter->new_preview(
        impl_->device_preview_image.get(),
        impl_->device_preview_albedo.get(),
        impl_->device_preview_normal.get(),
        impl_->width,
        impl_->height);
}

RC<Renderer> PathTracerCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    PathTracer::Params params;
    params.spp          = node->parse_child_or("spp", params.spp);
    params.min_depth    = node->parse_child_or("min_depth", params.min_depth);
    params.max_depth    = node->parse_child_or("max_depth", params.max_depth);
    params.rr_threshold = node->parse_child_or("rr_threshold", params.rr_threshold);
    params.rr_cont_prob = node->parse_child_or("rr_cont_prob", params.rr_cont_prob);
    params.albedo       = node->parse_child_or("albedo", params.albedo);
    params.normal       = node->parse_child_or("normal", params.normal);

    RC<FilmFilter> filter;
    if(auto n = node->find_child_node("filter"))
        filter = context.create<FilmFilter>(n);
    else
        filter = newRC<BoxFilter>();

    auto pt = newRC<PathTracer>(context.get_optix_context());
    pt->set_params(params);
    pt->set_film_filter(std::move(filter));
    return pt;
}

BTRC_BUILTIN_END
