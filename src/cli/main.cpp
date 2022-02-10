#include <chrono>
#include <iostream>

#include <btrc/core/camera/pinhole.h>
#include <btrc/core/common/context.h>
#include <btrc/core/geometry/triangle_mesh.h>
#include <btrc/core/light/gradient_sky.h>
#include <btrc/core/light/mesh_light.h>
#include <btrc/core/material/black.h>
#include <btrc/core/material/diffuse.h>
#include <btrc/core/material/glass.h>
#include <btrc/core/scene/scene.h>
#include <btrc/core/texture2d/constant2d.h>
#include <btrc/core/utils/cuda/context.h>
#include <btrc/core/utils/optix/context.h>
#include <btrc/core/utils/image.h>
#include <btrc/core/wavefront/generate.h>
#include <btrc/core/wavefront/shade.h>
#include <btrc/core/wavefront/shadow.h>
#include <btrc/core/wavefront/sort.h>
#include <btrc/core/wavefront/trace.h>

using namespace btrc::core;

constexpr int SPP = 128;
constexpr int STATE_COUNT = 2000000;

constexpr int WIDTH = 512;
constexpr int HEIGHT = 512;

struct Pipeline
{
    wf::GeneratePipeline generate;
    wf::TracePipeline    trace;
    wf::SortPipeline     sort;
    wf::ShadePipeline    shade;
    wf::ShadowPipeline   shadow;
};

struct SOAState
{
    CUDABuffer<cstd::LCGData> input_rng;

    // generate output

    CUDABuffer<Vec4f> ray_o_t0;
    CUDABuffer<Vec4f> ray_d_t1;
    CUDABuffer<Vec2u> ray_time_mask;
    CUDABuffer<Vec2f> pixel_coord;

    CUDABuffer<Spectrum> beta;
    CUDABuffer<Spectrum> beta_le;
    CUDABuffer<float>    bsdf_pdf;
    CUDABuffer<int32_t>  depth;
    CUDABuffer<Spectrum> path_radiance;

    // trace output

    CUDABuffer<float> inct_t;
    CUDABuffer<Vec4u> inct_uv_id;

    // sort output

    CUDABuffer<int32_t> sorted_state_indices;

    // shade output

    CUDABuffer<cstd::LCGData> output_rng;

    CUDABuffer<Spectrum> output_path_radiance;
    CUDABuffer<Vec2f>    output_pixel_coord;
    CUDABuffer<int32_t>  output_depth;
    CUDABuffer<Spectrum> output_beta;

    CUDABuffer<Vec2f>    shadow_pixel_coord;
    CUDABuffer<Vec4f>    shadow_ray_o_t0;
    CUDABuffer<Vec4f>    shadow_ray_d_t1;
    CUDABuffer<Vec2u>    shadow_ray_time_mask;
    CUDABuffer<Spectrum> shadow_beta_li;

    CUDABuffer<Vec4f> output_ray_o_t0;
    CUDABuffer<Vec4f> output_ray_d_t1;
    CUDABuffer<Vec2u> output_ray_time_mask;

    CUDABuffer<Spectrum> output_beta_le;
    CUDABuffer<float>    output_bsdf_pdf;
};

Scene build_scene(optix::Context &optix_ctx)
{
    Scene scene;

    auto camera = newRC<PinholeCamera>();
    camera->set_eye({ 0, 0, 1.5f });
    camera->set_dst({ 0, 0, 0 });
    camera->set_up({ 0, 1, 0 });
    camera->set_fov_y_deg(60);
    camera->set_w_over_h(static_cast<float>(WIDTH) / HEIGHT);
    scene.set_camera(std::move(camera));
    
    auto glass = newRC<Glass>();
    glass->set_color(Spectrum::from_rgb(0.6f, 0.9f, 0.9f));
    glass->set_ior(1.43f);
    auto teapot = newRC<TriangleMesh>(optix_ctx, "./asset/teapot.obj", true);
    scene.add_instance(
        Scene::Instance{
            .geometry = std::move(teapot),
            .material = std::move(glass),
            .transform = Transform{
                .translate = { 0, -0.29f, 0 },
                .scale = 0.5f,
                .rotate = Quaterion({ 1, 0, 0 }, 0)
            }
        });

    auto black = newRC<Black>();
    auto box = newRC<TriangleMesh>(optix_ctx, "./asset/box.obj", true);
    auto box_trans = Transform{
        .translate = { 0, 0.1f, 0 },
        .scale = 0.1f,
        .rotate = Quaterion({ 1, 1, 1 }, 2.7f)
    };
    auto light = newRC<MeshLight>(box, box_trans, 8 * Spectrum::from_rgb(1, 1, 1));
    scene.add_instance(
        Scene::Instance{
            .geometry = std::move(box),
            .material = std::move(black),
            .light = std::move(light),
            .transform = box_trans
        });

    auto diffuse_albedo = newRC<Constant2D>();
    diffuse_albedo->set_value(0.8f);
    auto diffuse = newRC<Diffuse>();
    diffuse->set_albedo(std::move(diffuse_albedo));
    auto cbox = newRC<TriangleMesh>(optix_ctx, "./asset/cbox.obj", true);
    scene.add_instance(
        Scene::Instance{
            .geometry = std::move(cbox),
            .material = std::move(diffuse),
            .transform = Transform{
                .translate = {},
                .scale = 1,
                .rotate = Quaterion({ 1, 0, 0 }, 0)
            }
        });

    //auto sky = newRC<GradientSky>();
    //sky->set_up({ 0, 1, 0 });
    //sky->set_lower(Spectrum::zero());
    //sky->set_upper(Spectrum::one());
    //scene.set_envir_light(std::move(sky));

    scene.preprocess(optix_ctx);
    return scene;
}

Pipeline build_pipeline(
    optix::Context &optix_ctx, Film &film, const Scene &scene)
{
    Pipeline pipeline;

    pipeline.generate = wf::GeneratePipeline(
        *scene.get_camera(), { film.width(), film.height() }, SPP, STATE_COUNT);

    pipeline.trace = wf::TracePipeline(optix_ctx, false, true, 2);

    pipeline.sort = wf::SortPipeline();

    pipeline.shade = wf::ShadePipeline(
        film,
        scene,
        wf::ShadePipeline::ShadeParams{
            .min_depth    = 100,
            .max_depth    = 100,
            .rr_threshold = 0.1f,
            .rr_cont_prob = 0.5f
        });

    pipeline.shadow = wf::ShadowPipeline(
        film, optix_ctx, false, true, 2);

    return pipeline;
}

SOAState build_soa_state(int state_count)
{
    SOAState ret;
    ret.input_rng.initialize(state_count);
    {
        std::vector<cstd::LCGData> rng_data(state_count);
        for(int i = 0; i < state_count; ++i)
            rng_data[i].state = static_cast<uint32_t>(i + 1);
        ret.input_rng.from_cpu(rng_data.data());
    }

    ret.ray_o_t0.initialize(state_count);
    ret.ray_d_t1.initialize(state_count);
    ret.ray_time_mask.initialize(state_count);
    ret.pixel_coord.initialize(state_count);

    ret.beta.initialize(state_count);
    ret.beta_le.initialize(state_count);
    ret.bsdf_pdf.initialize(state_count);
    ret.depth.initialize(state_count);
    ret.path_radiance.initialize(state_count);

    ret.inct_t.initialize(state_count);
    ret.inct_uv_id.initialize(state_count);

    ret.sorted_state_indices.initialize(state_count);
    
    ret.output_rng.initialize(state_count);

    ret.output_path_radiance.initialize(state_count);
    ret.output_pixel_coord.initialize(state_count);
    ret.output_depth.initialize(state_count);
    ret.output_beta.initialize(state_count);

    ret.shadow_pixel_coord.initialize(state_count);
    ret.shadow_ray_o_t0.initialize(state_count);
    ret.shadow_ray_d_t1.initialize(state_count);
    ret.shadow_ray_time_mask.initialize(state_count);
    ret.shadow_beta_li.initialize(state_count);

    ret.output_ray_o_t0.initialize(state_count);
    ret.output_ray_d_t1.initialize(state_count);
    ret.output_ray_time_mask.initialize(state_count);

    ret.output_beta_le.initialize(state_count);
    ret.output_bsdf_pdf.initialize(state_count);

    return ret;
}

void run()
{
    cuda::Context cuda_ctx(0);
    optix::Context optix_ctx(cuda_ctx);

    CompileContext cc;
    CompileContext::push_context(&cc);
    BTRC_SCOPE_EXIT{ CompileContext::pop_context(); };

    auto scene = build_scene(optix_ctx);

    Film film(WIDTH, HEIGHT);
    film.add_output(Film::OUTPUT_RADIANCE, Film::Float3);
    film.add_output(Film::OUTPUT_ALBEDO,   Film::Float3);
    film.add_output(Film::OUTPUT_NORMAL,   Film::Float3);
    film.add_output(Film::OUTPUT_WEIGHT,   Film::Float);

    auto pipeline = build_pipeline(optix_ctx, film, scene);

    auto soa = build_soa_state(STATE_COUNT);

    int active_state_count = 0;
    while(!pipeline.generate.is_done() || active_state_count > 0)
    {
        const int generated_state_count = pipeline.generate.generate(
            active_state_count,
            wf::GeneratePipeline::SOAParams{
                .rng                  = soa.input_rng,
                .output_pixel_coord   = soa.pixel_coord,
                .output_ray_o_t0      = soa.ray_o_t0,
                .output_ray_d_t1      = soa.ray_d_t1,
                .output_ray_time_mask = soa.ray_time_mask,
                .output_beta          = soa.beta,
                .output_beta_le       = soa.beta_le,
                .output_bsdf_pdf      = soa.bsdf_pdf,
                .output_depth         = soa.depth,
                .output_path_radiance = soa.path_radiance
            });

        active_state_count += generated_state_count;

        pipeline.trace.trace(
            scene.get_tlas(),
            active_state_count,
            wf::TracePipeline::SOAParams{
                .ray_o_t0      = soa.ray_o_t0,
                .ray_d_t1      = soa.ray_d_t1,
                .ray_time_mask = soa.ray_time_mask,
                .inct_t        = soa.inct_t,
                .inct_uv_id    = soa.inct_uv_id,
                .state_index   = soa.sorted_state_indices
            });

        //pipeline.sort.sort(
        //    active_state_count, soa.inct_t, soa.inct_uv_id, soa.sorted_state_indices);

        const auto shade_counters = pipeline.shade.shade(
            active_state_count,
            wf::ShadePipeline::SOAParams{
                .rng                         = soa.input_rng,
                .active_state_indices        = soa.sorted_state_indices,
                .path_radiance               = soa.path_radiance,
                .pixel_coord                 = soa.pixel_coord,
                .depth                       = soa.depth,
                .beta                        = soa.beta,
                .beta_le                     = soa.beta_le,
                .bsdf_pdf                    = soa.bsdf_pdf,
                .inct_t                      = soa.inct_t,
                .inct_uv_id                  = soa.inct_uv_id,
                .ray_o_t0                    = soa.ray_o_t0,
                .ray_d_t1                    = soa.ray_d_t1,
                .ray_time_mask               = soa.ray_time_mask,
                .output_rng                  = soa.output_rng,
                .output_path_radiance        = soa.output_path_radiance,
                .output_pixel_coord          = soa.output_pixel_coord,
                .output_depth                = soa.output_depth,
                .output_beta                 = soa.output_beta,
                .output_shadow_pixel_coord   = soa.shadow_pixel_coord,
                .output_shadow_ray_o_t0      = soa.shadow_ray_o_t0,
                .output_shadow_ray_d_t1      = soa.shadow_ray_d_t1,
                .output_shadow_ray_time_mask = soa.shadow_ray_time_mask,
                .output_shadow_beta_li       = soa.shadow_beta_li,
                .output_new_ray_o_t0         = soa.output_ray_o_t0,
                .output_new_ray_d_t1         = soa.output_ray_d_t1,
                .output_new_ray_time_mask    = soa.output_ray_time_mask,
                .output_beta_le              = soa.output_beta_le,
                .output_bsdf_pdf             = soa.output_bsdf_pdf
            });

        active_state_count = shade_counters.active_state_counter;

        if(shade_counters.shadow_ray_counter)
        {
            pipeline.shadow.test(
                scene.get_tlas(),
                shade_counters.shadow_ray_counter,
                wf::ShadowPipeline::SOAParams{
                    .pixel_coord   = soa.shadow_pixel_coord,
                    .ray_o_t0      = soa.shadow_ray_o_t0,
                    .ray_d_t1      = soa.shadow_ray_d_t1,
                    .ray_time_mask = soa.shadow_ray_time_mask,
                    .beta_li       = soa.shadow_beta_li
                });
        }

        soa.input_rng.swap(soa.output_rng);

        soa.ray_o_t0.swap(soa.output_ray_o_t0);
        soa.ray_d_t1.swap(soa.output_ray_d_t1);
        soa.ray_time_mask.swap(soa.output_ray_time_mask);
        soa.pixel_coord.swap(soa.output_pixel_coord);

        soa.beta.swap(soa.output_beta);
        soa.beta_le.swap(soa.output_beta_le);
        soa.bsdf_pdf.swap(soa.output_bsdf_pdf);
        soa.depth.swap(soa.output_depth);
        soa.path_radiance.swap(soa.output_path_radiance);
    }

    throw_on_error(cudaStreamSynchronize(nullptr));

    std::vector<float> film_radiance(WIDTH * HEIGHT * 4);
    std::vector<float> film_albedo  (WIDTH * HEIGHT * 4);
    std::vector<float> film_normal  (WIDTH * HEIGHT * 4);
    std::vector<float> film_weight  (WIDTH * HEIGHT);

    film.get_float3_output(Film::OUTPUT_RADIANCE).to_cpu(film_radiance.data());
    film.get_float3_output(Film::OUTPUT_ALBEDO)  .to_cpu(film_albedo.data());
    film.get_float3_output(Film::OUTPUT_NORMAL)  .to_cpu(film_normal.data());
    film.get_float_output (Film::OUTPUT_WEIGHT)  .to_cpu(film_weight.data());

    Image<Vec3f> image_radiance(WIDTH, HEIGHT);
    Image<Vec3f> image_albedo  (WIDTH, HEIGHT);
    Image<Vec3f> image_normal  (WIDTH, HEIGHT);
    for(int y = 0; y < HEIGHT; ++y)
    {
        for(int x = 0; x < WIDTH; ++x)
        {
            const int i = y * WIDTH + x;
            const float weight = film_weight[i];
            if(weight > 0)
            {
                const Vec3f radiance =
                {
                    film_radiance[i * 4],
                    film_radiance[i * 4 + 1],
                    film_radiance[i * 4 + 2],
                };
                const Vec3f albedo =
                {
                    film_albedo[i * 4],
                    film_albedo[i * 4 + 1],
                    film_albedo[i * 4 + 2],
                };
                const Vec3f normal =
                {
                    film_normal[i * 4],
                    film_normal[i * 4 + 1],
                    film_normal[i * 4 + 2],
                };
                image_radiance(x, y) = radiance / weight;
                image_normal(x, y) = (0.5f + 0.5f * normal / weight);
                image_albedo(x, y) = albedo / weight;
            }
        }
    }

    //image_radiance.pow_(1 / 2.2f);
    image_radiance.save("output.exr");

    image_albedo.pow_(1 / 2.2f);
    image_albedo.save("output_albedo.png");

    image_normal.save("output_normal.png");
}

int main()
{
    run();
    try
    {
    }
    catch(const std::exception &err)
    {
        std::cerr << "fatal error: " << err.what() << std::endl;
        return -1;
    }
}
