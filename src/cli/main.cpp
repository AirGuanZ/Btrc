#include <any>
#include <chrono>
#include <iostream>

#include <btrc/core/camera/pinhole.h>
#include <btrc/core/light/gradient_sky.h>
#include <btrc/core/light_sampler/uniform_light_sampler.h>
#include <btrc/core/material/diffuse.h>
#include <btrc/core/spectrum/rgb.h>
#include <btrc/core/utils/cuda/context.h>
#include <btrc/core/utils/optix/context.h>
#include <btrc/core/utils/image.h>
#include <btrc/core/wavefront/generate.h>
#include <btrc/core/wavefront/shade.h>
#include <btrc/core/wavefront/shadow.h>
#include <btrc/core/wavefront/sort.h>
#include <btrc/core/wavefront/trace.h>

using namespace btrc::core;

constexpr int SPP = 512;
constexpr int STATE_COUNT = 1000000;

constexpr int WIDTH = 512;
constexpr int HEIGHT = 512;

struct Scene
{
    const SpectrumType *spec_type;

    PinholeCamera                    camera;
    RC<CUDABuffer<wf::InstanceInfo>> instances;
    RC<CUDABuffer<wf::GeometryInfo>> geometries;
    RC<CUDABuffer<int32_t>>          inst_id_to_mat_id;
    std::vector<RC<const Material>>  materials;
    RC<const LightSampler>           light_sampler;

    std::vector<optix::TriangleAS> blas;
    optix::InstanceAS              tlas;

    std::vector<std::any> owned_data;
};

struct Pipeline
{
    wf::GeneratePipeline generate;
    wf::TracePipeline    trace;
    wf::SortPipeline     sort;
    wf::ShadePipeline    shade;
    //wf::ShadowPipeline   shadow;
};

struct SOAState
{
    CUDABuffer<cstd::LCGData> input_rng;

    // generate output

    CUDABuffer<Vec4f> ray_o_t0;
    CUDABuffer<Vec4f> ray_d_t1;
    CUDABuffer<Vec2u> ray_time_mask;
    CUDABuffer<Vec2f> pixel_coord;

    CUDABuffer<float>   beta;
    CUDABuffer<float>   beta_le;
    CUDABuffer<float>   bsdf_pdf;
    CUDABuffer<int32_t> depth;
    CUDABuffer<float>   path_radiance;

    // trace output

    CUDABuffer<float> inct_t;
    CUDABuffer<Vec4u> inct_uv_id;

    // sort output

    CUDABuffer<int32_t> sorted_state_indices;

    // shade output

    CUDABuffer<cstd::LCGData> output_rng;

    CUDABuffer<float>   output_path_radiance;
    CUDABuffer<Vec2f>   output_pixel_coord;
    CUDABuffer<int32_t> output_depth;
    CUDABuffer<float>   output_beta;

    CUDABuffer<Vec2f> shadow_pixel_coord;
    CUDABuffer<Vec4f> shadow_ray_o_t0;
    CUDABuffer<Vec4f> shadow_ray_d_t1;
    CUDABuffer<Vec2u> shadow_ray_time_mask;
    CUDABuffer<float> shadow_beta_li;

    CUDABuffer<Vec4f> output_ray_o_t0;
    CUDABuffer<Vec4f> output_ray_d_t1;
    CUDABuffer<Vec2u> output_ray_time_mask;

    CUDABuffer<float> output_beta_le;
    CUDABuffer<float> output_bsdf_pdf;
};

Scene build_scene(optix::Context &optix_ctx)
{
    Scene scene;

    scene.spec_type = RGBSpectrumType::get_instance();

    scene.camera.set_spectrum_builder(scene.spec_type);
    scene.camera.set_eye({ -2, 0, 0 });
    scene.camera.set_dst({ 0, 0, 0 });
    scene.camera.set_up({ 0, 0, 1 });
    scene.camera.set_fov_y_deg(60);
    scene.camera.set_w_over_h(static_cast<float>(WIDTH) / HEIGHT);

    {
        const wf::InstanceInfo inst = {
            .geometry_id = 0,
            .material_id = 0,
            .light_id    = -1,
            .transform   = Transform{
                .scale     = 1,
                .rotate    = Quaterion({ 1, 0, 0 }, 0),
                .translate = { 0, 0, 0 }
            }
        };
        scene.instances = newRC<CUDABuffer<wf::InstanceInfo>>(1, &inst);
    }

    {
        const std::vector geometry_info_data = {
            Vec4f(0, 1, 0, 0),
            Vec4f(0, 0, -1, 0),
            Vec4f(-1, 0, 0, 1),
            Vec4f(-1, 0, 0, 0),
            Vec4f(-1, 0, 0, 0.5f),
            Vec4f(-1, 0, 0, 0),
        };
        auto vec4_buf = CUDABuffer(std::span(geometry_info_data));

        wf::GeometryInfo geometry;
        geometry.geometry_ex_tex_coord_u_a      = vec4_buf.get();
        geometry.geometry_ey_tex_coord_u_ba     = vec4_buf.get() + 1;
        geometry.geometry_ez_tex_coord_u_ca     = vec4_buf.get() + 2;
        geometry.shading_normals_tex_coord_v_a  = vec4_buf.get() + 3;
        geometry.shading_normals_tex_coord_v_ba = vec4_buf.get() + 4;
        geometry.shading_normals_tex_coord_v_ca = vec4_buf.get() + 5;

        scene.geometries = newRC<CUDABuffer<wf::GeometryInfo>>(1, &geometry);
        scene.owned_data.emplace_back(
            newRC<CUDABuffer<Vec4f>>(std::move(vec4_buf)));

        const int32_t mat_id_data[] = { 0 };
        scene.inst_id_to_mat_id = newRC<CUDABuffer<int32_t>>(1, mat_id_data);
    }
    
    {
        const Vec3f positions[] = {
            { 0, -1, -1 },
            { 0, 0, 1 },
            { 0, 1, -1 }
        };
        auto gas = optix_ctx.create_triangle_as(positions);

        optix::Context::Instance inst = {
            .local_to_world = std::array{
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f
            },
            .id     = 0,
            .mask   = 0xff,
            .handle = gas
        };
        scene.tlas = optix_ctx.create_instance_as(std::span(&inst, 1));
        scene.blas.push_back(std::move(gas));
    }

    {
        auto diffuse = newRC<Diffuse>();
        diffuse->set_albedo(RGBSpectrumImpl::from_rgb(0, 0.8f, 0.8f));
        scene.materials.push_back(std::move(diffuse));
    }

    {
        auto sky = newRC<GradientSky>();
        sky->set_up({ 0, 0, 1 });
        sky->set_lower(RGBSpectrumType::get_instance()->create_zero());
        sky->set_upper(RGBSpectrumType::get_instance()->create_one());

        auto sampler = newRC<UniformLightSampler>();
        sampler->add_light(std::move(sky));

        scene.light_sampler = std::move(sampler);
    }

    return scene;
}

Pipeline build_pipeline(
    optix::Context &optix_ctx, Film &film, const Scene &scene)
{
    Pipeline pipeline;

    pipeline.generate = wf::GeneratePipeline(
        scene.camera, { film.width(), film.height() }, SPP, STATE_COUNT);

    pipeline.trace = wf::TracePipeline(
        optix_ctx, false, true, 2);

    pipeline.sort = wf::SortPipeline();

    pipeline.shade = wf::ShadePipeline(
        film,
        wf::SceneData{
            .instances         = scene.instances,
            .geometries        = scene.geometries,
            .inst_id_to_mat_id = scene.inst_id_to_mat_id,
            .materials         = scene.materials,
            .light_sampler     = scene.light_sampler
        },
        RGBSpectrumType::get_instance(),
        wf::ShadePipeline::ShadeParams{
            .min_depth    = 4,
            .max_depth    = 8,
            .rr_threshold = 0.1f,
            .rr_cont_prob = 0.3f
        });

    //pipeline.shadow = wf::ShadowPipeline(
    //    film, RGBSpectrumType::get_instance(), optix_ctx, false, true, 2);

    return pipeline;
}

SOAState build_soa_state(int state_count, const SpectrumType *spec_type)
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

    const int beta_float_count = state_count * spec_type->get_word_count();
    ret.beta.initialize(beta_float_count);
    ret.beta_le.initialize(beta_float_count);
    ret.bsdf_pdf.initialize(state_count);
    ret.depth.initialize(state_count);
    ret.path_radiance.initialize(beta_float_count);

    ret.inct_t.initialize(state_count);
    ret.inct_uv_id.initialize(state_count);

    ret.sorted_state_indices.initialize(state_count);
    
    ret.output_rng.initialize(state_count);

    ret.output_path_radiance.initialize(beta_float_count);
    ret.output_pixel_coord.initialize(state_count);
    ret.output_depth.initialize(state_count);
    ret.output_beta.initialize(beta_float_count);

    ret.shadow_pixel_coord.initialize(state_count);
    ret.shadow_ray_o_t0.initialize(state_count);
    ret.shadow_ray_d_t1.initialize(state_count);
    ret.shadow_ray_time_mask.initialize(state_count);
    ret.shadow_beta_li.initialize(beta_float_count);

    ret.output_ray_o_t0.initialize(state_count);
    ret.output_ray_d_t1.initialize(state_count);
    ret.output_ray_time_mask.initialize(state_count);

    ret.output_beta_le.initialize(beta_float_count);
    ret.output_bsdf_pdf.initialize(state_count);

    return ret;
}

void run()
{
    cuda::Context cuda_ctx(0);
    optix::Context optix_ctx(cuda_ctx);

    auto scene = build_scene(optix_ctx);

    Film film(512, 512);
    film.add_output(Film::OUTPUT_RADIANCE, Film::Float3);
    film.add_output(Film::OUTPUT_WEIGHT, Film::Float);

    auto pipeline = build_pipeline(optix_ctx, film, scene);

    auto soa = build_soa_state(STATE_COUNT, RGBSpectrumType::get_instance());

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
            scene.tlas,
            active_state_count,
            wf::TracePipeline::SOAParams{
                .ray_o_t0      = soa.ray_o_t0,
                .ray_d_t1      = soa.ray_d_t1,
                .ray_time_mask = soa.ray_time_mask,
                .inct_t        = soa.inct_t,
                .inct_uv_id    = soa.inct_uv_id
            });

        pipeline.sort.sort(
            active_state_count, soa.inct_t, soa.sorted_state_indices);

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
            // TODO
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
    std::vector<float> film_weight(WIDTH * HEIGHT);

    film.get_float3_output(Film::OUTPUT_RADIANCE).to_cpu(film_radiance.data());
    film.get_float_output(Film::OUTPUT_WEIGHT).to_cpu(film_weight.data());

    Image<Vec3f> image(WIDTH, HEIGHT);
    for(int y = 0; y < HEIGHT; ++y)
    {
        for(int x = 0; x < WIDTH; ++x)
        {
            const int i = y * WIDTH + x;
            const Vec3f radiance =
            {
                film_radiance[i * 4],
                film_radiance[i * 4 + 1],
                film_radiance[i * 4 + 2],
            };
            const float weight = film_weight[i];
            image(x, y) = radiance / weight;
        }
    }
    image.pow_(1 / 2.2f);
    image.save("output.png");
}

int main()
{
    try
    {
        run();
    }
    catch(const std::exception &err)
    {
        std::cerr << "fatal error: " << err.what() << std::endl;
        return -1;
    }
}
