#include <chrono>
#include <iostream>

#include <btrc/core/camera/pinhole.h>
#include <btrc/core/spectrum/rgb.h>
#include <btrc/core/utils/cuda/context.h>
#include <btrc/core/utils/optix/context.h>
#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/image.h>
#include <btrc/core/wavefront/generate.h>
#include <btrc/core/wavefront/shade.h>
#include <btrc/core/wavefront/sort.h>
#include <btrc/core/wavefront/trace.h>

using namespace btrc::core;

struct Scene
{
    PinholeCamera                   camera;
    std::vector<wf::InstanceInfo>   instances;
    std::vector<RC<const Material>> materials;
    RC<const LightSampler>          light_sampler;

    std::vector<optix::TriangleAS> blas;
    optix::InstanceAS              tlas;
};

struct Pipeline
{
    wf::GeneratePipeline generate;
    wf::TracePipeline    trace;
    wf::SortPipeline     sort;
    wf::ShadePipeline    shade;
};

Scene build_scene();

void run()
{
    constexpr int WIDTH       = 512;
    constexpr int HEIGHT      = 512;
    constexpr int SPP         = 1;
    constexpr int STATE_COUNT = 1000000;

    cuda::Context cuda_ctx(0);
    optix::Context optix_ctx(cuda_ctx);

    PinholeCamera camera;
    camera.set_spectrum_builder(RGBSpectrumType::get_instance());
    camera.set_eye({ -2, 0, 0 });
    camera.set_dst({ 0, 0, 0 });
    camera.set_up({ 0, 0, 1 });
    camera.set_fov_y_deg(60);
    camera.set_w_over_h(static_cast<float>(WIDTH) / HEIGHT);

    std::cout << "initialize generate pipeline" << std::endl;

    wf::GeneratePipeline generate_pipeline;
    generate_pipeline = wf::GeneratePipeline(
        camera, { WIDTH, HEIGHT}, SPP, STATE_COUNT);

    std::cout << "initialize scene" << std::endl;

    const Vec3f vertices[] = {
        { 0, -1, -1 },
        { 0, 0, 1 },
        { 0, 1, -1 }
    };
    auto gas = optix_ctx.create_triangle_as(vertices);

    std::vector instances = {
        optix::Context::Instance{
            .local_to_world = std::array{
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f
            },
            .id = 0,
            .mask = 0xff,
            .handle = gas
        }
    };
    auto ias = optix_ctx.create_instance_as(instances);

    std::cout << "initialize trace pipeline" << std::endl;

    wf::TracePipeline trace_pipeline(optix_ctx, false, true, 2);

    std::cout << "initialize sort pipeline" << std::endl;

    wf::SortPipeline sort_pipeline;

    std::cout << "create states" << std::endl;

    CUDABuffer<cstd::LCGData> lcgs           (STATE_COUNT);
    CUDABuffer<float2>        ray_pixel_coord(STATE_COUNT);
    CUDABuffer<float4>        ray_o_t0       (STATE_COUNT);
    CUDABuffer<float4>        ray_d_t1       (STATE_COUNT);
    CUDABuffer<uint2>         ray_time_mask  (STATE_COUNT);

    CUDABuffer<float> inct_t(STATE_COUNT);
    CUDABuffer<uint4> inct_uv_id(STATE_COUNT);

    CUDABuffer<int32_t> active_state_indices(STATE_COUNT);

    std::vector<cstd::LCGData> lcg_data(STATE_COUNT);
    for(size_t i = 0; i < lcg_data.size(); ++i)
        lcg_data[i].state = static_cast<uint32_t>(i + 1);
    lcgs.from_cpu(lcg_data.data());

    auto start_time = std::chrono::steady_clock::now();

    std::cout << "generate rays" << std::endl;

    generate_pipeline.clear();
    const int generated_state_count = generate_pipeline.generate(
        0, lcgs, ray_pixel_coord, { ray_o_t0, ray_d_t1, ray_time_mask });

    assert(generate_pipeline.is_done());

    std::cout << "trace rays" << std::endl;

    trace_pipeline.trace(
        ias, generated_state_count,
        { ray_o_t0, ray_d_t1, ray_time_mask, },
        { inct_t, inct_uv_id });

    std::cout << "sort rays" << std::endl;

    const int active_state_count = sort_pipeline.sort(
        generated_state_count, inct_t, active_state_indices);

    std::cout << "initial state count: " << generated_state_count << std::endl;
    std::cout << "active  state count: " << active_state_count << std::endl;

    auto end_time = std::chrono::steady_clock::now();
    auto delta_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(end_time - start_time);
    std::cout << delta_ms.count() << "ms" << std::endl;

    /*{
        std::cout << "save rays" << std::endl;

        std::vector<float4> d_t1(generated_state_count);
        ray_d_t1.to_cpu(d_t1.data(), 0, generated_state_count);

        Image<Vec3f> image(WIDTH, HEIGHT);
        for(int y = 0; y < HEIGHT; ++y)
        {
            for(int x = 0; x < WIDTH; ++x)
            {
                const int i = y * WIDTH + x;
                const float xf = 0.5f * d_t1[i].x + 0.5f;
                const float yf = 0.5f * d_t1[i].y + 0.5f;
                const float zf = 0.5f * d_t1[i].z + 0.5f;
                image(x, y) = Vec3f{ xf, yf, zf };
            }
        }
        image.save("dir.png");
        std::cout << "result written to dir.png" << std::endl;
    }

    {
        std::cout << "save incts" << std::endl;

        std::vector<float> d_t(generated_state_count);
        inct_t.to_cpu(d_t.data(), 0, generated_state_count);

        Image<Vec3f> image(WIDTH, HEIGHT);
        for(int y = 0; y < HEIGHT; ++y)
        {
            for(int x = 0; x < WIDTH; ++x)
            {
                const int i = y * WIDTH + x;
                const float t = d_t[i];
                const float v = t >= 0 ? 1.0f : 0.0f;
                image(x, y) = Vec3f{ v, v, v };
            }
        }
        image.save("inct.png");
        std::cout << "result written to inct.png" << std::endl;
    }*/
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
