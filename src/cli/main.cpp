#include <iostream>
#include <fstream>

#include <btrc/core/camera/pinhole.h>
#include <btrc/core/utils/cuda/context.h>
#include <btrc/core/utils/optix/context.h>
#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/wavefront/generate.h>
#include <btrc/core/wavefront/trace.h>

using namespace btrc::core;

void run()
{
    constexpr int WIDTH       = 512;
    constexpr int HEIGHT      = 512;
    constexpr int SPP         = 1;
    constexpr int STATE_COUNT = 1000000;

    cuda::Context cuda_ctx(0);
    optix::Context optix_ctx(cuda_ctx);

    PinholeCamera camera;
    camera.set_eye({ -2, 0, 0 });
    camera.set_dst({ 0, 0, 0 });
    camera.set_up({ 0, 0, 1 });
    camera.set_fov_y_deg(60);
    camera.set_w_over_h(static_cast<float>(WIDTH) / HEIGHT);
    camera.preprocess();

    wf::GeneratePipeline generate_pipeline;
    generate_pipeline = wf::GeneratePipeline(
        camera, { WIDTH, HEIGHT}, SPP, STATE_COUNT);

    CUDABuffer<cstd::LCGData> lcgs             (STATE_COUNT);
    CUDABuffer<float2>        ray_pixel_coord  (STATE_COUNT);
    CUDABuffer<float4>        ray_o_t0         (STATE_COUNT);
    CUDABuffer<float4>        ray_d_t1         (STATE_COUNT);
    CUDABuffer<uint2>         ray_time_mask    (STATE_COUNT);
    CUDABuffer<float4>        throughput_weight(STATE_COUNT);

    std::vector<cstd::LCGData> lcg_data(STATE_COUNT);
    for(size_t i = 0; i < lcg_data.size(); ++i)
        lcg_data[i].state = static_cast<uint32_t>(i + 1);
    lcgs.from_cpu(lcg_data.data());

    while(!generate_pipeline.is_done())
    {
        const int state_count = generate_pipeline.generate(
            0, lcgs, ray_pixel_coord, ray_o_t0, ray_d_t1, ray_time_mask);

        assert(generate_pipeline.is_done());
        throw_on_error(cudaStreamSynchronize(nullptr));

        std::vector<float4> d_t1(state_count);
        ray_d_t1.to_cpu(d_t1.data(), 0, state_count);

        std::ofstream fout("output.ppm");
        if(!fout)
        {
            throw std::runtime_error(
                "failed to create output image: output.ppm");
        }

        fout << "P3\n" << WIDTH << " " << HEIGHT << std::endl << 255 << std::endl;
        for(int i = 0; i < WIDTH * HEIGHT; ++i)
        {
            const float xf = 0.5f * d_t1[i].x + 0.5f;
            const float yf = 0.5f * d_t1[i].y + 0.5f;
            const float zf = 0.5f * d_t1[i].z + 0.5f;

            const int ri = (std::min)(255, static_cast<int>(xf * 255));
            const int gi = (std::min)(255, static_cast<int>(yf * 255));
            const int bi = (std::min)(255, static_cast<int>(zf * 255));

            fout << ri << " " << gi << " " << bi << " ";
        }

        fout.close();
        std::cout << "result written to output.ppm" << std::endl;
    }

    /*const Vec3f vertices[] = {
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 }
    };
    auto gas = optix_ctx.create_triangle_as(vertices);

    std::vector instances = {
        optix::Context::Instance{
            .local_to_world = std::array{
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f
            },
            .id             = 0,
            .mask           = 0xff,
            .handle         = gas
        }
    };
    auto ias = optix_ctx.create_instance_as(instances);

    wf::TracePipeline trace_pipeline;
    trace_pipeline.initialize(optix_ctx, false, true, 2);*/
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
