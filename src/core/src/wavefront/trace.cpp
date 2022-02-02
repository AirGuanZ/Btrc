#include <btrc/core/utils/cmath/cmath.h>
#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/cuda/error.h>
#include <btrc/core/utils/optix/device_funcs.h>
#include <btrc/core/wavefront/trace.h>

BTRC_WAVEFRONT_BEGIN

namespace
{

    const char *LAUNCH_PARAMS_NAME = "launch_params";

    const char *RAYGEN_TRACE_NAME     = "__raygen__trace";
    const char *MISS_TRACE_NAME       = "__miss__trace";
    const char *CLOSESTHIT_TRACE_NAME = "__closesthit__trace";

    std::string generate_trace_kernel()
    {
        using namespace cuj;

        ScopedModule cuj_module;

        auto global_launch_params = allocate_constant_memory<
            TracePipeline::CLaunchParams>(LAUNCH_PARAMS_NAME);

        kernel(
            RAYGEN_TRACE_NAME,
            [global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_launch_index_x();

            var o_t0 = load_aligned(launch_params.ray_o_t0 + launch_idx);
            var d_t1 = load_aligned(launch_params.ray_d_t1 + launch_idx);
            var time_mask = load_aligned(launch_params.ray_time_mask + launch_idx);

            var o = o_t0.xyz();
            var d = d_t1.xyz();
            var t0 = o_t0.w;
            var t1 = d_t1.w;
            var time = bitcast<f32>(time_mask.x);
            var mask = time_mask.y;

            optix::trace(
                launch_params.handle,
                o, d, t0, t1, time, mask, OPTIX_RAY_FLAG_NONE,
                0, 1, 0, launch_idx);
        });

        kernel(
            MISS_TRACE_NAME,
            [global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_payload(0);
            launch_params.inct_t[launch_idx] = -1;
        });

        kernel(
            CLOSESTHIT_TRACE_NAME,
            [global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_payload(0);
            
            var t = optix::get_ray_tmax();
            var uv = optix::get_triangle_barycentrics();
            var prim_id = optix::get_primitive_index();
            var inst_id = optix::get_instance_id();

            launch_params.inct_t[launch_idx] = t;

            var uv_id = CVec4u(bitcast<u32>(uv.x), bitcast<u32>(uv.y), prim_id, inst_id);
            save_aligned(uv_id, launch_params.inct_uv_id + launch_idx);
        });

        PTXGenerator gen;
        gen.set_options(Options{
            .opt_level        = OptimizationLevel::O3,
            .fast_math        = true,
            .approx_math_func = true
        });
        gen.generate(cuj_module);

        return gen.get_ptx();
    }

} // namespace anonymous

TracePipeline::TracePipeline(
    OptixDeviceContext context,
    bool               motion_blur,
    bool               triangle_only,
    int                traversable_depth)
    : TracePipeline()
{
    initialize(context, motion_blur, triangle_only, traversable_depth);
}

TracePipeline::TracePipeline(TracePipeline &&other) noexcept
    : TracePipeline()
{
    swap(other);
}

TracePipeline &TracePipeline::operator=(TracePipeline &&other) noexcept
{
    swap(other);
    return *this;
}

TracePipeline::operator bool() const
{
    return pipeline_ != nullptr;
}

void TracePipeline::swap(TracePipeline &other) noexcept
{
    pipeline_.swap(other.pipeline_);
    device_launch_params_.swap(other.device_launch_params_);
}

void TracePipeline::trace(
    OptixTraversableHandle handle,
    int active_state_count,
    const SOAParams &soa_params) const
{
    const LaunchParams launch_params = {
        .handle        = handle,
        .ray_o_t0      = soa_params.ray_o_t0,
        .ray_d_t1      = soa_params.ray_d_t1,
        .ray_time_mask = soa_params.ray_time_mask,
        .inct_t        = soa_params.inct_t,
        .inct_uv_id    = soa_params.inct_uv_id
    };
    device_launch_params_.from_cpu(&launch_params);
    throw_on_error(optixLaunch(
        pipeline_, nullptr,
        device_launch_params_, sizeof(LaunchParams),
        &pipeline_.get_sbt(), active_state_count, 1, 1));
}

void TracePipeline::initialize(
    OptixDeviceContext context,
    bool               motion_blur,
    bool               triangle_only,
    int                traversable_depth)
{
    pipeline_ = optix::SimpleOptixPipeline(
        context,
        optix::SimpleOptixPipeline::Program{
            .ptx                = generate_trace_kernel(),
            .launch_params_name = LAUNCH_PARAMS_NAME,
            .raygen_name        = RAYGEN_TRACE_NAME,
            .miss_name          = MISS_TRACE_NAME,
            .closesthit_name    = CLOSESTHIT_TRACE_NAME
        },
        optix::SimpleOptixPipeline::Config{
            .payload_count     = 1,
            .traversable_depth = traversable_depth,
            .motion_blur       = motion_blur,
            .triangle_only     = triangle_only
        });
    device_launch_params_ = CUDABuffer<LaunchParams>(1);
}

BTRC_WAVEFRONT_END
