#include <fstream>

#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/cuda/buffer.h>
#include <btrc/utils/cuda/error.h>
#include <btrc/utils/file.h>
#include <btrc/utils/optix/device_funcs.h>
#include <btrc/utils/ptx_cache.h>

#include "./trace.h"

BTRC_WFPT_BEGIN

namespace
{

    const char *LAUNCH_PARAMS_NAME = "launch_params";

    const char *RAYGEN_TRACE_NAME     = "__raygen__trace";
    const char *MISS_TRACE_NAME       = "__miss__trace";
    const char *CLOSESTHIT_TRACE_NAME = "__closesthit__trace";

    const char *KERNEL_CACHE_RELATIVE_PATH = "./.btrc_cache/wfpt_trace.ptx";

    std::string generate_trace_kernel()
    {
        using namespace cuj;

        const std::string cache_filename = (get_executable_filename().parent_path() / KERNEL_CACHE_RELATIVE_PATH).string();

        auto cached_ptx = load_kernel_cache(cache_filename);
        if(!cached_ptx.empty())
            return cached_ptx;

        ScopedModule cuj_module;

        auto global_launch_params = allocate_constant_memory<
            TracePipeline::CLaunchParams>(LAUNCH_PARAMS_NAME);

        kernel(
            RAYGEN_TRACE_NAME,
            [global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_launch_index_x();

            var o_medium_id = load_aligned(launch_params.ray_o_medium_id + launch_idx);
            var d_t1 = load_aligned(launch_params.ray_d_t1 + launch_idx);
            var time_mask = load_aligned(launch_params.ray_time_mask + launch_idx);

            var o = o_medium_id.xyz();
            var d = d_t1.xyz();
            var t1 = d_t1.w;
            var time = bitcast<f32>(time_mask.x);
            var mask = time_mask.y;

            optix::trace(
                launch_params.handle,
                o, d, 0, t1, time, mask, OPTIX_RAY_FLAG_NONE,
                0, 1, 0, launch_idx);
        });

        kernel(
            MISS_TRACE_NAME,
            [global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_payload(0);
            var inct_inst_launch_index = CVec2u(INST_ID_MISS_MASK, launch_idx);
            save_aligned(inct_inst_launch_index, launch_params.inct_inst_launch_index + launch_idx);
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

            var inct_inst_launch_index = CVec2u(inst_id, launch_idx);
            var inct_t_prim_uv = CVec4u(bitcast<u32>(t), prim_id, bitcast<u32>(uv.x), bitcast<u32>(uv.y));

            save_aligned(inct_inst_launch_index, launch_params.inct_inst_launch_index + launch_idx);
            save_aligned(inct_t_prim_uv, launch_params.inct_t_prim_uv + launch_idx);
        });

        PTXGenerator gen;
        gen.set_options(Options{
            .opt_level        = OptimizationLevel::O3,
            .fast_math        = true,
            .approx_math_func = true
        });
        gen.generate(cuj_module);

        create_kernel_cache(cache_filename, gen.get_ptx());
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
        .handle                 = handle,
        .ray_o_medium_id        = soa_params.ray_o_medium_id,
        .ray_d_t1               = soa_params.ray_d_t1,
        .ray_time_mask          = soa_params.ray_time_mask,
        .inct_inst_launch_index = soa_params.inct_inst_launch_index,
        .inct_t_prim_uv         = soa_params.inct_t_prim_uv
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
    device_launch_params_ = cuda::Buffer<LaunchParams>(1);
}

BTRC_WFPT_END
