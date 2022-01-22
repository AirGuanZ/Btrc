#include <array>
#include <cassert>
#include <vector>

#include <btrc/core/wavefront/launch_params.h>
#include <btrc/core/wavefront/trace.h>
#include <btrc/core/utils/cmath/cmath.h>
#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/cuda/error.h>
#include <btrc/core/utils/optix/device_funcs.h>
#include <btrc/core/utils/scope_guard.h>

//#include <embed_ptx/trace.inl>

#include <optix_stubs.h>

BTRC_WAVEFRONT_BEGIN

namespace
{

    struct TraceLaunchParams
    {
        uint64_t handle;

        Vec4f *ray_o_t0;
        Vec4f *ray_d_t1;
        Vec2u *ray_time_mask;

        float *inct_t;
        Vec4u *inct_uv_id;
    };

    CUJ_PROXY_CLASS(
        CTraceLaunchParams, TraceLaunchParams,
        handle, ray_o_t0, ray_d_t1, ray_time_mask, inct_t, inct_uv_id);

    const char *LAUNCH_PARAMS_NAME = "launch_params";

    const char *RAYGEN_TRACE_NAME     = "__raygen__trace";
    const char *MISS_TRACE_NAME       = "__miss__trace";
    const char *CLOSESTHIT_TRACE_NAME = "__closesthit__trace";

    std::string generate_trace_kernel()
    {
        using namespace cuj;

        ScopedModule cuj_module;

        auto global_launch_params =
            allocate_constant_memory<CTraceLaunchParams>(LAUNCH_PARAMS_NAME);

        kernel(
            RAYGEN_TRACE_NAME,
            [global_launch_params]
        {
            ref launch_params = global_launch_params.get_reference();
            var launch_idx = optix::get_launch_index_x();

            var o_t0 = launch_params.ray_o_t0[launch_idx];
            var d_t1 = launch_params.ray_d_t1[launch_idx];
            var time_mask = launch_params.ray_time_mask[launch_idx];

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
            launch_params.inct_uv_id[launch_idx] = CVec4u(
                bitcast<u32>(uv.x),
                bitcast<u32>(uv.y),
                prim_id, inst_id);
        });

        Options opts;
        opts.approx_math_func = true;
        opts.fast_math = true;
        opts.opt_level = OptimizationLevel::O3;

        PTXGenerator gen;
        gen.set_options(opts);
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

TracePipeline::~TracePipeline()
{
    if(!pipeline_)
    {
        assert(!module_ && !raygen_group_ && !miss_group_ && !hit_group_);
        return;
    }
    assert(module_ && raygen_group_ && miss_group_ && hit_group_);
    optixPipelineDestroy(pipeline_);
    optixProgramGroupDestroy(raygen_group_);
    optixProgramGroupDestroy(miss_group_);
    optixProgramGroupDestroy(hit_group_);
    optixModuleDestroy(module_);
    sbt_ = {};
}

TracePipeline::operator bool() const
{
    return pipeline_ != nullptr;
}

void TracePipeline::swap(TracePipeline &other) noexcept
{
    std::swap(module_,       other.module_);
    std::swap(pipeline_,     other.pipeline_);
    std::swap(raygen_group_, other.raygen_group_);
    std::swap(miss_group_,   other.miss_group_);
    std::swap(hit_group_,    other.hit_group_);
    std::swap(sbt_,          other.sbt_);
}

void TracePipeline::trace(
    OptixTraversableHandle traversable,
    int                    active_state_count,
    const RaySOA          &input_ray,
    const IntersectionSOA &output_inct) const
{
    const TraceParams launch_params = {
        .handle        = traversable,
        .ray_o_t0      = input_ray.o_t0,
        .ray_d_t1      = input_ray.d_t1,
        .ray_time_mask = input_ray.time_mask,
        .inct_t        = output_inct.t,
        .inct_uv_id    = output_inct.uv_id
    };
    CUDABuffer device_launch_params(1, &launch_params);
    throw_on_error(optixLaunch(
        pipeline_, nullptr,
        device_launch_params, sizeof(TraceParams),
        &sbt_.get_table(), active_state_count, 1, 1));
}

void TracePipeline::initialize(
    OptixDeviceContext context,
    bool               motion_blur,
    bool               triangle_only,
    int                traversable_depth)
{
    assert(!module_ && !pipeline_);
    assert(!raygen_group_ && !miss_group_ && !hit_group_);

    // create module

/*#if BTRC_IS_DEBUG
    constexpr OptixCompileOptimizationLevel opt_level =
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    constexpr OptixCompileDebugLevel debug_level =
        OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    constexpr unsigned int exception_flag =
        OPTIX_EXCEPTION_FLAG_DEBUG |
        OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else*/
    constexpr OptixCompileOptimizationLevel opt_level =
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    constexpr OptixCompileDebugLevel debug_level =
        OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    constexpr unsigned int exception_flag = OPTIX_EXCEPTION_FLAG_NONE;
//#endif

    const OptixModuleCompileOptions module_compile_options = {
        .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        .optLevel         = opt_level,
        .debugLevel       = debug_level,
        .boundValues      = nullptr,
        .numBoundValues   = 0
    };

    const unsigned int graph_flag = motion_blur ?
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY :
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    const unsigned int primitive_type_flag =
        triangle_only ? OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE : 0;

    const OptixPipelineCompileOptions pipeline_compile_options = {
        .usesMotionBlur                   = motion_blur,
        .traversableGraphFlags            = graph_flag,
        .numPayloadValues                 = 1,
        .numAttributeValues               = 0,
        .exceptionFlags                   = exception_flag,
        .pipelineLaunchParamsVariableName = LAUNCH_PARAMS_NAME,
        .usesPrimitiveTypeFlags           = primitive_type_flag
    };

    const std::string ptx = generate_trace_kernel();
    std::vector<char> log(2048); size_t log_len = log.size();
    OptixResult create_result = optixModuleCreateFromPTX(
        context, &module_compile_options, &pipeline_compile_options,
        ptx.data(), ptx.size(),
        log.data(), &log_len, &module_);
    if(create_result != OPTIX_SUCCESS)
    {
        module_ = nullptr;
        throw BtrcException(log.data());
    }
    BTRC_SCOPE_FAIL{
        optixModuleDestroy(module_);
        module_ = nullptr;
    };

    // raygen group

    const OptixProgramGroupOptions group_options = {};

    const OptixProgramGroupDesc raygen_group_desc = {
        .kind   = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        .raygen = OptixProgramGroupSingleModule{
            .module            = module_,
            .entryFunctionName = RAYGEN_TRACE_NAME
        }
    };
    log_len = log.size();
    create_result = optixProgramGroupCreate(
        context, &raygen_group_desc, 1, &group_options,
        log.data(), &log_len, &raygen_group_);
    if(create_result != OPTIX_SUCCESS)
        throw BtrcException(log.data());
    BTRC_SCOPE_FAIL{
        optixProgramGroupDestroy(raygen_group_);
        raygen_group_ = nullptr;
    };

    // miss group

    const OptixProgramGroupDesc miss_group_desc = {
        .kind   = OPTIX_PROGRAM_GROUP_KIND_MISS,
        .miss   = OptixProgramGroupSingleModule{
            .module            = module_,
            .entryFunctionName = MISS_TRACE_NAME
        }
    };
    log_len = log.size();
    create_result = optixProgramGroupCreate(
        context, &miss_group_desc, 1, &group_options,
        log.data(), &log_len, &miss_group_);
    if(create_result != OPTIX_SUCCESS)
        throw BtrcException(log.data());
    BTRC_SCOPE_FAIL{
        optixProgramGroupDestroy(miss_group_);
        miss_group_ = nullptr;
    };

    // hit group

    const OptixProgramGroupDesc hit_group_desc = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        .hitgroup = OptixProgramGroupHitgroup{
            .moduleCH            = module_,
            .entryFunctionNameCH = CLOSESTHIT_TRACE_NAME
        }
    };
    log_len = log.size();
    create_result = optixProgramGroupCreate(
        context, &hit_group_desc, 1, &group_options,
        log.data(), &log_len, &hit_group_);
    if(create_result != OPTIX_SUCCESS)
        throw BtrcException(log.data());
    BTRC_SCOPE_FAIL{
        optixProgramGroupDestroy(hit_group_);
        hit_group_ = nullptr;
    };

    // pipeline

    std::array<const OptixProgramGroup, 3> program_groups = {
        raygen_group_, miss_group_, hit_group_
    };
    const OptixPipelineLinkOptions pipeline_link_options = {
        .maxTraceDepth = 1,
        .debugLevel    = debug_level
    };
    log_len = log.size();
    create_result = optixPipelineCreate(
        context, &pipeline_compile_options, &pipeline_link_options,
        program_groups.data(), program_groups.size(),
        log.data(), &log_len, &pipeline_);
    if(create_result != OPTIX_SUCCESS)
        throw BtrcException(log.data());
    BTRC_SCOPE_FAIL{
        optixPipelineDestroy(pipeline_);
        pipeline_ = nullptr;
    };

    // stack size

    OptixStackSizes raygen_stack_sizes, miss_stack_sizes, hit_stack_sizes;
    throw_on_error(optixProgramGroupGetStackSize(
        raygen_group_, &raygen_stack_sizes));
    throw_on_error(optixProgramGroupGetStackSize(
        miss_group_, &miss_stack_sizes));
    throw_on_error(optixProgramGroupGetStackSize(
        hit_group_, &hit_stack_sizes));

    const unsigned int cssRG = raygen_stack_sizes.cssRG;
    const unsigned int cssMS = miss_stack_sizes.cssMS;
    const unsigned int cssCH = hit_stack_sizes.cssCH;
    const unsigned int css = cssRG + (std::max)(cssMS, cssCH);

    throw_on_error(optixPipelineSetStackSize(
        pipeline_, 0, 0, css, traversable_depth));

    // sbt

    sbt_.set_raygen_shader(raygen_group_);
    sbt_.set_miss_shader(miss_group_);
    sbt_.set_hit_shader(hit_group_);
}

BTRC_WAVEFRONT_END
