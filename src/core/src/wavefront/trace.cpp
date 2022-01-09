#include <cassert>
#include <vector>

#include <btrc/core/wavefront/trace.h>
#include <btrc/core/utils/cuda/error.h>
#include <btrc/core/utils/scope_guard.h>

#include <embed_ptx/trace.inl>

#include <optix_stubs.h>

BTRC_WAVEFRONT_BEGIN

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

void TracePipeline::initialize(
    OptixDeviceContext context,
    bool               motion_blur,
    bool               triangle_only,
    int                traversable_depth)
{
    assert(!module_ && !pipeline_);
    assert(!raygen_group_ && !miss_group_ && !hit_group_);

    // create module

#if BTRC_IS_DEBUG
    constexpr OptixCompileOptimizationLevel opt_level =
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    constexpr OptixCompileDebugLevel debug_level =
        OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    constexpr unsigned int exception_flag =
        OPTIX_EXCEPTION_FLAG_DEBUG |
        OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    constexpr OptixCompileOptimizationLevel opt_level =
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    constexpr OptixCompileDebugLevel debug_level =
        OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    constexpr unsigned int exception_flag = OPTIX_EXCEPTION_FLAG_NONE;
#endif

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
        .pipelineLaunchParamsVariableName = "launch_params",
        .usesPrimitiveTypeFlags           = primitive_type_flag
    };

    std::vector<char> log(2048); size_t log_len = log.size();
    OptixResult create_result = optixModuleCreateFromPTX(
        context, &module_compile_options, &pipeline_compile_options,
        embed_var_trace, sizeof(embed_var_trace),
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
            .entryFunctionName = "__raygen__trace"
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
            .entryFunctionName = "__miss__trace"
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
            .entryFunctionNameCH = "__closesthit__trace"
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

    const OptixProgramGroup program_groups[] = {
        raygen_group_, miss_group_, hit_group_
    };
    const OptixPipelineLinkOptions pipeline_link_options = {
        .maxTraceDepth = 1,
        .debugLevel    = debug_level
    };
    log_len = log.size();
    create_result = optixPipelineCreate(
        context, &pipeline_compile_options, &pipeline_link_options,
        program_groups, sizeof(program_groups) / sizeof(program_groups[0]),
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
