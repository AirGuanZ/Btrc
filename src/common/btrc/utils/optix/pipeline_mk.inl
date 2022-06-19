#pragma once

#include <array>
#include <execution>

#include <btrc/utils/optix/device_funcs.h>

BTRC_OPTIX_BEGIN

namespace pipeline_mk_detail
{

    inline const char *LAUNCH_PARAM_NAME = "__btrc_launch_params";

    inline const char *KERNEL_NAME_RAYGEN      = "__raygen__main";
    inline const char *KERNEL_NAME_TRACE_MISS  = "__miss__trace";
    inline const char *KERNEL_NAME_TRACE_HIT   = "__closesthit__trace";
    inline const char *KERNEL_NAME_SHADOW_MISS = "__miss__shadow";
    inline const char *KERNEL_NAME_SHADOW_HIT  = "__closesthit__shadow";

} // namespace pipeline_mk_detail

template<typename LaunchParams, typename CLaunchParams>
MegaKernelPipeline<LaunchParams, CLaunchParams>::MegaKernelPipeline(
    OptixDeviceContext context, RayGenRecorder raygen_recorder, const Config &config)
{
    using namespace pipeline_mk_detail;

    // record ptx

    CompileContext cc;

    std::vector<std::string> ptxs;
    {
        using namespace cuj;

        ScopedModule cuj_module;

        auto global_launch_params = allocate_constant_memory<CLaunchParams>(LAUNCH_PARAM_NAME);

        kernel(KERNEL_NAME_TRACE_HIT, []
        {
            f32 t = get_ray_tmax();
            u32 inst_id = get_instance_id();
            u32 prim_id = get_primitive_index();
            CVec2f uv = get_triangle_barycentrics();
            set_payload(0, bitcast<u32>(t));
            set_payload(1, inst_id);
            set_payload(2, prim_id);
            set_payload(3, bitcast<u32>(uv.x));
            set_payload(4, bitcast<u32>(uv.y));
        });

        kernel(KERNEL_NAME_TRACE_MISS, []
        {
            set_payload(0, bitcast<u32>(f32(-1)));
        });

        kernel(KERNEL_NAME_SHADOW_HIT, []
        {
           set_payload(0, 1); 
        });

        kernel(KERNEL_NAME_SHADOW_MISS, []
        {
            set_payload(0, 0);
        });

        RecordContext record_context;
        record_context.cc = &cc;
        record_context.launch_params = global_launch_params;
        record_context.find_closest_intersection = [](u64 handle, const CRay &ray)
        {
            u32 p0, p1, p2, p3, p4;
            trace(
                handle, ray.o, ray.d, 0.0f, ray.t, 0.0f,
                u32(RAY_MASK_ALL), OPTIX_RAY_FLAG_NONE, 0, 0, 0,
                p0, p1, p2, p3, p4);
            Hit hit;
            hit.t = bitcast<f32>(p0);
            $if(hit.t >= 0.0f)
            {
                hit.inst_id = p1;
                hit.prim_id = p2;
                hit.uv.x = bitcast<f32>(p3);
                hit.uv.y = bitcast<f32>(p4);
            }
            $else
            {
                hit.inst_id = 0;
                hit.prim_id = 0;
                hit.uv.x = 0;
                hit.uv.y = 0;
            };
            return hit;
        };
        record_context.has_intersection = [](u64 handle, const CRay &ray)
        {
            u32 p0;
            trace(
                handle, ray.o, ray.d, 0.0f, ray.t, 0.0f,
                u32(RAY_MASK_ALL), OPTIX_RAY_FLAG_NONE, 1, 0, 1, p0);
            return p0 != 0;
        };

        kernel(KERNEL_NAME_RAYGEN, [&record_context, &raygen_recorder]
        {
            raygen_recorder(record_context);
        });

        auto modules = cc.get_separate_modules();
        modules.push_back(&cuj_module);

        ptxs.resize(modules.size());
        std::for_each(std::execution::par, modules.begin(), modules.end(), [&](const Module* &mod)
        {
            PTXGenerator ptx_gen;
            ptx_gen.set_options(Options{
                .opt_level = OptimizationLevel::O3,
                .fast_math = true,
                .approx_math_func = true
            });
            ptx_gen.generate(*mod);

            const size_t offset = &mod - &modules[0];
            ptxs[offset] = ptx_gen.get_ptx();
        });
    }

    std::string ptx = ptxs[0];
    for(size_t i = 1; i < ptxs.size(); ++i)
    {
        auto &p = ptxs[i];

        // remove header

        const std::string_view header = ".address_size 64";
        const size_t header_pos = p.find(header);
        p = p.substr(header_pos + header.size());

        // remove extern specifier

        size_t next_pos = 0;
        while(true)
        {
            const size_t pos = p.find(".extern .func", next_pos);
            if(pos == std::string::npos)
                break;

            const size_t bpos = p.find("btrc_", pos);
            p.insert(bpos, "dummy_extern_");
            next_pos = bpos;
        }

        ptx += p;
    }

    // create module

    constexpr OptixCompileDebugLevel debug_level = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

#if BTRC_IS_DEBUG
    constexpr OptixCompileOptimizationLevel opt_level =
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    constexpr unsigned int exception_flag =
        OPTIX_EXCEPTION_FLAG_DEBUG |
        OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    constexpr OptixCompileOptimizationLevel opt_level =
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    constexpr unsigned int exception_flag = OPTIX_EXCEPTION_FLAG_NONE;
#endif

    const OptixModuleCompileOptions module_compile_options = {
        .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        .optLevel         = opt_level,
        .debugLevel       = debug_level,
        .boundValues      = nullptr,
        .numBoundValues   = 0
    };

    const unsigned int graph_flag = config.motion_blur ?
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY :
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    const unsigned int primitive_type_flag =
        config.triangle_only ? OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE : 0;

    const OptixPipelineCompileOptions pipeline_compile_options = {
        .usesMotionBlur                   = config.motion_blur,
        .traversableGraphFlags            = graph_flag,
        .numPayloadValues                 = 5,
        .exceptionFlags                   = exception_flag,
        .pipelineLaunchParamsVariableName = LAUNCH_PARAM_NAME,
        .usesPrimitiveTypeFlags           = primitive_type_flag
    };

    std::vector<char> log(2048); size_t log_len = log.size();
    const auto module_create_result = optixModuleCreateFromPTX(
        context, &module_compile_options, &pipeline_compile_options,
        ptx.data(), ptx.size(), log.data(), &log_len, &module_);
    if(module_create_result != OPTIX_SUCCESS)
        throw BtrcException(log.data());
    BTRC_SCOPE_FAIL
    {
        optixModuleDestroy(module_);
        module_ = nullptr;
    };

    auto create_group = [&](const OptixProgramGroupDesc &desc)
    {
        const OptixProgramGroupOptions group_options = {};
        OptixProgramGroup group;
        log_len = log.size();
        const auto result = optixProgramGroupCreate(
            context, &desc, 1, &group_options, log.data(), &log_len, &group);
        if(result != OPTIX_SUCCESS)
            throw BtrcException(log.data());
        return group;
    };

    // raygen group

    raygen_group_ = create_group(OptixProgramGroupDesc{
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        .raygen = OptixProgramGroupSingleModule{
            .module = module_,
            .entryFunctionName = KERNEL_NAME_RAYGEN
        }
    });
    BTRC_SCOPE_FAIL
    {
        optixProgramGroupDestroy(raygen_group_);
        raygen_group_ = nullptr;
    };

    // miss group

    trace_miss_group_ = create_group(OptixProgramGroupDesc{
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
        .miss = OptixProgramGroupSingleModule{
            .module = module_,
            .entryFunctionName = KERNEL_NAME_TRACE_MISS
        }
    });

    shadow_miss_group_ = create_group(OptixProgramGroupDesc{
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
        .miss = OptixProgramGroupSingleModule{
            .module = module_,
            .entryFunctionName = KERNEL_NAME_SHADOW_MISS
        }
    });

    // hit group

    trace_hit_group_ = create_group(OptixProgramGroupDesc{
        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        .hitgroup = OptixProgramGroupHitgroup{
            .moduleCH = module_,
            .entryFunctionNameCH = KERNEL_NAME_TRACE_HIT
        }
    });

    shadow_hit_group_ = create_group(OptixProgramGroupDesc{
        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        .hitgroup = OptixProgramGroupHitgroup{
            .moduleCH = module_,
            .entryFunctionNameCH = KERNEL_NAME_SHADOW_HIT
        }
    });

    // pipeline

    std::array<const OptixProgramGroup, 5> program_groups = {
        raygen_group_,
        trace_miss_group_,
        shadow_miss_group_,
        trace_hit_group_,
        shadow_hit_group_
    };
    const OptixPipelineLinkOptions pipeline_link_options = {
        .maxTraceDepth = 1,
        .debugLevel = debug_level
    };
    log_len = log.size();
    const auto pipeline_create_result = optixPipelineCreate(
        context, &pipeline_compile_options, &pipeline_link_options,
        program_groups.data(), program_groups.size(),
        log.data(), &log_len, &pipeline_);
    if(pipeline_create_result != OPTIX_SUCCESS)
        throw BtrcException(log.data());
    BTRC_SCOPE_FAIL{
        optixPipelineDestroy(pipeline_);
        pipeline_ = nullptr;
    };

    // stack size

    OptixStackSizes raygen_stack_sizes;
    OptixStackSizes trace_miss_stack_sizes;
    OptixStackSizes shadow_miss_stack_sizes;
    OptixStackSizes trace_hit_stack_sizes;
    OptixStackSizes shadow_hit_stack_sizes;

    throw_on_error(optixProgramGroupGetStackSize(raygen_group_, &raygen_stack_sizes));
    throw_on_error(optixProgramGroupGetStackSize(trace_miss_group_, &trace_miss_stack_sizes));
    throw_on_error(optixProgramGroupGetStackSize(shadow_miss_group_, &shadow_miss_stack_sizes));
    throw_on_error(optixProgramGroupGetStackSize(trace_hit_group_, &trace_hit_stack_sizes));
    throw_on_error(optixProgramGroupGetStackSize(shadow_hit_group_, &shadow_hit_stack_sizes));

    const unsigned int cssRG = raygen_stack_sizes.cssRG;
    const unsigned int cssMS_trace = trace_miss_stack_sizes.cssMS;
    const unsigned int cssCH_trace = trace_hit_stack_sizes.cssCH;
    const unsigned int cssMS_shadow = shadow_miss_stack_sizes.cssMS;
    const unsigned int cssCH_shadow = shadow_hit_stack_sizes.cssCH;
    const unsigned int css = cssRG + (std::max)(
        (std::max)(cssMS_trace, cssCH_trace), (std::max)(cssMS_shadow, cssCH_shadow));

    throw_on_error(optixPipelineSetStackSize(pipeline_, 0, 0, css, config.traversal_depth));

    // sbt

    sbt_.set_raygen_shader(raygen_group_);
    sbt_.set_miss_shaders({ trace_miss_group_, shadow_miss_group_ });
    sbt_.set_hit_shaders({ trace_hit_group_, shadow_hit_group_ });

    // launch params

    device_launch_params_.initialize(1);
}

template<typename LaunchParams, typename CLaunchParams>
MegaKernelPipeline<LaunchParams, CLaunchParams>::MegaKernelPipeline(MegaKernelPipeline &&other) noexcept
    : MegaKernelPipeline()
{
    swap(other);
}

template<typename LaunchParams, typename CLaunchParams>
MegaKernelPipeline<LaunchParams, CLaunchParams> &MegaKernelPipeline<LaunchParams, CLaunchParams>::operator=(MegaKernelPipeline &&other) noexcept
{
    swap(other);
    return *this;
}

template<typename LaunchParams, typename CLaunchParams>
MegaKernelPipeline<LaunchParams, CLaunchParams>::~MegaKernelPipeline()
{
    if(!pipeline_)
        return;
    assert(module_ && raygen_group_ && miss_group_ && hit_group_);
    optixPipelineDestroy(pipeline_);
    optixProgramGroupDestroy(raygen_group_);
    optixProgramGroupDestroy(trace_miss_group_);
    optixProgramGroupDestroy(trace_hit_group_);
    optixProgramGroupDestroy(shadow_miss_group_);
    optixProgramGroupDestroy(shadow_hit_group_);
    optixModuleDestroy(module_);
    sbt_ = {};
}

template<typename LaunchParams, typename CLaunchParams>
void MegaKernelPipeline<LaunchParams, CLaunchParams>::swap(MegaKernelPipeline &other) noexcept
{
    std::swap(module_, other.module_);
    std::swap(pipeline_, other.pipeline_);
    std::swap(raygen_group_, other.raygen_group_);
    std::swap(trace_miss_group_, other.trace_miss_group_);
    std::swap(trace_hit_group_, other.trace_hit_group_);
    std::swap(shadow_miss_group_, other.shadow_miss_group_);
    std::swap(shadow_hit_group_, other.shadow_hit_group_);
    std::swap(device_launch_params_, other.device_launch_params_);
    sbt_.swap(other.sbt_);
}

template<typename LaunchParams, typename CLaunchParams>
MegaKernelPipeline<LaunchParams, CLaunchParams>::operator bool() const
{
    return pipeline_ != nullptr;
}

template<typename LaunchParams, typename CLaunchParams>
void MegaKernelPipeline<LaunchParams, CLaunchParams>::launch(
    const LaunchParams &launch_params, int width, int height, int depth)
{
    device_launch_params_.from_cpu(&launch_params);
    throw_on_error(optixLaunch(
        pipeline_, nullptr, device_launch_params_, sizeof(LaunchParams), &sbt_.get_table(), width, height, depth));
}

BTRC_OPTIX_END
