#pragma once

#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/optix/sbt.h>

BTRC_OPTIX_BEGIN

namespace pipeline_mk_detail
{

    CUJ_CLASS_BEGIN(Hit)
        CUJ_MEMBER_VARIABLE(f32,    t)
        CUJ_MEMBER_VARIABLE(u32,    inst_id)
        CUJ_MEMBER_VARIABLE(u32,    prim_id)
        CUJ_MEMBER_VARIABLE(CVec2f, uv)
        boolean miss() const { return t >= 0.0f; }
    CUJ_CLASS_END

} // namespace pipeline_mk_detail

template<typename LaunchParams, typename CLaunchParams = cuj::cxx<LaunchParams>>
class MegaKernelPipeline : public Uncopyable
{
public:

    using Hit = pipeline_mk_detail::Hit;

    struct RecordContext
    {
        cuj::dsl::GlobalVariable<CLaunchParams>          launch_params;
        std::function<Hit(u64 handle, const CRay &)>     find_closest_intersection;
        std::function<boolean(u64 handle, const CRay &)> has_intersection;
    };

    using RayGenRecorder = std::function<void(const RecordContext &)>;

    struct Config
    {
        int  traversal_depth;
        bool motion_blur;
        bool triangle_only;
    };

    MegaKernelPipeline() = default;

    MegaKernelPipeline(OptixDeviceContext context, RayGenRecorder raygen_recorder, const Config &config);

    MegaKernelPipeline(MegaKernelPipeline &&other) noexcept;

    MegaKernelPipeline &operator=(MegaKernelPipeline &&other) noexcept;

    ~MegaKernelPipeline();

    void swap(MegaKernelPipeline &other) noexcept;

    operator bool() const;

    void launch(const LaunchParams &launch_params, int width, int height, int depth);

private:

    OptixModule   module_   = nullptr;
    OptixPipeline pipeline_ = nullptr;

    OptixProgramGroup raygen_group_      = nullptr;
    OptixProgramGroup trace_miss_group_  = nullptr;
    OptixProgramGroup trace_hit_group_   = nullptr;
    OptixProgramGroup shadow_miss_group_ = nullptr;
    OptixProgramGroup shadow_hit_group_  = nullptr;

    SBT sbt_;

    cuda::Buffer<LaunchParams> device_launch_params_;
};

BTRC_OPTIX_END
