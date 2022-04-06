#pragma once

#include <btrc/builtin/renderer/wavefront/soa.h>
#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/cuda/buffer.h>
#include <btrc/utils/optix/pipeline.h>
#include <btrc/utils/uncopyable.h>

BTRC_WFPT_BEGIN

namespace trace_pipeline_detail
{

    struct LaunchParams
    {
        /*Vec4f *ray_o_medium_id;
        Vec4f *ray_d_t1;

        uint32_t *path_flag;
        Vec4u    *inct_t_prim_uv;*/

        OptixTraversableHandle tlas;
        RaySOA                 ray;
        IntersectionSOA        inct;
    };

    CUJ_PROXY_CLASS(CLaunchParams, LaunchParams, tlas, ray, inct);

    /*struct LaunchParams
    {
        OptixTraversableHandle handle;

        // ray

        Vec4f *ray_o_medium_id;
        Vec4f *ray_d_t1;

        // incts

        uint32_t *path_flag;
        Vec4u    *inct_t_prim_uv;
    };

    CUJ_PROXY_CLASS(
        CLaunchParams, LaunchParams,
        handle, ray_o_medium_id, ray_d_t1, path_flag, inct_t_prim_uv);*/

} // namespace trace_pipeline_detail

class TracePipeline : public Uncopyable
{
public:

    struct SOAParams
    {
        RaySOA          ray;
        IntersectionSOA inct;
    };

    using LaunchParams = trace_pipeline_detail::LaunchParams;
    using CLaunchParams = trace_pipeline_detail::CLaunchParams;

    TracePipeline() = default;

    TracePipeline(
        OptixDeviceContext context,
        bool               motion_blur,
        bool               triangle_only,
        int                traversable_depth);

    TracePipeline(TracePipeline &&other) noexcept;

    TracePipeline &operator=(TracePipeline &&other) noexcept;

    operator bool() const;

    void swap(TracePipeline &other) noexcept;

    void trace(
        OptixTraversableHandle handle,
        int active_state_count,
        const SOAParams &soa_params) const;

private:

    void initialize(
        OptixDeviceContext context,
        bool               motion_blur,
        bool               triangle_only,
        int                traversable_depth);

    optix::SimpleOptixPipeline pipeline_;

    mutable cuda::Buffer<LaunchParams> device_launch_params_;
};

BTRC_WFPT_END
