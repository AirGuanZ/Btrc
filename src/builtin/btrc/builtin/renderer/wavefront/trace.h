#pragma once

#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/cuda/buffer.h>
#include <btrc/utils/optix/pipeline.h>
#include <btrc/utils/uncopyable.h>

#include "./common.h"

BTRC_WFPT_BEGIN

namespace trace_pipeline_detail
{

    struct SOAParams
    {
        Vec4f *ray_o_t0;
        Vec4f *ray_d_t1;
        Vec2u *ray_time_mask;

        Vec2u *inct_inst_launch_index;
        Vec4u *inct_t_prim_uv;
    };

    struct LaunchParams
    {
        OptixTraversableHandle handle;

        // ray

        Vec4f *ray_o_t0;
        Vec4f *ray_d_t1;
        Vec2u *ray_time_mask;

        // incts

        Vec2u *inct_inst_launch_index;
        Vec4u *inct_t_prim_uv;
    };

    CUJ_PROXY_CLASS(
        CLaunchParams, LaunchParams,
        handle, ray_o_t0, ray_d_t1, ray_time_mask, inct_inst_launch_index, inct_t_prim_uv);

} // namespace trace_pipeline_detail

class TracePipeline : public Uncopyable
{
public:

    using SOAParams = trace_pipeline_detail::SOAParams;
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

    mutable cuda::CUDABuffer<LaunchParams> device_launch_params_;
};

BTRC_WFPT_END
