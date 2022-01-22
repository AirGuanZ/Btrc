#pragma once

#include <btrc/core/utils/optix/sbt.h>
#include <btrc/core/utils/uncopyable.h>
#include <btrc/core/wavefront/soa.h>

BTRC_WAVEFRONT_BEGIN

class TracePipeline : public Uncopyable
{
public:

    TracePipeline() = default;

    TracePipeline(
        OptixDeviceContext context,
        bool               motion_blur,
        bool               triangle_only,
        int                traversable_depth);

    TracePipeline(TracePipeline &&other) noexcept;

    TracePipeline &operator=(TracePipeline &&other) noexcept;

    ~TracePipeline();

    operator bool() const;

    void swap(TracePipeline &other) noexcept;

    void trace(
        OptixTraversableHandle traversable,
        int                    active_state_count,
        const RaySOA          &input_ray,
        const IntersectionSOA &output_inct) const;

private:

    void initialize(
        OptixDeviceContext context,
        bool               motion_blur,
        bool               triangle_only,
        int                traversable_depth);

    OptixModule   module_   = nullptr;
    OptixPipeline pipeline_ = nullptr;

    OptixProgramGroup raygen_group_ = nullptr;
    OptixProgramGroup miss_group_   = nullptr;
    OptixProgramGroup hit_group_    = nullptr;

    optix::SBT sbt_;
};

BTRC_WAVEFRONT_END
