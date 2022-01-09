#pragma once

#include <btrc/core/utils/optix/sbt.h>
#include <btrc/core/utils/uncopyable.h>

BTRC_WAVEFRONT_BEGIN

class TracePipeline : public Uncopyable
{
public:

    TracePipeline() = default;

    TracePipeline(TracePipeline &&other) noexcept;

    TracePipeline &operator=(TracePipeline &&other) noexcept;

    ~TracePipeline();

    operator bool() const;

    void swap(TracePipeline &other) noexcept;

    void initialize(
        OptixDeviceContext context,
        bool               motion_blur,
        bool               triangle_only,
        int                traversable_depth);

private:

    OptixModule   module_   = nullptr;
    OptixPipeline pipeline_ = nullptr;

    OptixProgramGroup raygen_group_ = nullptr;
    OptixProgramGroup miss_group_   = nullptr;
    OptixProgramGroup hit_group_    = nullptr;

    optix::SBT sbt_;
};

BTRC_WAVEFRONT_END
