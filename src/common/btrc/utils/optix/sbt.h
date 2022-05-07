#pragma once

#include <vector>

#include <optix.h>

#include <btrc/utils/uncopyable.h>

BTRC_OPTIX_BEGIN

class SBT : public Uncopyable
{
public:

    SBT() = default;

    SBT(SBT &&other) noexcept;

    SBT &operator=(SBT &&other) noexcept;

    ~SBT();

    void swap(SBT &other) noexcept;

    void set_raygen_shader(OptixProgramGroup group);

    void set_miss_shaders(const std::vector<OptixProgramGroup> &groups);

    void set_hit_shaders(const std::vector<OptixProgramGroup> &groups);

    operator const OptixShaderBindingTable &() const;

    const OptixShaderBindingTable &get_table() const;

private:

    static CUdeviceptr prepare_records(const std::vector<OptixProgramGroup> &groups);

    OptixShaderBindingTable sbt_ = {};
};

BTRC_OPTIX_END
