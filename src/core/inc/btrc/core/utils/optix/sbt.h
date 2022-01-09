#pragma once

#include <vector>

#include <optix.h>

#include <btrc/core/utils/uncopyable.h>

BTRC_OPTIX_BEGIN

template<typename Data>
struct SBTRecord
{
    alignas(OPTIX_SBT_RECORD_ALIGNMENT)
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    Data data;
};

class SBT : public Uncopyable
{
public:

    SBT() = default;

    SBT(SBT &&other) noexcept;

    SBT &operator=(SBT &&other) noexcept;

    ~SBT();

    void swap(SBT &other) noexcept;
    
    void set_raygen_shader(OptixProgramGroup group);

    void set_miss_shader(OptixProgramGroup group);

    void set_hit_shader(OptixProgramGroup group);

    operator const OptixShaderBindingTable &() const;

    const OptixShaderBindingTable &get_table() const;

private:

    CUdeviceptr prepare_record(OptixProgramGroup group) const;

    OptixShaderBindingTable sbt_ = {};
};

BTRC_OPTIX_END
