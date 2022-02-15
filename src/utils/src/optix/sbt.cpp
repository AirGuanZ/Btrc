#include <cuda_runtime.h>

#include <btrc/utils/cuda/error.h>
#include <btrc/utils/optix/sbt.h>
#include <btrc/utils/scope_guard.h>

BTRC_OPTIX_BEGIN

SBT::SBT(SBT &&other) noexcept
    : SBT()
{
    swap(other);
}

SBT &SBT::operator=(SBT &&other) noexcept
{
    swap(other);
    return *this;
}

SBT::~SBT()
{
    if(sbt_.raygenRecord)
        cudaFree(reinterpret_cast<void*>(sbt_.raygenRecord));
    if(sbt_.missRecordBase)
        cudaFree(reinterpret_cast<void *>(sbt_.missRecordBase));
    if(sbt_.hitgroupRecordBase)
        cudaFree(reinterpret_cast<void *>(sbt_.hitgroupRecordBase));
}

void SBT::swap(SBT &other) noexcept
{
    std::swap(sbt_, other.sbt_);
}

void SBT::set_raygen_shader(OptixProgramGroup group)
{
    auto record = prepare_record(group);
    cudaFree(reinterpret_cast<void *>(sbt_.raygenRecord));
    sbt_.raygenRecord = record;
}

void SBT::set_miss_shader(OptixProgramGroup group)
{
    auto record = prepare_record(group);
    cudaFree(reinterpret_cast<void *>(sbt_.missRecordBase));
    sbt_.missRecordBase          = record;
    sbt_.missRecordCount         = 1;
    sbt_.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
}

void SBT::set_hit_shader(OptixProgramGroup group)
{
    auto record = prepare_record(group);
    cudaFree(reinterpret_cast<void *>(sbt_.hitgroupRecordBase));
    sbt_.hitgroupRecordBase          = record;
    sbt_.hitgroupRecordCount         = 1;
    sbt_.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
}

SBT::operator const OptixShaderBindingTable&() const
{
    return sbt_;
}

const OptixShaderBindingTable &SBT::get_table() const
{
    return sbt_;
}

CUdeviceptr SBT::prepare_record(OptixProgramGroup group) const
{
    char head[OPTIX_SBT_RECORD_HEADER_SIZE];
    optixSbtRecordPackHeader(group, head);

    void *result = nullptr;
    throw_on_error(cudaMalloc(&result, OPTIX_SBT_RECORD_HEADER_SIZE));
    BTRC_SCOPE_FAIL{ cudaFree(result); };
    throw_on_error(cudaMemcpy(
        result, head, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));

    return reinterpret_cast<CUdeviceptr>(result);
}

BTRC_OPTIX_END
