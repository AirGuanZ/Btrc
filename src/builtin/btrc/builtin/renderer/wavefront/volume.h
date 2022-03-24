#pragma once

#include <btrc/builtin/renderer/wavefront/common.h>
#include <btrc/builtin/volume/aggregate.h>
#include <btrc/builtin/volume/bvh.h>

BTRC_WFPT_BEGIN

class VolumeManager : public Uncopyable
{
public:

    VolumeManager() = default;

    explicit VolumeManager(const std::vector<RC<VolumePrimitive>> &vols);

    VolumeManager(VolumeManager &&other) noexcept;

    VolumeManager &operator=(VolumeManager &&other) noexcept;

    void swap(VolumeManager &other) noexcept;

    Medium::SampleResult sample_scattering(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, Sampler &sampler) const;

    CSpectrum tr(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, Sampler &sampler) const;

private:

    Box<volume::Aggregate> aggregate_;
    Box<volume::BVH> bvh_;
};

BTRC_WFPT_END
