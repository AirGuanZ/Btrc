#pragma once

#include <btrc/core/volume/bvh.h>
#include <btrc/core/medium.h>

BTRC_BEGIN

namespace volume
{

    class Aggregate : public Object
    {
    public:

        using Overlap = BVH::Overlap;

        Aggregate() = default;

        explicit Aggregate(const std::vector<RC<VolumePrimitive>> &vols);

        Aggregate(Aggregate &&other) noexcept;

        Aggregate &operator=(Aggregate &&other) noexcept;

        void swap(Aggregate &other) noexcept;

        void sample_scattering(
            CompileContext              &cc,
            ref<Overlap>                 overlap,
            ref<CVec3f>                  a,
            ref<CVec3f>                  b,
            Sampler                     &sampler,
            ref<boolean>                 output_scattered,
            ref<CSpectrum>               output_throughput,
            ref<CVec3f>                  output_position,
            HenyeyGreensteinPhaseShader &output_shader) const;

        CSpectrum tr(
            CompileContext &cc,
            ref<Overlap>    overlap,
            ref<CVec3f>     a,
            ref<CVec3f>     b,
            Sampler        &sampler) const;

    private:

        i32 find_overlap_index(ref<Overlap> overlap) const;

        float get_max_density(const std::set<RC<VolumePrimitive>> &vols) const;

        void sample_scattering_in_overlap(
            CompileContext                      &cc,
            const std::set<RC<VolumePrimitive>> &vols,
            ref<CVec3f>                          a,
            ref<CVec3f>                          b,
            Sampler                             &sampler,
            ref<boolean>                         output_scattered,
            ref<CSpectrum>                       output_throughput,
            ref<CVec3f>                          output_position,
            HenyeyGreensteinPhaseShader         &output_shader) const;
        
        CSpectrum tr_in_overlap(
            CompileContext                      &cc,
            const std::set<RC<VolumePrimitive>> &vols,
            ref<CVec3f>                          a,
            ref<CVec3f>                          b,
            Sampler                             &sampler) const;

        std::vector<std::set<RC<VolumePrimitive>>> overlaps_;
        cuda::Buffer<int32_t> overlap_trie_;
    };

} // namespace volume

BTRC_END
