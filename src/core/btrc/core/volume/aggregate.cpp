#include <btrc/core/volume/aggregate.h>
#include <btrc/core/volume/indexing.h>
#include <btrc/core/volume/overlap.h>
#include <btrc/utils/enumerate.h>

BTRC_BEGIN

volume::Aggregate::Aggregate(const std::vector<RC<VolumePrimitive>> &vols)
{
    if(vols.empty())
        return;

    // find all overlap areas

    VolumeOverlapResolver overlap_resolver;
    for(auto &vol : vols)
        overlap_resolver.add_volume(vol);

    // build overlap trie

    std::map<RC<VolumePrimitive>, int> vol_to_id;
    for(auto &&[i, vol] : enumerate(vols))
        vol_to_id[vol] = static_cast<int>(i);

    overlaps_ = overlap_resolver.get_overlaps();
    OverlapIndexer indexer(overlaps_, vols, vol_to_id);

    auto &trie_indices = indexer.get_indices();
    overlap_trie_ = cuda::Buffer<int32_t>(trie_indices);
}

volume::Aggregate::Aggregate(Aggregate &&other) noexcept
    : Aggregate()
{
    swap(other);
}

volume::Aggregate &volume::Aggregate::operator=(Aggregate &&other) noexcept
{
    swap(other);
    return *this;
}

void volume::Aggregate::swap(Aggregate &other) noexcept
{
    overlap_trie_.swap(other.overlap_trie_);
}

void volume::Aggregate::sample_scattering(
    CompileContext              &cc,
    ref<Overlap>                 overlap,
    ref<CVec3f>                  a,
    ref<CVec3f>                  b,
    Sampler                     &sampler,
    ref<boolean>                 output_scattered,
    ref<CSpectrum>               output_throughput,
    ref<CVec3f>                  output_position,
    HenyeyGreensteinPhaseShader &output_shader) const
{
    $if(overlap.count == i32(0))
    {
        output_scattered = false;
        output_throughput = CSpectrum::one();
    }
    $else
    {
        var overlap_index = find_overlap_index(overlap);
        $switch(overlap_index)
        {
            for(size_t i = 0; i < overlaps_.size(); ++i)
            {
                $case(i)
                {
                    sample_scattering_in_overlap(
                        cc, overlaps_[i], a, b, sampler,
                        output_scattered, output_throughput,
                        output_position, output_shader);
                };
            }
            $default
            {
                output_scattered = false;
                output_throughput = CSpectrum::one();
            };
        };
    };
}

CSpectrum volume::Aggregate::tr(
    CompileContext &cc, ref<Overlap> overlap,
    ref<CVec3f> a, ref<CVec3f> b, Sampler &sampler) const
{
    CSpectrum result;
    $if(overlap.count == i32(0))
    {
        result = CSpectrum::one();
    }
    $else
    {
        var overlap_index = find_overlap_index(overlap);
        $switch(overlap_index)
        {
            for(size_t i = 0; i < overlaps_.size(); ++i)
            {
                $case(i)
                {
                    result = tr_in_overlap(cc, overlaps_[i], a, b, sampler);
                };
            }
            $default
            {
                result = CSpectrum::one();
            };
        };
    };
    return result;
}

i32 volume::Aggregate::find_overlap_index(ref<Overlap> overlap) const
{
    var overlap_index = 0;
    var trie = cuj::import_pointer(overlap_trie_.get());
    $forrange(i, 0, overlap.count)
    {
        var vol_idx = overlap.data[i];
        overlap_index = trie[overlap_index + 1 + i32(vol_idx)];
        CUJ_ASSERT(overlap_index >= 0);
    };
    overlap_index = trie[overlap_index];
    CUJ_ASSERT(0 <= overlap_index & overlap_index < static_cast<int>(overlaps_.size()));
    return overlap_index;
}

float volume::Aggregate::get_max_density(const std::set<RC<VolumePrimitive>> &vols) const
{
    float max_density = 0.0f;
    for(auto &vol : vols)
        max_density += vol->get_sigma_t()->get_max_float();
    return (std::max)(max_density, 0.01f);
}

void volume::Aggregate::sample_scattering_in_overlap(
    CompileContext                      &cc,
    const std::set<RC<VolumePrimitive>> &vols,
    ref<CVec3f>                          a,
    ref<CVec3f>                          b,
    Sampler                             &sampler,
    ref<boolean>                         output_scattered,
    ref<CSpectrum>                       output_throughput,
    ref<CVec3f>                          output_position,
    HenyeyGreensteinPhaseShader         &output_shader) const
{
    const float max_density = get_max_density(vols);
    const float inv_max_density = 1.0f / max_density;

    std::vector<CVec3f> local_uvw_a(vols.size());
    std::vector<CVec3f> local_uvw_b(vols.size());
    for(auto [i, vol] : enumerate(vols))
    {
        local_uvw_a[i] = vol->world_pos_to_uvw(a);
        local_uvw_b[i] = vol->world_pos_to_uvw(b);
    }

    var t_max = length(b - a), t = 0.0f;
    $loop
    {
        var dt = -cstd::log(1.0f - sampler.get1d()) * inv_max_density;
        t = t + dt;
        $if(t >= t_max)
        {
            output_scattered = false;
            output_throughput = CSpectrum::one();
            $break;
        };
        var tf = t / t_max;

        std::vector<CVec3f> uvws(vols.size());
        std::vector<f32> densities(vols.size());
        var density_sum = 0.0f;
        for(auto [i, vol] : enumerate(vols))
        {
            var uvw = local_uvw_a[i] * (1.0f - tf) + local_uvw_b[i] * tf;
            uvws[i] = uvw;
            var density = vol->get_sigma_t()->sample_float(cc, uvw);
            densities[i] = density;
            density_sum = density_sum + density;
        }

        $if(sampler.get1d() < density_sum * inv_max_density)
        {
            output_scattered = true;
            output_throughput = CSpectrum::one();
            output_position = a * (1.0f - tf) + b * tf;

            var albedo = CSpectrum::zero();

            for(auto [i, vol] : enumerate(vols))
            {
                var uvw = uvws[i];
                var weight = densities[i] / density_sum;
                var lobe_albedo = vol->get_albedo()->sample_spectrum(cc, uvw);
                albedo = albedo + weight * lobe_albedo;
            }

            output_shader.set_g(0.0f);
            output_shader.set_color(albedo);

            $break;
        };
    };
}

CSpectrum volume::Aggregate::tr_in_overlap(
    CompileContext                      &cc,
    const std::set<RC<VolumePrimitive>> &vols,
    ref<CVec3f>                          a,
    ref<CVec3f>                          b,
    Sampler                             &sampler) const
{
    const float max_density = get_max_density(vols);
    const float inv_max_density = 1.0f / max_density;

    std::vector<CVec3f> local_uvw_a(vols.size());
    std::vector<CVec3f> local_uvw_b(vols.size());
    for(auto [i, vol] : enumerate(vols))
    {
        local_uvw_a[i] = vol->world_pos_to_uvw(a);
        local_uvw_b[i] = vol->world_pos_to_uvw(b);
    }

    var result = 1.0f, t_max = length(b - a), t = 0.0f;
    $loop
    {
        var dt = -cstd::log(1.0f - sampler.get1d()) * inv_max_density;
        t = t + dt;
        $if(t >= t_max)
        {
            $break;
        };
        var tf = t / t_max;

        var density_sum = 0.0f;
        for(auto [i, vol] : enumerate(vols))
        {
            var uvw = local_uvw_a[i] * (1.0f - tf) + local_uvw_b[i] * tf;
            var density = vol->get_sigma_t()->sample_float(cc, uvw);
            density_sum = density_sum + density;
        }

        result = result * (1.0f - density_sum * inv_max_density);
    };

    return CSpectrum::from_rgb(result, result, result);
}

BTRC_END
