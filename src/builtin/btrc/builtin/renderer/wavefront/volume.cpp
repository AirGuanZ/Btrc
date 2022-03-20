#include <btrc/builtin/renderer/wavefront/volume.h>

BTRC_WFPT_BEGIN

VolumeManager::VolumeManager(const std::vector<RC<VolumePrimitive>> &vols)
{
    aggregate_ = newBox<volume::Aggregate>(vols);
    bvh_ = newBox<volume::BVH>(vols);
}

VolumeManager::VolumeManager(VolumeManager &&other) noexcept
    : VolumeManager()
{
    swap(other);
}

VolumeManager &VolumeManager::operator=(VolumeManager &&other) noexcept
{
    swap(other);
    return *this;
}

void VolumeManager::swap(VolumeManager &other) noexcept
{
    aggregate_.swap(other.aggregate_);
    bvh_.swap(other.bvh_);
}

Medium::SampleResult VolumeManager::sample_scattering(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, ref<CRNG> rng) const
{
    Medium::SampleResult result;
    auto shader = newRC<HenyeyGreensteinPhaseShader>();
    result.shader = shader;

    if(bvh_->is_empty())
    {
        result.scattered = false;
        result.throughput = CSpectrum::one();
        return result;
    }

    var o = a, d = normalize(b - a);
    $loop
    {
        $if(dot(b - o, d) <= 0.0f | length_square(b - o) < 0.00001f)
        {
            result.scattered = false;
            result.throughput = CSpectrum::one();
            $break;
        };
        CVec3f inct_pos;
        $if(!bvh_->find_closest_intersection(o, b, inct_pos))
        {
            inct_pos = b;
        };
        var overlap = bvh_->get_overlap(0.5f * (o + inct_pos));
        $if(overlap.count != i32(0))
        {
            aggregate_->sample_scattering(
                cc, overlap, o, inct_pos, rng,
                result.scattered, result.throughput, result.position, *shader);
            $if(result.scattered)
            {
                // result.throughput will always be one in this case
                $break;
            };
        };
        o = inct_pos + 0.001f * d;
    };

    return result;
}

CSpectrum VolumeManager::tr(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, ref<CRNG> rng) const
{
    var result = CSpectrum::one();
    if(bvh_->is_empty())
        return result;

    var o = a, d = normalize(b - a);
    $loop
    {
        $if(dot(b - o, d) <= 0.0f | length_square(b - o) < 0.0001f)
        {
            $break;
        };
        CVec3f inct_pos;
        $if(!bvh_->find_closest_intersection(o, b, inct_pos))
        {
            inct_pos = b;
        };
        var overlap = bvh_->get_overlap(0.5f * (o + inct_pos));
        $if(overlap.count != i32(0))
        {
            var seg_tr = aggregate_->tr(cc, overlap, o, inct_pos, rng);
            result = result * seg_tr;
        };
        o = inct_pos + 0.001f * d;
    };

    return result;
}

BTRC_WFPT_END
