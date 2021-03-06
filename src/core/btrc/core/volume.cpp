#include <btrc/core/volume/aggregate.h>
#include <btrc/core/volume/bvh.h>

BTRC_BEGIN

namespace
{

    float min(float a, float b, float c, float d)
    {
        return std::min(std::min(a, b), std::min(c, d));
    }

    float max(float a, float b, float c, float d)
    {
        return std::max(std::max(a, b), std::max(c, d));
    }

} // namespace anonymous

void VolumePrimitive::set_geometry(const Vec3f &o, const Vec3f &x, const Vec3f &y, const Vec3f &z)
{
    o_ = o;
    x_ = x;
    y_ = y;
    z_ = z;
}

void VolumePrimitive::set_sigma_t(RC<Texture3D> sigma_t)
{
    sigma_t_ = std::move(sigma_t);
}

void VolumePrimitive::set_albedo(RC<Texture3D> albedo)
{
    albedo_ = std::move(albedo);
}

VolumePrimitive::VolumeGeometryInfo VolumePrimitive::get_geometry_info() const
{
    return { o_, x_, y_, z_ };
}

CVec3f VolumePrimitive::world_pos_to_uvw(ref<CVec3f> p) const
{
    var op = p - o_;
    var u = dot(op, x_) * (1.0f / length_square(x_));
    var v = dot(op, y_) * (1.0f / length_square(y_));
    var w = dot(op, z_) * (1.0f / length_square(z_));
    return CVec3f(u, v, w);
}

AABB3f VolumePrimitive::get_bounding_box() const
{
    const Vec3f o = o_, x = x_, y = y_, z = z_;
    const Vec3f lower = {
        min(o.x, o.x + x.x, o.x + y.x, o.x + z.x),
        min(o.y, o.y + x.y, o.y + y.y, o.y + z.y),
        min(o.z, o.z + x.z, o.z + y.z, o.z + z.z),
    };
    const Vec3f upper = {
        max(o.x, o.x + x.x, o.x + y.x, o.x + z.x),
        max(o.y, o.y + x.y, o.y + y.y, o.y + z.y),
        max(o.z, o.z + x.z, o.z + y.z, o.z + z.z),
    };
    return { lower, upper };
}

RC<Texture3D> VolumePrimitive::get_sigma_t() const
{
    return sigma_t_.get();
}

RC<Texture3D> VolumePrimitive::get_albedo() const
{
    return albedo_.get();
}

struct VolumePrimitiveMedium::Impl
{
    std::vector<RC<VolumePrimitive>> vols;
    Box<volume::Aggregate> aggregate;
    Box<volume::BVH> bvh;
};

VolumePrimitiveMedium::VolumePrimitiveMedium()
{
    impl_ = new Impl;
}

VolumePrimitiveMedium::~VolumePrimitiveMedium()
{
    delete impl_;
}

void VolumePrimitiveMedium::add_volume(RC<VolumePrimitive> vol)
{
    impl_->vols.push_back(std::move(vol));
}

const std::vector<RC<VolumePrimitive>> &VolumePrimitiveMedium::get_prims() const
{
    return impl_->vols;
}

void VolumePrimitiveMedium::commit()
{
    impl_->aggregate = newBox<volume::Aggregate>(impl_->vols);
    impl_->bvh = newBox<volume::BVH>(impl_->vols);
}

Medium::SampleResult VolumePrimitiveMedium::sample(
    CompileContext &cc,
    ref<CVec3f>     a,
    ref<CVec3f>     b,
    ref<CVec3f>     uvw_a,
    ref<CVec3f>     uvw_b,
    Sampler        &sampler) const
{
    SampleResult result;
    auto shader = newRC<HenyeyGreensteinPhaseShader>();
    result.shader = shader;

    if(impl_->bvh->is_empty())
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
        $if(!impl_->bvh->find_closest_intersection(o, b, inct_pos))
        {
            inct_pos = b;
        };
        var overlap = impl_->bvh->get_overlap(0.5f * (o + inct_pos));
        $if(overlap.count != i32(0))
        {
            impl_->aggregate->sample_scattering(
                cc, overlap, o, inct_pos, sampler,
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

CSpectrum VolumePrimitiveMedium::tr(
    CompileContext &cc,
    ref<CVec3f>     a,
    ref<CVec3f>     b,
    ref<CVec3f>     uvw_a,
    ref<CVec3f>     uvw_b,
    Sampler        &sampler) const
{
    var result = CSpectrum::one();
    if(impl_->bvh->is_empty())
        return result;

    var o = a, d = normalize(b - a);
    $loop
    {
        $if(dot(b - o, d) <= 0.0f | length_square(b - o) < 0.0001f)
        {
            $break;
        };
        CVec3f inct_pos;
        $if(!impl_->bvh->find_closest_intersection(o, b, inct_pos))
        {
            inct_pos = b;
        };
        var overlap = impl_->bvh->get_overlap(0.5f * (o + inct_pos));
        $if(overlap.count != i32(0))
        {
            var seg_tr = impl_->aggregate->tr(cc, overlap, o, inct_pos, sampler);
            result = result * seg_tr;
        };
        o = inct_pos + 0.001f * d;
    };

    return result;
}

float VolumePrimitiveMedium::get_priority() const
{
    return std::numeric_limits<float>::lowest();
}

std::vector<RC<Object>> VolumePrimitiveMedium::get_dependent_objects()
{
    std::vector<RC<Object>> result;
    for(auto &v : impl_->vols)
        result.push_back(v);
    return result;
}

BTRC_END
