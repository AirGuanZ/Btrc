#include <btrc/core/medium.h>

BTRC_BEGIN

void TransformMedium::set_tranformed(RC<Medium> transformed)
{
    transformed_ = std::move(transformed);
}

void TransformMedium::set_transform(const Transform &world_to_local)
{
    world_to_local_ = world_to_local;
}

Medium::SampleResult TransformMedium::sample(
    CompileContext &cc,
    ref<CVec3f>     a,
    ref<CVec3f>     b,
    ref<CVec3f>     uvw_a,
    ref<CVec3f>     uvw_b,
    Sampler        &sampler) const
{
    var world_to_local = CTransform(world_to_local_);
    var local_uvw_a = world_to_local.apply_to_point(uvw_a);
    var local_uvw_b = world_to_local.apply_to_point(uvw_b);
    return transformed_->sample(cc, a, b, local_uvw_a, local_uvw_b, sampler);
}

CSpectrum TransformMedium::tr(
    CompileContext &cc,
    ref<CVec3f>     a,
    ref<CVec3f>     b,
    ref<CVec3f>     uvw_a,
    ref<CVec3f>     uvw_b,
    Sampler        &sampler) const
{
    var world_to_local = CTransform(world_to_local_);
    var local_uvw_a = world_to_local.apply_to_point(uvw_a);
    var local_uvw_b = world_to_local.apply_to_point(uvw_b);
    return transformed_->tr(cc, a, b, local_uvw_a, local_uvw_b, sampler);
}

float TransformMedium::get_priority() const
{
    return transformed_->get_priority();
}

BTRC_END
