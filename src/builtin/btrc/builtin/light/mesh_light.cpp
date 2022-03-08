#include <btrc/builtin/light/mesh_light.h>

BTRC_BUILTIN_BEGIN

void MeshLight::set_intensity(const Spectrum &intensity)
{
    intensity_ = intensity;
    set_recompile(true);
}

void MeshLight::set_geometry(RC<Geometry> geometry, const Transform &local_to_world)
{
    geometry_ = std::move(geometry);
    local_to_world_ = local_to_world;
    set_recompile(true);
}

CSpectrum MeshLight::eval_le_inline(
    CompileContext &cc,
    ref<CVec3f>     pos,
    ref<CVec3f>     nor,
    ref<CVec2f>     uv,
    ref<CVec2f>     tex_coord,
    ref<CVec3f>     wr) const
{
    CSpectrum result;
    $if(dot(nor, wr) >= 0)
    {
        result = CSpectrum(intensity_);
    }
    $else
    {
        result = CSpectrum::zero();
    };
    return result;
}

AreaLight::SampleLiResult MeshLight::sample_li_inline(
    CompileContext &cc,
    ref<CVec3f>     ref_pos,
    ref<CVec3f>     sam) const
{
    CTransform ctrans = local_to_world_;

    var surface_sample = geometry_->sample(cc, sam);
    var spos = ctrans.apply_to_point(surface_sample.point.position);
    var snor = normalize(ctrans.apply_to_vector(surface_sample.point.frame.z));
    var pos_to_ref = ref_pos - spos;
    var dist2 = length_square(pos_to_ref);
    var pdf_sa = 1 / (local_to_world_.scale.x * local_to_world_.scale.x)
               * surface_sample.pdf * dist2 / cstd::abs(dot(snor, normalize(pos_to_ref)));

    CSpectrum rad;
    $if(dot(pos_to_ref, snor) > 0)
    {
        rad = CSpectrum(intensity_);
    }
    $else
    {
        rad = CSpectrum::zero();
    };

    SampleLiResult result;
    result.position   = spos;
    result.normal     = snor;
    result.pdf        = pdf_sa;
    result.radiance = rad;
    return result;
}

f32 MeshLight::pdf_li_inline(
    CompileContext &cc,
    ref<CVec3f>     ref_pos,
    ref<CVec3f>     pos,
    ref<CVec3f>     nor) const
{
    f32 result;
    $if(dot(nor, ref_pos - pos) <= 0)
    {
        result = 0;
    }
    $else
    {
        var pdf_area = geometry_->pdf(cc, ref_pos, pos);
        var spt_to_ref = ref_pos - pos;
        var dist2 = length_square(spt_to_ref);
        var dist3 = cstd::sqrt(dist2) * dist2;
        result = 1 / (local_to_world_.scale.x * local_to_world_.scale.x)
               * pdf_area * dist3 / cstd::abs(dot(nor, spt_to_ref));
    };
    return result;
}

RC<Light> MeshLightCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto intensity = node->parse_child<Spectrum>("intensity");
    auto result = newRC<MeshLight>();
    result->set_intensity(intensity);
    return result;
}

BTRC_BUILTIN_END
