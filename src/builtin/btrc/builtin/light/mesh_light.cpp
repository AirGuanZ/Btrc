#include <btrc/builtin/light/mesh_light.h>

BTRC_BUILTIN_BEGIN

void MeshLight::set_intensity(const Spectrum &intensity)
{
    intensity_ = intensity;
}

void MeshLight::set_geometry(RC<Geometry> geometry, const Transform3D &local_to_world)
{
    geometry_ = std::move(geometry);
    local_to_world_ = local_to_world;
    scale_ = length(local_to_world.apply_to_point({ 1, 0, 0 }) - local_to_world.apply_to_point({ 0, 0, 0 }));
}

CSpectrum MeshLight::eval_le_inline(CompileContext &cc, ref<SurfacePoint> spt, ref<CVec3f> wr) const
{
    CSpectrum result;
    $if(dot(spt.frame.z, wr) >= 0)
    {
        result = CSpectrum(intensity_);
    }
    $else
    {
        result = CSpectrum::zero();
    };
    return result;
}

AreaLight::SampleLiResult MeshLight::sample_li_inline(CompileContext &cc, ref<CVec3f> ref_pos, ref<Sam3> sam) const
{
    CTransform3D ctrans = local_to_world_;

    var surface_sample = geometry_->sample(cc, sam);
    var spos = ctrans.apply_to_point(surface_sample.point.position);
    var snor = normalize(ctrans.apply_to_normal(surface_sample.point.frame.z));
    var pos_to_ref = ref_pos - spos;
    var dist2 = length_square(pos_to_ref);
    var pdf_sa = 1 / (scale_ * scale_) * surface_sample.pdf * dist2 / cstd::abs(dot(snor, normalize(pos_to_ref)));

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
    result.position = spos;
    result.normal   = snor;
    result.pdf      = pdf_sa;
    result.radiance = rad;
    return result;
}

f32 MeshLight::pdf_li_inline(CompileContext &cc, ref<CVec3f> ref_pos, ref<CVec3f> pos, ref<CVec3f> nor) const
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
        result = 1 / (scale_ * scale_) * pdf_area * dist3 / cstd::abs(dot(nor, spt_to_ref));
    };
    return result;
}

AreaLight::SampleEmitResult MeshLight::sample_emit_inline(CompileContext &cc, ref<Sam<5>> sam) const
{
    var surface_sample = geometry_->sample(cc, make_sample(sam[0], sam[1], sam[2]));
    var point = surface_sample.point;
    var pdf_pos = surface_sample.pdf;

    var local_dir = sample_hemisphere_zweighted(sam[3], sam[4]);
    var pdf_dir = pdf_sample_hemisphere_zweighted(local_dir);

    apply(CTransform3D(local_to_world_), point);

    SampleEmitResult result;
    result.point = point;
    result.direction = point.frame.local_to_global(local_dir);
    result.radiance = CSpectrum(intensity_);
    result.pdf_pos = 1 / (scale_ * scale_) * pdf_pos;
    result.pdf_dir = pdf_dir;

    return result;
}

AreaLight::PdfEmitResult MeshLight::pdf_emit_inline(CompileContext &cc, ref<SurfacePoint> spt, ref<CVec3f> wr) const
{
    var local_dir = normalize(spt.frame.global_to_local(wr));
    var pdf_dir = pdf_sample_hemisphere_zweighted(local_dir);

    var pdf_pos = geometry_->pdf(cc, spt.position);
    pdf_pos = 1 / (scale_ * scale_) * pdf_pos;

    PdfEmitResult result;
    result.pdf_pos = pdf_pos;
    result.pdf_dir = pdf_dir;

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
