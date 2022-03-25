#include <btrc/builtin/texture2d/transform.h>

BTRC_BUILTIN_BEGIN

void TransformTexture2D::set_transform(const Transform2D &transform)
{
    transform_ = transform;
}

void TransformTexture2D::set_texture(RC<Texture2D> tex)
{
    tex_ = std::move(tex);
}

CSpectrum TransformTexture2D::sample_spectrum_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    return tex_->sample_spectrum(cc, map_uv(uv));
}

CSpectrum TransformTexture2D::sample_spectrum_inline(CompileContext &cc, ref<SurfacePoint> spt) const
{
    var spt2 = spt;
    spt2.tex_coord = map_uv(spt.tex_coord);
    return tex_->sample_spectrum(cc, spt2);
}

f32 TransformTexture2D::sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    return tex_->sample_float(cc, map_uv(uv));
}

f32 TransformTexture2D::sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const
{
    var spt2 = spt;
    spt2.tex_coord = map_uv(spt.tex_coord);
    return tex_->sample_float(cc, spt2);
}

CVec2f TransformTexture2D::map_uv(const CVec2f &uv) const
{
    return CTransform2D(transform_).apply_to_point(uv);
}

RC<Texture2D> TransformTexture2DCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const Transform2D trans = node->parse_child<Transform2D>("transform");
    auto tex = context.create<Texture2D>(node->child_node("tex"));
    auto ret = newRC<TransformTexture2D>();
    ret->set_transform(trans);
    ret->set_texture(std::move(tex));
    return ret;
}

BTRC_BUILTIN_END
