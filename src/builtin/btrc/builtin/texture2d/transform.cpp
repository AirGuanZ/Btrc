#include <btrc/builtin/texture2d/transform.h>

BTRC_BUILTIN_BEGIN

void TransformTexture2D::set_inverse_u(bool inv)
{
    inv_u_ = inv;
}

void TransformTexture2D::set_inverse_v(bool inv)
{
    inv_v_ = inv;
}

void TransformTexture2D::set_swap_uv(bool swap)
{
    swap_uv_ = swap;
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
    spt2.uv = map_uv(spt2.uv);
    return tex_->sample_spectrum(cc, spt2);
}

f32 TransformTexture2D::sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    return tex_->sample_float(cc, map_uv(uv));
}

f32 TransformTexture2D::sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const
{
    var spt2 = spt;
    spt2.uv = map_uv(spt2.uv);
    return tex_->sample_float(cc, spt2);
}

CVec2f TransformTexture2D::map_uv(const CVec2f &uv) const
{
    var ret = uv;
    if(inv_u_)
        ret.x = 1.0f - ret.x;
    if(inv_v_)
        ret.y = 1.0f - ret.y;
    if(swap_uv_)
    {
        var t = ret.x;
        ret.x = ret.y;
        ret.y = t;
    }
    return ret;
}

RC<Texture2D> TransformTexture2DCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const bool inv_u = node->parse_child_or("inv_u", false);
    const bool inv_v = node->parse_child_or("inv_v", false);
    const bool swap_uv = node->parse_child_or("swap_uv", false);
    auto tex = context.create<Texture2D>(node->child_node("tex"));
    auto ret = newRC<TransformTexture2D>();
    ret->set_inverse_u(inv_u);
    ret->set_inverse_v(inv_v);
    ret->set_swap_uv(swap_uv);
    ret->set_texture(std::move(tex));
    return ret;
}

BTRC_BUILTIN_END
