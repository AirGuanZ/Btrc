#pragma once

#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class TransformTexture2D : public Texture2D
{
public:

    void set_inverse_u(bool inv);

    void set_inverse_v(bool inv);

    void set_swap_uv(bool swap);

    void set_texture(RC<Texture2D> tex);

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec2f> uv) const override;

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<SurfacePoint> spt) const override;

    f32 sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const override;

    f32 sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const override;

private:

    CVec2f map_uv(const CVec2f &uv) const;

    bool inv_u_   = false;
    bool inv_v_   = false;
    bool swap_uv_ = false;
    BTRC_OBJECT(Texture2D, tex_);
};

class TransformTexture2DCreator : public factory::Creator<Texture2D>
{
public:

    std::string get_name() const override { return "transform"; }

    RC<Texture2D> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
