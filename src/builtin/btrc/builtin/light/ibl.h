#pragma once

#include <btrc/builtin/light/env_sampler.h>
#include <btrc/core/light.h>
#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class IBL : public EnvirLight
{
public:

    void set_texture(RC<Texture2D> tex);

    void set_up(const Vec3f &up);
    
    void set_lut_res(const Vec2i &lut_res);

    void commit() override;

    CSpectrum eval_le_inline(CompileContext &cc, ref<CVec3f> to_light) const override;

    SampleLiResult sample_li_inline(CompileContext &cc, ref<CVec3f> sam) const override;

    f32 pdf_li_inline(CompileContext &cc, ref<CVec3f> to_light) const override;

private:

    CSpectrum eval_local(CompileContext &cc, ref<CVec3f> normalized_to_light) const;

    BTRC_OBJECT(Texture2D, tex_);
    Frame                  frame_;
    Vec2i                  lut_res_;
    Box<EnvirLightSampler> sampler_;
};

class IBLCreator : public factory::Creator<Light>
{
public:

    std::string get_name() const override { return "ibl"; }

    RC<Light> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
