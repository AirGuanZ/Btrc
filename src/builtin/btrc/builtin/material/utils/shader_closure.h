#pragma once

#include <btrc/core/material.h>

BTRC_BEGIN

template<typename ShaderImpl>
class ShaderClosure : public Shader
{
    RC<const Object> material_;
    ShaderImpl       impl_;

public:

    ShaderClosure(RC<const Object> material, ref<ShaderImpl> impl)
        : material_(std::move(material)), impl_(impl)
    {
        
    }

    SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     wo,
        ref<CVec3f>     sam,
        TransportMode   mode) const override
    {
        return cc.record_object_action(
            material_, mode == TransportMode::Radiance ?
            "sample_radiance" : "sample_importance",
            [mode](ref<ShaderImpl> impl, ref<CVec3f> _wo, ref<CVec3f> _sam)
            { return impl.sample(_wo, _sam, mode); }, ref(impl_), wo, sam);
    }

    CSpectrum eval(
        CompileContext &cc,
        ref<CVec3f>     wi,
        ref<CVec3f>     wo,
        TransportMode   mode) const override
    {
        return cc.record_object_action(
            material_, mode == TransportMode::Radiance ?
            "eval_radiance" : "eval_importance",
            [mode](ref<ShaderImpl> impl, ref<CVec3f> _wi, ref<CVec3f> _wo)
            { return impl.eval(_wi, _wo, mode); }, ref(impl_), wi, wo);
    }

    f32 pdf(
        CompileContext &cc,
        ref<CVec3f>     wi,
        ref<CVec3f>     wo,
        TransportMode   mode) const override
    {
        return cc.record_object_action(
            material_, mode == TransportMode::Radiance ?
            "pdf_radiance" : "pdf_importance",
            [mode](ref<ShaderImpl> impl, ref<CVec3f> _wi, ref<CVec3f> _wo)
            { return impl.pdf(_wi, _wo, mode); }, ref(impl_), wi, wo);
    }

    CSpectrum albedo(CompileContext &cc) const override
    {
        return cc.record_object_action(
            material_, "albedo",
            [](ref<ShaderImpl> impl) { return impl.albedo(); }, ref(impl_));
    }

    CVec3f normal(CompileContext &cc) const override
    {
        return cc.record_object_action(
            material_, "normal",
            [](ref<ShaderImpl> impl) { return impl.normal(); }, ref(impl_));
    }
};

BTRC_END
