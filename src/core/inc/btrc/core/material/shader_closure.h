#pragma once

#include <btrc/core/material/material.h>

BTRC_CORE_BEGIN

template<typename ShaderImpl>
class ShaderClosure : public Shader
{
    RC<const Material> material_;
    ShaderImpl         impl_;

public:

    ShaderClosure(RC<const Material> material, ref<ShaderImpl> impl)
        : material_(std::move(material)), impl_(impl)
    {
        
    }

    SampleResult sample(
        ref<CVec3f>   wo,
        ref<CVec3f>   sam,
        TransportMode mode) const override
    {
        return CompileContext::get_current_context()->record_object_action(
            material_, mode == TransportMode::Radiance ?
            "sample_radiance" : "sample_importance",
            [mode](ref<ShaderImpl> impl, ref<CVec3f> _wo, ref<CVec3f> _sam)
            { return impl.sample(_wo, _sam, mode); }, impl_, wo, sam);
    }

    CSpectrum eval(
        ref<CVec3f>   wi,
        ref<CVec3f>   wo,
        TransportMode mode) const override
    {
        return CompileContext::get_current_context()->record_object_action(
            material_, mode == TransportMode::Radiance ?
            "eval_radiance" : "eval_importance",
            [mode](ref<ShaderImpl> impl, ref<CVec3f> _wi, ref<CVec3f> _wo)
            { return impl.eval(_wi, _wo, mode); }, impl_, wi, wo);
    }

    f32 pdf(
        ref<CVec3f>   wi,
        ref<CVec3f>   wo,
        TransportMode mode) const override
    {
        return CompileContext::get_current_context()->record_object_action(
            material_, mode == TransportMode::Radiance ?
            "pdf_radiance" : "pdf_importance",
            [mode](ref<ShaderImpl> impl, ref<CVec3f> _wi, ref<CVec3f> _wo)
            { return impl.pdf(_wi, _wo, mode); }, impl_, wi, wo);
    }

    CSpectrum albedo() const override
    {
        return CompileContext::get_current_context()->record_object_action(
            material_, "albedo",
            [](ref<ShaderImpl> impl) { return impl.albedo(); }, impl_);
    }

    CVec3f normal() const override
    {
        return CompileContext::get_current_context()->record_object_action(
            material_, "normal",
            [](ref<ShaderImpl> impl) { return impl.normal(); }, impl_);
    }

    boolean is_delta() const override
    {
        return CompileContext::get_current_context()->record_object_action(
            material_, "is_delta",
            [](ref<ShaderImpl> impl) { return impl.is_delta(); }, impl_);
    }
};

BTRC_CORE_END
