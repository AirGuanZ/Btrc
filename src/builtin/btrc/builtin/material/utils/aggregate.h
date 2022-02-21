#pragma once

#include <btrc/builtin/material/utils/shader_frame.h>
#include <btrc/core/material.h>

BTRC_BUILTIN_BEGIN

class BSDFComponent
{
public:

    virtual ~BSDFComponent() = default;

    virtual Shader::SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     lwo,
        ref<CVec3f>     sam,
        TransportMode   mode) const = 0;

    virtual CSpectrum eval(
        CompileContext &cc,
        ref<CVec3f>     lwi,
        ref<CVec3f>     lwo,
        TransportMode   mode) const = 0;

    virtual f32 pdf(
        CompileContext &cc,
        ref<CVec3f>     lwi,
        ref<CVec3f>     lwo,
        TransportMode   mode) const = 0;

    virtual CSpectrum albedo(CompileContext &cc) const = 0;
};

template<typename Impl>
class BSDFComponentClosure : public BSDFComponent
{
    RC<const Object> material_;
    std::string      name_;
    Impl             impl_;

public:

    BSDFComponentClosure(RC<const Object> material, std::string name, ref<Impl> impl);

    Shader::SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     lwo,
        ref<CVec3f>     sam,
        TransportMode   mode) const override;

    CSpectrum eval(
        CompileContext &cc,
        ref<CVec3f>     lwi,
        ref<CVec3f>     lwo,
        TransportMode   mode) const override;

    f32 pdf(
        CompileContext &cc,
        ref<CVec3f>     lwi,
        ref<CVec3f>     lwo,
        TransportMode   mode) const override;

    CSpectrum albedo(CompileContext &cc) const override;
};

class BSDFAggregate : public Shader
{
public:

    BSDFAggregate(RC<const Object> material, boolean is_delta, ShaderFrame frame);

    void add_component(f32 sample_weight, Box<const BSDFComponent> comp);

    template<typename Impl>
    void add_closure(f32 sample_weight, std::string name, const Impl &impl);

    SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     wo,
        ref<CVec3f>     sam,
        TransportMode   mode) const override;

    CSpectrum eval(
        CompileContext &cc,
        ref<CVec3f>     wi,
        ref<CVec3f>     wo,
        TransportMode   mode) const override;

    f32 pdf(CompileContext &cc,
        ref<CVec3f>         wi,
        ref<CVec3f>         wo,
        TransportMode       mode) const override;

    CSpectrum albedo(CompileContext &cc) const override;

    CVec3f normal(CompileContext &cc) const override;

    boolean is_delta(CompileContext &cc) const override;

private:

    void preprocess(CompileContext &cc) const;

    struct Component
    {
        f32 sample_weight;
        Box<const BSDFComponent> component;
    };

    mutable bool           is_dirty_;
    mutable CSpectrum      albedo_;
    mutable f32            sum_weight_;
    RC<const Object>       material_;
    boolean                is_delta_;
    ShaderFrame            frame_;
    std::vector<Component> components_;
};

// ========================== impl ==========================

template<typename Impl>
BSDFComponentClosure<Impl>::BSDFComponentClosure(
    RC<const Object> material, std::string name, ref<Impl> impl)
    : material_(std::move(material)), name_(std::move(name)), impl_(impl)
{
    
}

template<typename Impl>
Shader::SampleResult BSDFComponentClosure<Impl>::sample(
    CompileContext &cc, ref<CVec3f> lwo, ref<CVec3f> sam, TransportMode mode) const
{
    const std::string name = std::format(
        "sample_component_{}_{}", name_,
        mode == TransportMode::Radiance ? "radiance" : "importance");
    auto action = [mode](ref<Impl> impl, ref<CVec3f> _lwo, ref<CVec3f> _sam)
    { return impl.sample(_lwo, _sam, mode); };
    return cc.record_object_action(
        material_, name, action, ref(impl_), lwo, sam);
}

template<typename Impl>
CSpectrum BSDFComponentClosure<Impl>::eval(
    CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const
{
    const std::string name = std::format(
        "eval_component_{}_{}", name_,
        mode == TransportMode::Radiance ? "radiance" : "importance");
    auto action = [mode](ref<Impl> impl, ref<CVec3f> _lwi, ref<CVec3f> _lwo)
    { return impl.eval(_lwi, _lwo, mode); };
    return cc.record_object_action(
        material_, name, action, ref(impl_), lwi, lwo);
}

template<typename Impl>
f32 BSDFComponentClosure<Impl>::pdf(
    CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const
{
    const std::string name = std::format(
        "pdf_component_{}_{}", name_,
        mode == TransportMode::Radiance ? "radiance" : "importance");
    auto action = [mode](ref<Impl> impl, ref<CVec3f> _lwi, ref<CVec3f> _lwo)
    { return impl.pdf(_lwi, _lwo, mode); };
    return cc.record_object_action(
        material_, name, action, ref(impl_), lwi, lwo);
}

template<typename Impl>
CSpectrum BSDFComponentClosure<Impl>::albedo(CompileContext &cc) const
{
    const std::string name = std::format("albedo_component_{}", name_);
    auto action = [](ref<Impl> impl) { return impl.albedo(); };
    return cc.record_object_action(
        material_, name, action, ref(impl_));
}

template<typename Impl>
void BSDFAggregate::add_closure(f32 sample_weight, std::string name, const Impl &impl)
{
    auto comp = newBox<BSDFComponentClosure<Impl>>(material_, std::move(name), impl);
    this->add_component(sample_weight, std::move(comp));
}

BTRC_BUILTIN_END
