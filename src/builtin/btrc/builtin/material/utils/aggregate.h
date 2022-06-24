#pragma once

#include <btrc/builtin/material/utils/shader_frame.h>
#include <btrc/core/material.h>

BTRC_BUILTIN_BEGIN

class BSDFComponent
{
public:

    CUJ_CLASS_BEGIN(SampleResult)
        CUJ_MEMBER_VARIABLE(CSpectrum, bsdf)
        CUJ_MEMBER_VARIABLE(CVec3f,    dir)
        CUJ_MEMBER_VARIABLE(f32,       pdf)
        void clear()
        {
            bsdf = CSpectrum::zero();
            dir = CVec3f(0);
            pdf = 0;
        }
    CUJ_CLASS_END
    
    CUJ_CLASS_BEGIN(SampleBidirResult)
        CUJ_MEMBER_VARIABLE(CSpectrum, bsdf)
        CUJ_MEMBER_VARIABLE(CVec3f,    dir)
        CUJ_MEMBER_VARIABLE(f32,       pdf)
        CUJ_MEMBER_VARIABLE(f32,       pdf_rev)
        void clear()
        {
            bsdf = CSpectrum::zero();
            dir = CVec3f(0);
            pdf = 0;
            pdf_rev = 0;
        }
    CUJ_CLASS_END

    virtual ~BSDFComponent() = default;

    virtual SampleResult sample(CompileContext &cc, ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const = 0;

    virtual SampleBidirResult sample_bidir(
        CompileContext &cc, ref<CVec3f> lwo, ref<Sam3> sam, TransportMode   mode) const = 0;

    virtual CSpectrum eval(CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const = 0;

    virtual f32 pdf(CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const = 0;

    virtual CSpectrum albedo(CompileContext &cc) const = 0;

    static SampleResult discard_pdf_rev(const SampleBidirResult &result);
};

template<typename Impl>
class BSDFComponentClosure : public BSDFComponent
{
    RC<const Object> material_;
    std::string      name_;
    Impl             impl_;

public:

    BSDFComponentClosure(RC<const Object> material, std::string name, ref<Impl> impl);

    SampleResult sample(CompileContext &cc, ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const override;

    SampleBidirResult sample_bidir(
        CompileContext &cc, ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const override;

    CSpectrum eval(CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const override;

    f32 pdf(CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const override;

    CSpectrum albedo(CompileContext &cc) const override;
};

class BSDFAggregate : public Shader
{
public:

    BSDFAggregate(
        CompileContext  &cc,
        RC<const Object> material,
        ShaderFrame      frame,
        bool             shadow_terminator_term);

    void add_component(f32 sample_weight, Box<const BSDFComponent> comp);

    template<typename Impl>
    void add_closure(f32 sample_weight, std::string name, const Impl &impl);

    SampleResult sample(CompileContext &cc, ref<CVec3f> wo, ref<Sam3> sam, TransportMode mode) const override;

    SampleBidirResult sample_bidir(CompileContext &cc, ref<CVec3f> wo, ref<Sam3> sam, TransportMode mode) const override;

    CSpectrum eval(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const override;

    f32 pdf(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const override;

    CSpectrum albedo(CompileContext &cc) const override;

    CVec3f normal(CompileContext &cc) const override;

private:

    f32 eval_shadow_terminator_term(ref<CVec3f> wi) const;

    struct Component
    {
        f32 sample_weight;
        Box<const BSDFComponent> component;
    };

    CompileContext        &cc_;
    mutable CSpectrum      albedo_;
    mutable f32            sum_weight_;
    RC<const Object>       material_;
    ShaderFrame            frame_;
    std::vector<Component> components_;
    bool                   shadow_terminator_term_;
};

// ========================== impl ==========================

template<typename Impl>
BSDFComponentClosure<Impl>::BSDFComponentClosure(
    RC<const Object> material, std::string name, ref<Impl> impl)
    : material_(std::move(material)), name_(std::move(name)), impl_(impl)
{
    
}

template<typename Impl>
BSDFComponent::SampleResult BSDFComponentClosure<Impl>::sample(
    CompileContext &cc, ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const
{
    const std::string name = fmt::format(
        "sample_component_{}_{}", name_,
        mode == TransportMode::Radiance ? "radiance" : "importance");
    auto action = [mode](ref<Impl> impl, ref<CVec3f> _lwo, ref<Sam3> _sam)
    { return impl.sample(_lwo, _sam, mode); };
    return cc.record_object_action(material_, name, action, ref(impl_), lwo, sam);
}

template<typename Impl>
BSDFComponent::SampleBidirResult BSDFComponentClosure<Impl>::sample_bidir(
    CompileContext &cc, ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const
{
    const std::string name = fmt::format(
        "sample_bidir_component_{}_{}", name_,
        mode == TransportMode::Radiance ? "radiance" : "importance");
    auto action = [mode](ref<Impl> impl, ref<CVec3f> _lwo, ref<Sam3> _sam)
    { return impl.sample_bidir(_lwo, _sam, mode); };
    return cc.record_object_action(material_, name, action, ref(impl_), lwo, sam);
}

template<typename Impl>
CSpectrum BSDFComponentClosure<Impl>::eval(
    CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const
{
    const std::string name = fmt::format(
        "eval_component_{}_{}", name_,
        mode == TransportMode::Radiance ? "radiance" : "importance");
    auto action = [mode](ref<Impl> impl, ref<CVec3f> _lwi, ref<CVec3f> _lwo)
    { return impl.eval(_lwi, _lwo, mode); };
    return cc.record_object_action(
        material_, name, action, ref(impl_), lwi, lwo);
}

template<typename Impl>
f32 BSDFComponentClosure<Impl>::pdf(CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const
{
    const std::string name = fmt::format(
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
    const std::string name = fmt::format("albedo_component_{}", name_);
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
