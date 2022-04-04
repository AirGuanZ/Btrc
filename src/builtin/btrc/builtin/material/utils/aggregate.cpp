#include <btrc/builtin/material/utils/aggregate.h>

BTRC_BUILTIN_BEGIN

BSDFComponent::SampleResult BSDFComponent::discard_pdf_rev(const SampleBidirResult &result)
{
    SampleResult ret;
    ret.bsdf = result.bsdf;
    ret.dir = result.dir;
    ret.pdf = result.pdf;
    return ret;
}

BSDFAggregate::BSDFAggregate(
    CompileContext  &cc,
    RC<const Object> material,
    ShaderFrame      frame,
    bool             shadow_terminator_term)
    : cc_(cc), material_(std::move(material)), frame_(frame), shadow_terminator_term_(shadow_terminator_term)
{
    sum_weight_ = 0.0f;
    albedo_ = CSpectrum::zero();
}

void BSDFAggregate::add_component(f32 sample_weight, Box<const BSDFComponent> comp)
{
    sum_weight_ = sum_weight_ + sample_weight;
    albedo_ = albedo_ + comp->albedo(cc_);
    components_.push_back({ sample_weight, std::move(comp) });
}

Shader::SampleResult BSDFAggregate::sample(
    CompileContext &cc, ref<CVec3f> wo, ref<CVec3f> sam, TransportMode mode) const
{
    $declare_scope;
    BSDFComponent::SampleResult comp_result;
    SampleResult result;

    var frame = frame_.flip_for_black_fringes(wo);
    var lwo = normalize(frame.shading.global_to_local(wo));
    var comp_selector = sum_weight_ * sam.x;
    var selected_comp = -1;
    var selected_comp_weight = 0.0f;

    {
        $declare_scope;
        for(size_t i = 0; i + 1 < components_.size(); ++i)
        {
            auto &c = components_[i];
            $if(comp_selector <= c.sample_weight)
            {
                var new_sam = CVec3f(comp_selector / c.sample_weight, sam.y, sam.z);
                comp_result = c.component->sample(cc, lwo, new_sam, mode);
                selected_comp = static_cast<int>(i);
                selected_comp_weight = c.sample_weight;
                $exit_scope;
            }
            $else
            {
                comp_selector = comp_selector - c.sample_weight;
            };
        }
        auto &c = components_.back();
        var new_sam = CVec3f(comp_selector / c.sample_weight, sam.y, sam.z);
        selected_comp = static_cast<int>(components_.size() - 1);
        selected_comp_weight = c.sample_weight;
        comp_result = c.component->sample(cc, lwo, new_sam, mode);
    }

    $if(comp_result.pdf <= 0)
    {
        result.clear();
        $exit_scope;
    };

    var bsdf = comp_result.bsdf;
    var lwi = comp_result.dir;
    var pdf = selected_comp_weight * comp_result.pdf;

    for(size_t i = 0; i < components_.size(); ++i)
    {
        auto &c = components_[i];
        $if(selected_comp != static_cast<int>(i))
        {
            bsdf = bsdf + c.component->eval(cc, lwi, lwo, mode);
            pdf = pdf + c.sample_weight * c.component->pdf(cc, lwi, lwo, mode);
        };
    }

    result.dir = frame.shading.local_to_global(lwi);
    $if(frame_.is_black_fringes(result.dir))
    {
        result.clear();
    };

    result.bsdf = bsdf;
    result.pdf = pdf / sum_weight_;
    var corr_factor = frame.correct_shading_energy(result.dir);
    var shadow_terminator_term = eval_shadow_terminator_term(result.dir);
    result.bsdf = result.bsdf * corr_factor * shadow_terminator_term;
    result.is_delta = false;

    return result;
}

Shader::SampleBidirResult BSDFAggregate::sample_bidir(
    CompileContext &cc, ref<CVec3f> wo, ref<CVec3f> sam, TransportMode mode) const
{
    $declare_scope;
    BSDFComponent::SampleBidirResult comp_result;
    SampleBidirResult result;

    var frame = frame_.flip_for_black_fringes(wo);
    var lwo = normalize(frame.shading.global_to_local(wo));
    var comp_selector = sum_weight_ * sam.x;
    var selected_comp = -1;
    var selected_comp_weight = 0.0f;

    {
        $declare_scope;
        for(size_t i = 0; i + 1 < components_.size(); ++i)
        {
            auto &c = components_[i];
            $if(comp_selector <= c.sample_weight)
            {
                var new_sam = CVec3f(comp_selector / c.sample_weight, sam.y, sam.z);
                comp_result = c.component->sample_bidir(cc, lwo, new_sam, mode);
                selected_comp = static_cast<int>(i);
                selected_comp_weight = c.sample_weight;
                $exit_scope;
            }
            $else
            {
                comp_selector = comp_selector - c.sample_weight;
            };
        }
        auto &c = components_.back();
        var new_sam = CVec3f(comp_selector / c.sample_weight, sam.y, sam.z);
        selected_comp = static_cast<int>(components_.size() - 1);
        selected_comp_weight = c.sample_weight;
        comp_result = c.component->sample_bidir(cc, lwo, new_sam, mode);
    }

    $if(comp_result.pdf <= 0)
    {
        result.clear();
        $exit_scope;
    };

    var bsdf = comp_result.bsdf;
    var lwi = comp_result.dir;
    var pdf = selected_comp_weight * comp_result.pdf;
    var pdf_rev = selected_comp_weight * comp_result.pdf_rev;

    for(size_t i = 0; i < components_.size(); ++i)
    {
        auto &c = components_[i];
        $if(selected_comp != static_cast<int>(i))
        {
            bsdf = bsdf + c.component->eval(cc, lwi, lwo, mode);
            pdf = pdf + c.sample_weight * c.component->pdf(cc, lwi, lwo, mode);
            pdf_rev = pdf + c.sample_weight * c.component->pdf(cc, lwo, lwi, mode);
        };
    }

    result.dir = frame.shading.local_to_global(lwi);
    $if(frame_.is_black_fringes(result.dir))
    {
        result.clear();
        $exit_scope;
    };

    result.bsdf = bsdf;
    result.pdf = pdf / sum_weight_;
    result.pdf_rev = pdf_rev / sum_weight_;
    var corr_factor = frame.correct_shading_energy(result.dir);
    var shadow_terminator_term = eval_shadow_terminator_term(result.dir);
    result.bsdf = result.bsdf * corr_factor * shadow_terminator_term;
    result.is_delta = false;

    return result;
}

CSpectrum BSDFAggregate::eval(
    CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
{
    $declare_scope;
    CSpectrum result;

    var frame = frame_.flip_for_black_fringes(wo);
    var lwi = normalize(frame.shading.global_to_local(wi));
    var lwo = normalize(frame.shading.global_to_local(wo));
    for(auto &c : components_)
        result = result + c.component->eval(cc, lwi, lwo, mode);
    var corr_factor = frame.correct_shading_energy(wi);
    var shadow_terminator_term = eval_shadow_terminator_term(wi);
    result = result * corr_factor * shadow_terminator_term;
    return result;
}

f32 BSDFAggregate::pdf(
    CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
{
    $declare_scope;
    f32 result;

    var frame = frame_.flip_for_black_fringes(wo);
    var lwi = normalize(frame.shading.global_to_local(wi));
    var lwo = normalize(frame.shading.global_to_local(wo));
    result = 0;
    for(auto &c : components_)
        result = result + c.sample_weight * c.component->pdf(cc, lwi, lwo, mode);
    result = result / sum_weight_;
    return result;
}

CSpectrum BSDFAggregate::albedo(CompileContext &cc) const
{
    return albedo_;
}

CVec3f BSDFAggregate::normal(CompileContext &cc) const
{
    return frame_.shading.z;
}

f32 BSDFAggregate::eval_shadow_terminator_term(ref<CVec3f> wi) const
{
    if(!shadow_terminator_term_)
        return 1.0f;
    var wgwi = dot(frame_.geometry.z, wi);
    var wswi = dot(frame_.shading.z, wi);
    var wgws = dot(frame_.geometry.z, frame_.shading.z);
    var g = cstd::saturate(wgwi / (wswi * wgws));
    return -g * g * g + g * g + g;
}

BTRC_BUILTIN_END
