#include <btrc/builtin/material/utils/aggregate.h>

BTRC_BUILTIN_BEGIN

BSDFAggregate::BSDFAggregate(RC<const Object> material, boolean is_delta, ShaderFrame frame)
    : is_dirty_(false), material_(std::move(material)), is_delta_(is_delta), frame_(frame)
{

}

void BSDFAggregate::add_component(f32 sample_weight, Box<const BSDFComponent> comp)
{
    components_.push_back({ sample_weight, std::move(comp) });
    is_dirty_ = true;
}

Shader::SampleResult BSDFAggregate::sample(ref<CVec3f> wo, ref<CVec3f> sam, TransportMode mode) const
{
    preprocess();

    $declare_scope;
    SampleResult result;

    $if(frame_.is_black_fringes(wo))
    {
        result = frame_.sample_black_fringes(wo, sam, albedo());
        $exit_scope;
    };

    var lwo = normalize(frame_.shading.global_to_local(wo));
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
                result = c.component->sample(lwo, new_sam, mode);
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
        result = c.component->sample(lwo, new_sam, mode);
    }

    $if(result.pdf <= 0)
    {
        result.clear();
        $exit_scope;
    };

    var bsdf = result.bsdf;
    var lwi = result.dir;
    var pdf = selected_comp_weight * result.pdf;

    for(size_t i = 0; i < components_.size(); ++i)
    {
        auto &c = components_[i];
        $if(selected_comp != static_cast<int>(i))
        {
            bsdf = bsdf + c.component->eval(lwi, lwo, mode);
            pdf = pdf + c.sample_weight * c.component->pdf(lwi, lwo, mode);
        };
    }

    result.bsdf = bsdf;
    result.dir = frame_.shading.local_to_global(lwi);
    result.pdf = pdf / sum_weight_;
    var corr_factor = frame_.correct_shading_energy(result.dir);
    result.bsdf = result.bsdf * corr_factor;
    return result;
}

CSpectrum BSDFAggregate::eval(ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
{
    preprocess();
    $declare_scope;
    CSpectrum result;

    $if(frame_.is_black_fringes(wi, wo))
    {
        result = frame_.eval_black_fringes(wi, wo, albedo());
        $exit_scope;
    };

    var lwi = normalize(frame_.shading.global_to_local(wi));
    var lwo = normalize(frame_.shading.global_to_local(wo));
    for(auto &c : components_)
        result = result + c.component->eval(lwi, lwo, mode);
    var corr_factor = frame_.correct_shading_energy(wi);
    result = result * corr_factor;
    return result;
}

f32 BSDFAggregate::pdf(ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
{
    preprocess();
    $declare_scope;
    f32 result;

    $if(frame_.is_black_fringes(wi, wo))
    {
        result = frame_.pdf_black_fringes(wi, wo);
        $exit_scope;
    };

    var lwi = normalize(frame_.shading.global_to_local(wi));
    var lwo = normalize(frame_.shading.global_to_local(wo));
    result = 0;
    for(auto &c : components_)
        result = result + c.sample_weight * c.component->pdf(lwi, lwo, mode);
    return result / sum_weight_;
}

CSpectrum BSDFAggregate::albedo() const
{
    preprocess();
    return albedo_;
}

CVec3f BSDFAggregate::normal() const
{
    return frame_.shading.z;
}

boolean BSDFAggregate::is_delta() const
{
    return is_delta_;
}

void BSDFAggregate::preprocess() const
{
    if(!is_dirty_)
        return;
    is_dirty_ = false;
    
    sum_weight_ = 0;
    for(auto &c : components_)
        sum_weight_ = sum_weight_ + c.sample_weight;

    albedo_ = CSpectrum::zero();
    for(auto &c : components_)
        albedo_ = albedo_ + c.component->albedo();
}

BTRC_BUILTIN_END
