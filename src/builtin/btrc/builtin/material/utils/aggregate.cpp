#include <btrc/builtin/material/utils/aggregate.h>

BTRC_BUILTIN_BEGIN

BSDFAggregate::BSDFAggregate(CompileContext &cc, RC<const Object> material, ShaderFrame frame)
    : cc_(cc), material_(std::move(material)), frame_(frame)
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
    SampleResult result;

    $if(frame_.is_black_fringes(wo))
    {
        result = frame_.sample_black_fringes(wo, sam, albedo(cc));
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
                result = c.component->sample(cc, lwo, new_sam, mode);
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
        result = c.component->sample(cc, lwo, new_sam, mode);
    }

    $if(result.pdf <= 0)
    {
        result.clear();
        $exit_scope;
    };

    $if(result.is_delta)
    {
        result.dir = frame_.shading.local_to_global(result.dir);
        result.pdf = selected_comp_weight * result.pdf / sum_weight_;
        var corr_factor = frame_.correct_shading_energy(result.dir);
        result.bsdf = result.bsdf * corr_factor;
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
            bsdf = bsdf + c.component->eval(cc, lwi, lwo, mode);
            pdf = pdf + c.sample_weight * c.component->pdf(cc, lwi, lwo, mode);
        };
    }

    result.bsdf = bsdf;
    result.dir = frame_.shading.local_to_global(lwi);
    result.pdf = pdf / sum_weight_;
    var corr_factor = frame_.correct_shading_energy(result.dir);
    result.bsdf = result.bsdf * corr_factor;
    return result;
}

CSpectrum BSDFAggregate::eval(
    CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
{
    $declare_scope;
    CSpectrum result;

    $if(frame_.is_black_fringes(wi, wo))
    {
        result = frame_.eval_black_fringes(wi, wo, albedo(cc));
        $exit_scope;
    };

    var lwi = normalize(frame_.shading.global_to_local(wi));
    var lwo = normalize(frame_.shading.global_to_local(wo));
    for(auto &c : components_)
        result = result + c.component->eval(cc, lwi, lwo, mode);
    var corr_factor = frame_.correct_shading_energy(wi);
    result = result * corr_factor;
    return result;
}

f32 BSDFAggregate::pdf(
    CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
{
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

BTRC_BUILTIN_END
