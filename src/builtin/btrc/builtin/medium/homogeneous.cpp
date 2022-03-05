#include <btrc/builtin/medium/henyey_greenstein.h>
#include <btrc/builtin/medium/homogeneous.h>

BTRC_BUILTIN_BEGIN

void HomogeneousMedium::set_priority(float priority)
{
    priority_ = priority;
    set_recompile();
}

void HomogeneousMedium::set_sigma_t(float sigma_t)
{
    sigma_t_ = sigma_t;
}

void HomogeneousMedium::set_albedo(const Spectrum &albedo)
{
    albedo_ = albedo;
}

void HomogeneousMedium::set_g(float g)
{
    g_ = g;
}

Medium::SampleResult HomogeneousMedium::sample(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, ref<cstd::LCG> rng) const
{
    $declare_scope;
    SampleResult result;
    auto shader = newRC<HenyeyGreensteinPhaseShader>();
    result.shader = shader;

    var sigma_t = sigma_t_.read(cc);
    var albedo = albedo_.read(cc);
    var sigma_s = sigma_t * albedo;

    $if(sigma_s.is_zero())
    {
        result.scattered = false;
        result.throughput = tr(cc, a, b);
        $exit_scope;
    };

    var t_max = length(b - a);
    var st = -cstd::log(rng.uniform_float()) / sigma_t;

    $if(st < t_max)
    {
        var t = st / t_max;
        result.scattered = true;
        result.throughput = albedo;
        result.position = a * (1.0f - t) + b * t;
        shader->set_g(g_.read(cc));
    }
    $else
    {
        result.scattered = false;
        result.throughput = CSpectrum::one();
    };

    return result;
}

CSpectrum HomogeneousMedium::tr(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b) const
{
    var sigma_t = sigma_t_.read(cc);
    var albedo = albedo_.read(cc);
    var sigma_a = sigma_t * (CSpectrum::one() - albedo);
    var exp = -length(a - b) * sigma_a;
    return CSpectrum::from_rgb(cstd::exp(exp.r), cstd::exp(exp.g), cstd::exp(exp.b));
}

float HomogeneousMedium::get_priority() const
{
    return priority_;
}

RC<Medium> HomogeneousMediumCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const auto priority = node->parse_child_or("priority", 0.0f);
    const auto sigma_t = node->parse_child<float>("sigma_t");
    const auto albedo = node->parse_child<Spectrum>("albedo");
    const auto g = node->parse_child_or("g", 0.0f);
    auto result = newRC<HomogeneousMedium>();
    result->set_priority(priority);
    result->set_sigma_t(sigma_t);
    result->set_albedo(albedo);
    result->set_g(g);
    return result;
}

BTRC_BUILTIN_END
