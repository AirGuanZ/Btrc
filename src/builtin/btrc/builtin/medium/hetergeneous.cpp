#include <btrc/builtin/medium/henyey_greenstein.h>
#include <btrc/builtin/medium/hetergeneous.h>

BTRC_BUILTIN_BEGIN

void HetergeneousMedium::set_priority(float priority)
{
    priority_ = priority;
    set_recompile();
}

void HetergeneousMedium::set_sigma_t(RC<Texture3D> sigma_t)
{
    sigma_t_ = std::move(sigma_t);
}

void HetergeneousMedium::set_albedo(RC<Texture3D> albedo)
{
    albedo_ = std::move(albedo);
}

void HetergeneousMedium::set_g(RC<Texture3D> g)
{
    g_ = std::move(g);
}

Medium::SampleResult HetergeneousMedium::sample(
    CompileContext &cc,
    ref<CVec3f>     a,
    ref<CVec3f>     b,
    ref<CVec3f>     uvw_a,
    ref<CVec3f>     uvw_b,
    ref<CRNG>       rng) const
{
    var t_max = length(b - a), t = 0.0f;
    var max_density = cstd::max(sigma_t_->get_max_float(cc), 0.001f);
    var inv_max_density = 1.0f / max_density;
    
    var local_a = CVec3f(0.5f) + 0.5f * uvw_a;
    var local_b = CVec3f(0.5f) + 0.5f * uvw_b;

    SampleResult result;
    auto shader = newRC<HenyeyGreensteinPhaseShader>();
    result.shader = shader;

    $loop
    {
        var dt = -cstd::log(1.0f - rng.uniform_float()) * inv_max_density;
        t = t + dt;
        $if(t >= t_max)
        {
            result.scattered = false;
            result.throughput = CSpectrum::one();
            $break;
        };

        var tf = t / t_max;
        var uvw = local_a * (1.0f - tf) + local_b * tf;
        var density = sigma_t_->sample_float(cc, uvw);
        $if(rng.uniform_float() < density * inv_max_density)
        {
            var albedo = albedo_->sample_spectrum(cc, uvw);
            var g = g_->sample_float(cc, uvw);
            result.scattered = true;
            result.position = a * (1.0f - tf) + b * tf;
            result.throughput = albedo;
            shader->set_g(g);
            $break;
        };
    };

    return result;
}

CSpectrum HetergeneousMedium::tr(
    CompileContext &cc,
    ref<CVec3f>     a,
    ref<CVec3f>     b,
    ref<CVec3f>     uvw_a,
    ref<CVec3f>     uvw_b,
    ref<CRNG>       rng) const
{
    var result = 1.0f, t = 0.0f, t_max = length(b - a);
    var max_density = cstd::max(sigma_t_->get_max_float(cc), 0.001f);
    var inv_max_density = 1.0f / max_density;

    var local_a = CVec3f(0.5f) + 0.5f * uvw_a;
    var local_b = CVec3f(0.5f) + 0.5f * uvw_b;

    $loop
    {
        var dt = -cstd::log(1.0f - rng.uniform_float()) * inv_max_density;
        t = t + dt;
        $if(t >= t_max)
        {
            $break;
        };

        var tf = t / t_max;
        var uvw = local_a * (1.0f - tf) + local_b * tf;
        var density = sigma_t_->sample_float(cc, uvw);
        result = result * density * inv_max_density;
    };

    return CSpectrum::from_rgb(result, result, result);
}

float HetergeneousMedium::get_priority() const
{
    return priority_;
}

RC<Medium> HetergeneousMediumCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const auto priority = node->parse_child_or("priority", 0.0f);
    auto sigma_t = context.create<Texture3D>(node->child_node("sigma_t"));
    auto albedo = context.create<Texture3D>(node->child_node("albedo"));

    RC<Texture3D> g;
    if(auto tnode = node->find_child_node("g"))
        g = context.create<Texture3D>(tnode);
    else
    {
        auto c = newRC<Constant3D>();
        c->set_value(0.0f);
        g = std::move(c);
    }

    auto result = newRC<HetergeneousMedium>();
    result->set_priority(priority);
    result->set_sigma_t(std::move(sigma_t));
    result->set_albedo(std::move(albedo));
    result->set_g(std::move(g));
    return result;
}

BTRC_BUILTIN_END
