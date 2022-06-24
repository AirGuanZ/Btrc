#include <btrc/builtin/renderer/wavefront/helper.h>
#include <btrc/utils/intersection.h>

BTRC_WFPT_BEGIN

namespace
{

    template<typename F>
    boolean sample_light_li_impl(
        const WFPTScene &scene,
        const CVec3f    &ref_pos,
        Sampler         &sampler,
        ref<CVec3f>      d,
        ref<CVec3f>      o,
        ref<f32>         t1,
        ref<f32>         light_pdf,
        ref<CSpectrum>   li,
        const F         &dir_to_o)
    {
        var sam = sampler.get3d();
        const auto select_light = scene.scene.get_light_sampler()->sample(ref_pos, sampler.get1d());

        auto process_area_light = [&](const AreaLight *light)
        {
            auto sample = light->sample_li(scene.cc, ref_pos, sam);
            var diff = sample.position - ref_pos;
            var shadow_dst = intersection_offset(sample.position, sample.normal, -diff);
            o = dir_to_o(diff);
            d = shadow_dst - o;
            t1 = 1;
            li = sample.radiance;
            light_pdf = select_light.pdf * sample.pdf;
        };

        auto process_envir_light = [&](const EnvirLight *light)
        {
            auto sample = light->as_envir()->sample_li(scene.cc, sam);
            o = dir_to_o(sample.direction_to_light);
            d = sample.direction_to_light;
            t1 = btrc_max_float;
            li = sample.radiance;
            light_pdf = select_light.pdf * sample.pdf;
        };

        boolean result;
        $if(select_light.light_idx >= 0)
        {
            scene.scene.get_light_sampler()->access_light(select_light.light_idx, [&](const Light *light)
            {
                if(auto area = light->as_area())
                    process_area_light(area);
                else
                    process_envir_light(light->as_envir());
            });
            result = true;
        }
        $else
        {
            result = false;
        };
        return result;
    }

} // namespace anonymous

boolean simple_russian_roulette(
    ref<CSpectrum>                     path_beta,
    i32                                depth,
    GlobalSampler                     &sampler,
    const SimpleRussianRouletteParams &params)
{
    boolean ret = false;
    $if(depth >= params.min_depth)
    {
        $if(depth >= params.max_depth)
        {
            ret = true;
        }
        $else
        {
            var sam = sampler.get1d();
            $if(path_beta.get_lum() < params.beta_threshold)
            {
                $if(sam > params.cont_prob)
                {
                    ret = true;
                }
                $else
                {
                    path_beta = path_beta / params.cont_prob;
                };
            };
        };
    };
    return ret;
}

CSpectrum estimate_medium_tr(
    const WFPTScene &scene,
    CMediumID        medium_id,
    const CVec3f    &a,
    const CVec3f    &b,
    Sampler         &sampler)
{
    CSpectrum tr;
    scene.scene.access_medium(i32(medium_id), [&](const Medium *medium)
    {
        tr = medium->tr(scene.cc, a, b, a, b, sampler);
    });
    return tr;
}

SampleLiResult sample_medium_light_li(
    const WFPTScene &scene,
    const CVec3f    &ref_pos,
    Sampler         &sampler)
{
    SampleLiResult result;
    result.success = sample_light_li_impl(
        scene, ref_pos, sampler, result.d,
        result.o, result.t1, result.light_pdf, result.li,
        [&](const CVec3f &) { return ref_pos; });
    return result;
}

SampleLiResult sample_surface_light_li(
    const WFPTScene &scene,
    const CVec3f    &ref_pos,
    const CVec3f    &ref_nor,
    Sampler         &sampler)
{
    SampleLiResult result;
    result.success = sample_light_li_impl(
        scene, ref_pos, sampler, result.d,
        result.o, result.t1, result.light_pdf, result.li,
        [&](const CVec3f &dir)
        {
            return intersection_offset(ref_pos, ref_nor, dir);
        });
    return result;
}

CSpectrum eval_miss_le(
    const WFPTScene &scene,
    const CVec3f    &o,
    const CVec3f    &d,
    const CSpectrum &beta_le,
    f32              bsdf_pdf)
{
    var result = CSpectrum::zero();
    auto envir_light = scene.scene.get_light_sampler()->get_envir_light();
    if(envir_light)
    {
        var le = envir_light->eval_le(scene.cc, d);

        $if(bsdf_pdf < 0) // delta
        {
            result = beta_le * le / -bsdf_pdf;
        }
        $else
        {
            var select_light_pdf = scene.scene.get_light_sampler()->pdf(
                o, scene.scene.get_light_sampler()->get_envir_light_index());
            var envir_light_pdf = envir_light->pdf_li(scene.cc, d);
            var light_pdf = select_light_pdf * envir_light_pdf;

            result = beta_le * le / (bsdf_pdf + light_pdf);
        };
    }
    return result;
}

CSpectrum handle_intersected_light(
    const WFPTScene    &scene,
    const CVec3f       &o,
    const CVec3f       &d,
    const SurfacePoint &inct,
    const CSpectrum    &beta_le,
    f32                 bsdf_pdf,
    i32                 light_index)
{
    auto light_sampler = scene.scene.get_light_sampler();
    var result = CSpectrum::zero();
    $if(light_index >= 0)
    {
        var select_light_pdf = light_sampler->pdf(o, light_index);
        light_sampler->access_light(light_index, [&](const Light *light)
        {
            if(auto area = light->as_area())
            {
                var le = area->eval_le(scene.cc, inct, -d);
                $if(bsdf_pdf < 0)
                {
                    result = beta_le * le / -bsdf_pdf;
                }
                $else
                {
                    var light_dir_pdf = area->pdf_li(scene.cc, o, inct.position, inct.frame.z);
                    var light_pdf = select_light_pdf * light_dir_pdf;
                    result = beta_le * le / (bsdf_pdf + light_pdf);
                };
            }
        });
    };
    return result;
}

BTRC_WFPT_END
