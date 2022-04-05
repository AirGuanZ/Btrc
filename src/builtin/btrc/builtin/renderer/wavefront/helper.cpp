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
        $declare_scope;
        boolean result;

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

        $if(select_light.light_idx >= 0)
        {
            $switch(select_light.light_idx)
            {
                for(int i = 0; i < scene.scene.get_light_sampler()->get_light_count(); ++i)
                {
                    $case(i)
                    {
                        auto light = scene.scene.get_light_sampler()->get_light(i);
                        if(auto area = light->as_area())
                            process_area_light(area);
                        else
                            process_envir_light(light->as_envir());
                    };
                }
            };
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
    $switch(medium_id)
    {
        for(int i = 0; i < scene.scene.get_medium_count(); ++i)
        {
            $case(i)
            {
                tr = scene.scene.get_medium(i)->tr(scene.cc, a, b, a, b, sampler);
            };
        }
        $default
        {
            cstd::unreachable();
        };
    };
    return tr;
}

boolean sample_light_li(
    const WFPTScene &scene,
    const CVec3f    &ref_pos,
    Sampler         &sampler,
    ref<CVec3f>      d,
    ref<f32>         t1,
    ref<f32>         light_pdf,
    ref<CSpectrum>   li)
{
    return sample_light_li_impl(
        scene, ref_pos, sampler,
        d, CVec3f(0), t1, light_pdf, li,
        [&](const CVec3f &) { return ref_pos; });
}

boolean sample_light_li(
    const WFPTScene &scene,
    const CVec3f    &ref_pos,
    const CVec3f    &ref_nor,
    Sampler         &sampler,
    ref<CVec3f>      o,
    ref<CVec3f>      d,
    ref<f32>         t1,
    ref<f32>         light_pdf,
    ref<CSpectrum>   li)
{
    return sample_light_li_impl(
        scene, ref_pos, sampler,
        d, o, t1, light_pdf, li,
        [&](const CVec3f &dir)
        {
            return intersection_offset(ref_pos, ref_nor, dir);
        });
}

CSpectrum handle_miss(
    const WFPTScene &scene,
    const CVec3f    &o,
    const CVec3f    &d,
    const CSpectrum &beta_le,
    f32              bsdf_pdf)
{
    $declare_scope;
    CSpectrum result;

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
    var result = CSpectrum::zero();

    auto light_sampler = scene.scene.get_light_sampler();
    auto handle_light = [&](i32 light_id, const AreaLight *area)
    {
        var le = area->eval_le(scene.cc, inct.position, inct.frame.z, inct.uv, inct.tex_coord, -d);
        $if(bsdf_pdf < 0)
        {
            result = beta_le * le / -bsdf_pdf;
        }
        $else
        {
            var select_light_pdf = light_sampler->pdf(o, light_id);
            var light_dir_pdf = area->pdf_li(scene.cc, o, inct.position, inct.frame.z);
            var light_pdf = select_light_pdf * light_dir_pdf;
            result = beta_le * le / (bsdf_pdf + light_pdf);
        };
    };

    $if(light_index >= 0)
    {
        $switch(light_index)
        {
            for(int i = 0; i < light_sampler->get_light_count(); ++i)
            {
                if(auto area = light_sampler->get_light(i)->as_area())
                {
                    $case(i)
                    {
                        handle_light(i, area);
                    };
                }
            }
            $default
            {
                cstd::unreachable();
            };
        };
    };

    return result;
}

BTRC_WFPT_END
