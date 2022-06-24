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
    var result = CSpectrum::zero();

    auto light_sampler = scene.scene.get_light_sampler();
    auto handle_light = [&](i32 light_id, const AreaLight *area)
    {
        var le = area->eval_le(scene.cc, inct, -d);
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

SurfacePoint get_intersection(
    const CVec3f &o,
    const CVec3f &d,
    const CInstanceInfo &instance,
    const CGeometryInfo &geometry,
    f32 t, u32 prim_id, const CVec2f &uv)
{
    ref local_to_world = instance.transform;

    // position

    var position = o + t * d;

    // geometry frame

    var gx_ua = load_aligned(geometry.geometry_ex_tex_coord_u_a + prim_id);
    var gy_uba = load_aligned(geometry.geometry_ey_tex_coord_u_ba + prim_id);
    var gz_uca = load_aligned(geometry.geometry_ez_tex_coord_u_ca + prim_id);

    var sn_v_a = load_aligned(geometry.shading_normal_tex_coord_v_a + prim_id);
    var sn_v_ba = load_aligned(geometry.shading_normal_tex_coord_v_ba + prim_id);
    var sn_v_ca = load_aligned(geometry.shading_normal_tex_coord_v_ca + prim_id);

    CFrame geometry_frame = CFrame(gx_ua.xyz(), gy_uba.xyz(), gz_uca.xyz());

    geometry_frame.x = local_to_world.apply_to_vector(geometry_frame.x);
    geometry_frame.y = local_to_world.apply_to_vector(geometry_frame.y);
    geometry_frame.z = local_to_world.apply_to_normal(geometry_frame.z);
    geometry_frame.x = normalize(geometry_frame.x);
    geometry_frame.y = normalize(geometry_frame.y);
    geometry_frame.z = normalize(geometry_frame.z);

    // interpolated normal

    var interp_normal = sn_v_a.xyz() + sn_v_ba.xyz() * uv.x + sn_v_ca.xyz() * uv.y;
    interp_normal = normalize(local_to_world.apply_to_normal(interp_normal));

    // tex coord

    var tex_coord_u = gx_ua.w + gy_uba.w * uv.x + gz_uca.w * uv.y;
    var tex_coord_v = sn_v_a.w + sn_v_ba.w * uv.x + sn_v_ca.w * uv.y;
    var tex_coord = CVec2f(tex_coord_u, tex_coord_v);

    // intersection

    SurfacePoint material_inct;
    material_inct.position = position;
    material_inct.frame = geometry_frame;
    material_inct.interp_z = interp_normal;
    material_inct.uv = uv;
    material_inct.tex_coord = tex_coord;

    return material_inct;
}

BTRC_WFPT_END
