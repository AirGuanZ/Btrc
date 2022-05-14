#include <btrc/builtin/renderer/pt/trace.h>
#include <btrc/utils/intersection.h>

BTRC_PT_BEGIN

namespace
{

    SurfacePoint get_hitinfo(
        const CVec3f        &o,
        const CVec3f        &d,
        const CInstanceInfo &instance,
        const CGeometryInfo &geometry,
        f32                  t,
        u32                  prim_id,
        const CVec2f        &uv)
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

    CUJ_CLASS_BEGIN(LeParams)
        CUJ_MEMBER_VARIABLE(CSpectrum, beta)
        CUJ_MEMBER_VARIABLE(f32,       bsdf_pdf)
        CUJ_MEMBER_VARIABLE(boolean,   is_delta)
    CUJ_CLASS_END

    CSpectrum accumulate_le(
        CompileContext     &cc,
        const EnvirLight   *env,
        const LightSampler *light_sampler,
        const CRay         &r,
        const LeParams     &le_params)
    {
        var le = env->eval_le(cc, r.d);
        var le_pdf = 0.0f;
        $if(!le_params.is_delta)
        {
            var pdf_select_env = light_sampler->pdf(r.o, light_sampler->get_envir_light_index());
            var pdf_env_dir = env->pdf_li(cc, r.d);
            le_pdf = pdf_select_env * pdf_env_dir;
        };
        return le_params.beta * le / (le_pdf + le_params.bsdf_pdf);
    }
    
    CSpectrum accumulate_le(
        CompileContext     &cc,
        const AreaLight    *area,
        const LightSampler *light_sampler,
        const CRay         &r,
        const SurfacePoint &hit_info,
        const LeParams     &le_params)
    {
        var le = area->eval_le(cc, hit_info.position, hit_info.frame.z, hit_info.uv, hit_info.tex_coord, -r.d);
        var le_pdf = 0.0f;
        $if(!le_params.is_delta)
        {
            var pdf_select_env = light_sampler->pdf(r.o, light_sampler->get_envir_light_index());
            var pdf_env_dir = area->pdf_li(cc, r.o, hit_info.position, hit_info.frame.z);
            le_pdf = pdf_select_env * pdf_env_dir;
        };
        return le_params.beta * le / (le_pdf + le_params.bsdf_pdf);
    }

    void sample_li(
        CompileContext &cc,
        const SurfacePoint &hit_info,
        const Light        *light,
        Sampler            &sampler,
        ref<CSpectrum>      li,
        ref<f32>            pdf,
        ref<CVec3f>         dir,
        ref<CRay>           shadow_ray)
    {
        if(auto area = light->as_area())
        {
            var sample = area->sample_li(cc, hit_info.position, sampler.get3d());
            li = sample.radiance;
            pdf = sample.pdf;
            dir = sample.position - hit_info.position;
            var src = intersection_offset(hit_info.position, hit_info.frame.z, dir);
            var dst = intersection_offset(sample.position, sample.normal, -dir);
            shadow_ray = CRay(src, dst - src, 1);
        }
        else
        {
            auto envir = light->as_envir();
            var sample = envir->sample_li(cc, sampler.get3d());
            li = sample.radiance;
            pdf = sample.pdf;
            dir = sample.direction_to_light;
            var src = intersection_offset(hit_info.position, hit_info.frame.z, dir);
            shadow_ray = CRay(src, dir);
        }
    }

} // namespace anonymous

TraceResult trace_path(
    CompileContext   &cc,
    const TraceUtils &utils,
    const Params     &params,
    const Scene      &scene,
    const CRay       &ray,
    GlobalSampler    &sampler)
{
    auto light_sampler = scene.get_light_sampler();
    TraceResult result;

    LeParams le_params;
    le_params.beta = CSpectrum::one();
    le_params.bsdf_pdf = 1;
    le_params.is_delta = false;

    i32 depth = 1;
    var beta = CSpectrum::one();
    var r = ray;

    $loop
    {
        var hit = utils.find_closest_intersection(r);
        $if(hit.miss())
        {
            if(auto env = light_sampler->get_envir_light())
            {
                var le_contrib = accumulate_le(cc, env.get(), light_sampler, r, le_params);
                result.radiance = result.radiance + le_contrib;
            }
            $break;
        };

        // hit info

        var instances = cuj::import_pointer(scene.get_device_instance_info());
        var geometries = cuj::import_pointer(scene.get_device_geometry_info());
        ref instance = instances[hit.inst_id];
        ref geometry = geometries[instance.geometry_id];
        var hit_info = get_hitinfo(r.o, r.d, instance, geometry, hit.t, hit.prim_id, hit.uv);

        // intersected area light

        $switch(instance.light_id)
        {
            for(int i = 0; i < light_sampler->get_light_count(); ++i)
            {
                auto area = light_sampler->get_light(i)->as_area();
                if(!area)
                    continue;

                $case(i)
                {
                    var le_contrib = accumulate_le(cc, area, light_sampler, r, hit_info, le_params);
                    result.radiance = result.radiance + le_contrib;
                };
            }
        };

        // apply rr

        $if(depth >= params.min_depth)
        {
            $if(depth > params.max_depth)
            {
                $break;
            };

            $if(beta.get_lum() < params.rr_threshold)
            {
                $if(sampler.get1d() > params.rr_cont_prob)
                {
                    $break;
                }
                $else
                {
                    beta = beta * (1.0f / params.rr_cont_prob);
                };
            };
        };

        // sample light

        CVec3f    li_dir;
        CSpectrum li_rad;
        f32       li_pdf;
        CRay      li_ray;
        CSpectrum li_bsdf;
        f32       li_bsdf_pdf;
        boolean   has_li = true;

        var select_light = light_sampler->sample(hit_info.position, sampler.get1d());
        $switch(select_light.light_idx)
        {
            for(int i = 0; i < light_sampler->get_light_count(); ++i)
            {
                $case(i)
                {
                    auto light = light_sampler->get_light(i);
                    sample_li(cc, hit_info, light.get(), sampler, li_rad, li_pdf, li_dir, li_ray);
                };
            }
            $default
            {
                has_li = false;
            };
        };

        has_li = has_li & !utils.has_intersection(li_ray);

        // sample bsdf

        Shader::SampleResult bsdf_sample;

        $switch(instance.material_id)
        {
            for(int i = 0; i < scene.get_material_count(); ++i)
            {
                $case(i)
                {
                    auto material = scene.get_material(i);
                    auto shader = material->create_shader(cc, hit_info);

                    // g-buffer

                    if(params.albedo || params.normal)
                    {
                        $if(depth == 1)
                        {
                            if(params.normal)
                                result.normal = shader->normal(cc);
                            if(params.albedo)
                                result.albedo = shader->albedo(cc);
                        };
                    }

                    // eval bsdf for li

                    $if(has_li)
                    {
                        li_bsdf = shader->eval(cc, li_dir, -r.d, TransportMode::Radiance);
                        li_bsdf_pdf = shader->pdf(cc, li_dir, -r.d, TransportMode::Radiance);
                    };

                    // sample next ray

                    bsdf_sample = shader->sample(cc, -r.d, sampler.get3d(), TransportMode::Radiance);
                };
            }
        };

        // li

        $if(has_li & !li_bsdf.is_zero())
        {
            var abscos = cstd::abs(cos(hit_info.frame.z, li_dir));
            var li_contrib = beta * li_rad * abscos * li_bsdf / (li_bsdf_pdf + li_pdf);
            result.radiance = result.radiance + li_contrib;
        };

        // next ray

        $if(bsdf_sample.bsdf.is_zero())
        {
            $break;
        };

        var abscos = cstd::abs(cos(hit_info.frame.z, bsdf_sample.dir));
        le_params.beta = beta * bsdf_sample.bsdf * abscos;
        le_params.bsdf_pdf = bsdf_sample.pdf;
        le_params.is_delta = bsdf_sample.is_delta;

        beta = le_params.beta / le_params.bsdf_pdf;

        r.o = intersection_offset(hit_info.position, hit_info.frame.z, bsdf_sample.dir);
        r.d = bsdf_sample.dir;

        depth = depth + 1;
    };

    return result;
}

BTRC_PT_END
