#include <btrc/builtin/renderer/pt/trace.h>
#include <btrc/utils/intersection.h>

BTRC_PT_BEGIN

namespace
{

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
        var le = area->eval_le(cc, hit_info, -r.d);
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

    void sample_medium_li(
        CompileContext &cc,
        const CVec3f   &scatter_pos,
        const Light    *light,
        Sampler        &sampler,
        ref<CSpectrum>  li,
        ref<f32>        pdf,
        ref<CVec3f>     dir,
        ref<CRay>       shadow_ray)
    {
        if(auto area = light->as_area())
        {
            var sample = area->sample_li(cc, scatter_pos, sampler.get3d());
            li = sample.radiance;
            pdf = sample.pdf;
            dir = sample.position - scatter_pos;
            var src = scatter_pos;
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
            var src = scatter_pos;
            shadow_ray = CRay(src, dir);
        }
    }

} // namespace anonymous

TraceResult trace_path(
    CompileContext    &cc,
    const TraceUtils  &utils,
    const TraceParams &params,
    const Scene       &scene,
    const CRay        &ray,
    CMediumID          initial_ray_medium_id,
    GlobalSampler     &sampler,
    float              world_diagonal)
{
    auto light_sampler = scene.get_light_sampler();
    TraceResult result;

    LeParams le_params;
    le_params.beta = CSpectrum::one();
    le_params.bsdf_pdf = 1;
    le_params.is_delta = true;

    i32 depth = 1;
    var beta = CSpectrum::one();
    var r = ray;
    var r_medium_id = initial_ray_medium_id;

    $loop
    {
        var hit = utils.find_closest_intersection(r);

        var instances = cuj::import_pointer(scene.get_device_instance_info());
        var geometries = cuj::import_pointer(scene.get_device_geometry_info());
        ptr<CInstanceInfo> instance;
        ptr<CGeometryInfo> geometry;
        SurfacePoint hit_info;

        // resolve medium

        CMediumID medium_id; CVec3f medium_end;
        $if(hit.miss())
        {
            medium_id = scene.get_volume_primitive_medium_id();
            medium_end = r.o + world_diagonal * r.d;
        }
        $else
        {
            instance = instances[hit.inst_id].address();
            geometry = geometries[instance->geometry_id].address();
            hit_info = get_hitinfo(r.o, r.d, *instance, *geometry, hit.t, hit.prim_id, hit.uv);

            var is_inct_outer = dot(r.d, hit_info.frame.z) < 0;
            var inct_medium_id = cstd::select(is_inct_outer, instance->outer_medium_id, instance->inner_medium_id);
            
            medium_id = cstd::min(r_medium_id, inct_medium_id);
            medium_end = intersection_offset(hit_info.position, hit_info.frame.z, -r.d);
        };

        // sample medium

        boolean scattered = false;
        boolean scattered_rr_exit = false;
        CVec3f scatter_position;
        PhaseShader::SampleResult phase_sample;

        CVec3f    li_dir;
        CSpectrum li_rad;
        f32       li_pdf;
        CRay      li_ray;
        CSpectrum li_bsdf;
        f32       li_bsdf_pdf;
        boolean   has_li = true;

        scene.access_medium(i32(medium_id), [&](const Medium *medium)
        {
            auto sample_medium = medium->sample(cc, r.o, medium_end, sampler);

            // modify le_params using full transmittance

            var unscatter_tr = medium->tr(cc, r.o, medium_end, sampler);
            le_params.beta = le_params.beta * unscatter_tr;

            // handle scattering

            $if(sample_medium.scattered)
            {
                scattered = true;
                scatter_position = sample_medium.position;

                // rr termination

                $if(depth >= params.min_depth)
                {
                    $if(depth > params.max_depth)
                    {
                        scattered_rr_exit = true;
                    };

                    $if(beta.get_lum() < params.rr_threshold)
                    {
                        $if(sampler.get1d() > params.rr_cont_prob)
                        {
                            scattered_rr_exit = true;
                        }
                        $else
                        {
                            beta = beta * (1.0f / params.rr_cont_prob);
                        };
                    };
                };

                // shadow ray

                $if(!scattered_rr_exit)
                {
                    var select_light = light_sampler->sample(scatter_position, sampler.get1d());
                    $switch(select_light.light_idx)
                    {
                        for(int j = 0; j < light_sampler->get_light_count(); ++j)
                        {
                            $case(j)
                            {
                                auto light = light_sampler->get_light(j);
                                sample_medium_li(
                                    cc, scatter_position, light.get(), sampler, li_rad, li_pdf, li_dir, li_ray);
                            };
                        }
                        $default
                        {
                            has_li = false;
                        };
                    };
                };

                $if(has_li)
                {
                    li_bsdf = sample_medium.shader->eval(cc, li_dir, -r.d);
                    li_bsdf_pdf = sample_medium.shader->pdf(cc, li_dir, -r.d);
                };

                // next ray

                phase_sample = sample_medium.shader->sample(cc, -r.d, sampler.get3d());
            };
        });

        // accumulate direct le

        $if(hit.miss())
        {
            if(auto env = light_sampler->get_envir_light())
            {
                var le_contrib = accumulate_le(cc, env.get(), light_sampler, r, le_params);
                result.radiance = result.radiance + le_contrib;
            }
        }
        $elif(instance->light_id >= 0)
        {
            light_sampler->access_light(instance->light_id, [&](const Light *light)
            {
                if(auto area = light->as_area())
                {
                    var le_contrib = accumulate_le(cc, area, light_sampler, r, hit_info, le_params);
                    result.radiance = result.radiance + le_contrib;
                }
            });
        };

        // transmittance of shadow ray

        auto eval_shadow_ray_tr = [&](CMediumID shadow_medium_id, const CRay &shadow_ray)
        {
            CSpectrum tr;
            $if(shadow_ray.t > 1)
            {
                tr = scene.get_volume_primitive_medium()->tr(
                    cc, shadow_ray.o, shadow_ray.o + normalize(shadow_ray.d) * world_diagonal, sampler);
            }
            $else
            {
                var end_pnt = shadow_ray.o + shadow_ray.d * shadow_ray.t;
                scene.access_medium(i32(shadow_medium_id), [&](const Medium *medium)
                {
                    tr = medium->tr(cc, shadow_ray.o, end_pnt, sampler);
                });
            };
            return tr;
        };

        $if(scattered)
        {
            // TODO: albedo/normal is not accumulated in this case

            $if(scattered_rr_exit)
            {
                $break;
            };

            // shadow ray

            has_li = has_li & !utils.has_intersection(li_ray);
            $if(has_li & !li_bsdf.is_zero())
            {
                var tr = eval_shadow_ray_tr(medium_id, li_ray);
                var li_contrib = tr * beta * li_rad * li_bsdf / (li_bsdf_pdf + li_pdf);
                result.radiance = result.radiance + li_contrib;
            };

            // next ray

            $if(!phase_sample.phase.is_zero())
            {
                var beta_le = beta * phase_sample.phase;
                beta = beta_le / phase_sample.pdf;

                le_params.is_delta = false;
                le_params.beta = beta_le;
                le_params.bsdf_pdf = phase_sample.pdf;

                r.o = scatter_position;
                r.d = normalize(phase_sample.dir);

                r_medium_id = medium_id;

                depth = depth + 1;

                $continue;
            };

            $break;
        };

        $if(hit.miss())
        {
            $break;
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

        var select_light = light_sampler->sample(hit_info.position, sampler.get1d());
        $if(select_light.light_idx >= 0)
        {
            light_sampler->access_light(select_light.light_idx, [&](const Light *light)
            {
                sample_li(cc, hit_info, light, sampler, li_rad, li_pdf, li_dir, li_ray);
            });
        }
        $else
        {
            has_li = false;
        };

        has_li = has_li & !utils.has_intersection(li_ray);

        // sample bsdf

        Shader::SampleResult bsdf_sample;

        scene.access_material(instance->material_id, [&](const Material *material)
        {
            auto shader = material->create_shader(cc, hit_info);

            // g-buffer
            
            $if(depth == 1)
            {
                if(params.normal)
                    result.normal = shader->normal(cc);
                if(params.albedo)
                    result.albedo = shader->albedo(cc);
            };
            
            // eval bsdf for li
            
            $if(has_li)
            {
                li_bsdf = shader->eval(cc, li_dir, -r.d, TransportMode::Radiance);
                li_bsdf_pdf = shader->pdf(cc, li_dir, -r.d, TransportMode::Radiance);
            };
            
            // sample next ray
            
            bsdf_sample = shader->sample(cc, -r.d, sampler.get3d(), TransportMode::Radiance);
        });

        // li

        $if(has_li & !li_bsdf.is_zero())
        {
            CMediumID shadow_medium_id;
            $if(dot(r.d, li_dir) < 0)
            {
                shadow_medium_id = medium_id;
            }
            $else
            {
                $if(dot(li_dir, hit_info.frame.z) > 0)
                {
                    shadow_medium_id = instance->outer_medium_id;
                }
                $else
                {
                    shadow_medium_id = instance->inner_medium_id;
                };
            };
            var tr = eval_shadow_ray_tr(shadow_medium_id, li_ray);
            var abscos = cstd::abs(cos(hit_info.frame.z, li_dir));
            var li_contrib = tr * beta * li_rad * abscos * li_bsdf / (li_bsdf_pdf + li_pdf);
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

        $if(dot(r.d, bsdf_sample.dir) < 0)
        {
            r_medium_id = medium_id;
        }
        $else
        {
            $if(dot(bsdf_sample.dir, hit_info.frame.z) > 0)
            {
                r_medium_id = instance->outer_medium_id;
            }
            $else
            {
                r_medium_id = instance->inner_medium_id;
            };
        };

        r.o = intersection_offset(hit_info.position, hit_info.frame.z, bsdf_sample.dir);
        r.d = bsdf_sample.dir;

        depth = depth + 1;
    };

    return result;
}

BTRC_PT_END
