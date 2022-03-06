#include <array>

#include <btrc/core/film.h>
#include <btrc/utils/intersection.h>
#include <btrc/utils/optix/device_funcs.h>

#include "./shade.h"

BTRC_WFPT_BEGIN

using namespace cuj;
using namespace shade_pipeline_detail;

namespace
{

    constexpr float EPS = 1.5e-4f;

    constexpr char SHADE_KERNEL_NAME[] = "shade_kernel";

    SurfacePoint get_intersection(
        f32                inct_t,
        ref<CVec3f>        ray_o,
        ref<CVec3f>        ray_d,
        ref<CInstanceInfo> instance,
        ref<CGeometryInfo> geometry,
        u32                prim_id,
        ref<CVec2f>        uv)
    {
        ref local_to_world = instance.transform;

        // position

        var position = ray_o + inct_t * ray_d;
        
        // geometry frame

        var gx_ua  = load_aligned(geometry.geometry_ex_tex_coord_u_a  + prim_id);
        var gy_uba = load_aligned(geometry.geometry_ey_tex_coord_u_ba + prim_id);
        var gz_uca = load_aligned(geometry.geometry_ez_tex_coord_u_ca + prim_id);

        var sn_v_a  = load_aligned(geometry.shading_normal_tex_coord_v_a  + prim_id);
        var sn_v_ba = load_aligned(geometry.shading_normal_tex_coord_v_ba + prim_id);
        var sn_v_ca = load_aligned(geometry.shading_normal_tex_coord_v_ca + prim_id);

        CFrame geometry_frame = CFrame(gx_ua.xyz(), gy_uba.xyz(), gz_uca.xyz());

        geometry_frame.x = local_to_world.apply_to_vector(geometry_frame.x);
        geometry_frame.y = local_to_world.apply_to_vector(geometry_frame.y);
        geometry_frame.z = local_to_world.apply_to_vector(geometry_frame.z);
        geometry_frame.x = normalize(geometry_frame.x);
        geometry_frame.y = normalize(geometry_frame.y);
        geometry_frame.z = normalize(geometry_frame.z);

        // interpolated normal

        var interp_normal =
            sn_v_a.xyz() + sn_v_ba.xyz() * uv.x + sn_v_ca.xyz() * uv.y;
        interp_normal = normalize(
            local_to_world.apply_to_vector(interp_normal));

        // tex coord

        var tex_coord_u = gx_ua.w + gy_uba.w * uv.x + gz_uca.w * uv.y;
        var tex_coord_v = sn_v_a.w + sn_v_ba.w * uv.x + sn_v_ca.w * uv.y;
        var tex_coord = CVec2f(tex_coord_u, tex_coord_v);

        // intersection

        SurfacePoint material_inct;
        material_inct.position  = position;
        material_inct.frame     = geometry_frame;
        material_inct.interp_z  = interp_normal;
        material_inct.uv        = uv;
        material_inct.tex_coord = tex_coord;

        return material_inct;
    }

} // namespace anonymous

void ShadePipeline::record_device_code(CompileContext &cc, Film &film, const Scene &scene, const ShadeParams &shade_params)
{
    using namespace cuj;

    auto light_sampler = scene.get_light_sampler();
    
    kernel(
        SHADE_KERNEL_NAME, [&](
            i32                total_state_count,
            ptr<CInstanceInfo> instances,
            ptr<CGeometryInfo> geometries,
            ptr<i32>           active_state_counter,
            ptr<i32>           inactive_state_counter,
            ptr<i32>           shadow_ray_counter,
            CSOAParams         soa_params)
    {
        var thread_index = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
        $if(thread_index >= total_state_count)
        {
            $return();
        };

        // basic path state

        var inct_inst_launch_index = load_aligned(soa_params.inct_inst_launch_index + thread_index);
        var soa_index = inct_inst_launch_index.y;
        
        var beta = load_aligned(soa_params.beta + soa_index);
        var path_radiance = load_aligned(soa_params.path_radiance + soa_index);

        var pixel_coord = load_aligned(soa_params.pixel_coord + soa_index);
        i32 depth = soa_params.depth[soa_index];

        var rng = soa_params.rng[soa_index];

        var inst_id = inct_inst_launch_index.x;
        $if(inst_id == INST_ID_MISS)
        {
            handle_miss(cc, light_sampler, soa_params, soa_index, path_radiance);
            $if(depth == 0)
            {
                film.splat_atomic(pixel_coord, Film::OUTPUT_WEIGHT, f32(1));
            };
            film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, path_radiance.to_rgb());

            var output_index = total_state_count - 1 - cstd::atomic_add(inactive_state_counter, 1);
            soa_params.output_rng[output_index] = rng;
            $return();
        };

        boolean scattered = false;
        $if((inst_id & INST_ID_MEDIUM_MASK) != 0)
        {
            scattered = true;
            inst_id = inst_id & ~INST_ID_MEDIUM_MASK;
        };

        var ray_o = load_aligned(cuj::bitcast<ptr<CVec3f>>(soa_params.ray_o_t0 + soa_index));
        var ray_d = load_aligned(cuj::bitcast<ptr<CVec3f>>(soa_params.ray_d_t1 + soa_index));
        var ray_time = bitcast<f32>(soa_params.ray_time_mask[soa_index].x);

        var inct_t_prim_uv = load_aligned(soa_params.inct_t_prim_uv + soa_index);
        var inct_t = bitcast<f32>(inct_t_prim_uv.x);
        var prim_id = inct_t_prim_uv.y;
        var uv = CVec2f(bitcast<f32>(inct_t_prim_uv.z), bitcast<f32>(inct_t_prim_uv.w));

        ref instance = instances[inst_id];
        ref geometry = geometries[instance.geometry_id];
        var inct = get_intersection(inct_t, ray_o, ray_d, instance, geometry, prim_id, uv);

        var medium_id = soa_params.ray_medium_id[soa_index];

        // handle intersected light

        auto handle_intersected_light = [&](int light_id)
        {
            auto light = light_sampler->get_light(light_id);
            const AreaLight *area = light->as_area();
            if(!area)
                return;

            var le = area->eval_le(cc, inct.position, inct.frame.z, inct.uv, inct.tex_coord, -ray_d);
            var beta_le = load_aligned(soa_params.beta_le + soa_index);
            var bsdf_pdf = soa_params.bsdf_pdf[soa_index];

            $if(scattered & medium_id != MEDIUM_ID_VOID)
            {
                var tr = CSpectrum::one();
                $switch(medium_id)
                {
                    for(int i = 0; i < scene.get_medium_count(); ++i)
                    {
                        $case(i)
                        {
                            tr = scene.get_medium(i)->tr(cc, ray_o, inct.position, rng);
                        };
                    }
                    $default
                    {
                        cstd::unreachable();
                    };
                };
                beta_le = beta_le * tr;
            };

            $if(bsdf_pdf < 0)
            {
                path_radiance = path_radiance + beta_le * le / -bsdf_pdf;
            }
            $else
            {
                var select_light_pdf = light_sampler->pdf(ray_o, ray_time, light_id);
                var light_dir_pdf = area->pdf_li(cc, ray_o, inct.position, inct.frame.z);
                var light_pdf = select_light_pdf * light_dir_pdf;
                path_radiance = path_radiance + beta_le * le / (bsdf_pdf + light_pdf);
            };
        };

        var intersected_light_id = instance.light_id;
        $if(intersected_light_id >= 0)
        {
            $switch(intersected_light_id)
            {
                for(int i = 0; i < light_sampler->get_light_count(); ++i)
                {
                    $case(i)
                    {
                        handle_intersected_light(i);
                    };
                }
                $default{ cstd::unreachable(); };
            };
        };

        $if(scattered)
        {
            $return();
        };

        // rr

        var rr_exit = false;
        $if(depth >= shade_params.min_depth)
        {
            $if(depth >= shade_params.max_depth)
            {
                rr_exit = true;
            }
            $else
            {
                var sam = rng.uniform_float();
                var lum = beta.get_lum();
                $if(lum < shade_params.rr_threshold)
                {
                    $if(sam > shade_params.rr_cont_prob)
                    {
                        rr_exit = true;
                    }
                    $else
                    {
                        beta = beta / shade_params.rr_cont_prob;
                    };
                };
            };
        };

        $if(rr_exit)
        {
            $if(depth == 0)
            {
                film.splat_atomic(pixel_coord, Film::OUTPUT_WEIGHT, f32(1));
            };
            film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, path_radiance.to_rgb());

            var output_index = total_state_count - 1 - cstd::atomic_add(inactive_state_counter, 1);
            soa_params.output_rng[output_index] = rng;
            $return();
        };

        // sample light

        auto select_light = light_sampler->sample(inct.position, ray_time, rng.uniform_float());
        var light_id = select_light.light_idx;
        
        CVec3f shadow_o, shadow_d;
        var shadow_t1 = 0.0f, light_dir_pdf = 0.0f;
        var li = CSpectrum::zero();
        $if(light_id >= 0)
        {
            $switch(light_id)
            {
                for(int i = 0; i < light_sampler->get_light_count(); ++i)
                {
                    $case(i)
                    {
                        auto light = light_sampler->get_light(i);
                        var sam = CVec3f(rng);
                        if(auto area = light->as_area())
                        {
                            auto sample = area->sample_li(cc, inct.position, sam);
                            var diff = sample.position - inct.position;
                            var shadow_dst = intersection_offset(sample.position, sample.normal, -diff);
                            shadow_o = intersection_offset(inct.position, inct.frame.z, diff);
                            shadow_d = shadow_dst - shadow_o;
                            shadow_t1 = 1;
                            light_dir_pdf = sample.pdf;
                            li = sample.radiance;
                        }
                        else
                        {
                            assert(!light->is_area());
                            auto sample = light->as_envir()->sample_li(cc, sam);
                            shadow_o = intersection_offset(inct.position, inct.frame.z, sample.direction_to_light);
                            shadow_d = sample.direction_to_light;
                            shadow_t1 = btrc_max_float;
                            light_dir_pdf = sample.pdf;
                            li = sample.radiance;
                        }
                    };
                }
            };
        };

        // eval bsdf

        Shader::SampleResult bsdf_sample;

        var is_bsdf_delta = false;
        CVec3f gbuffer_albedo;
        CVec3f gbuffer_normal;

        CSpectrum shadow_bsdf_val;
        f32 shadow_bsdf_pdf = 0;
        var emit_shadow_ray = false;

        auto handle_material = [&](const Material *mat)
        {
            auto shader = mat->create_shader(cc, inct);

            // gbuffer

            $if(depth == 0)
            {
                gbuffer_albedo = shader->albedo(cc).to_rgb();
                gbuffer_normal = shader->normal(cc);
            };

            // shadow ray

            $if(!shader->is_delta(cc))
            {
                $if(light_id >= 0 & shadow_t1 > EPS & !li.is_zero())
                {
                    shadow_bsdf_val = shader->eval(cc, shadow_d, -ray_d, TransportMode::Radiance);
                    emit_shadow_ray = !shadow_bsdf_val.is_zero();
                    $if(emit_shadow_ray)
                    {
                        shadow_bsdf_pdf = shader->pdf(cc, shadow_d, -ray_d, TransportMode::Radiance);
                    };
                };
            };

            // sample bsdf

            bsdf_sample = shader->sample(cc, -ray_d, CVec3f(rng), TransportMode::Radiance);
            is_bsdf_delta = shader->is_delta(cc);
        };

        var mat_id = instance.material_id;
        $switch(mat_id)
        {
            for(int i = 0; i < scene.get_material_count(); ++i)
            {
                $case(i)
                {
                    handle_material(scene.get_material(i));
                };
            }
            $default{ cstd::unreachable(); };
        };

        $if(emit_shadow_ray)
        {
            var shadow_soa_index = cstd::atomic_add(shadow_ray_counter, 1);

            save_aligned(
                pixel_coord,
                soa_params.output_shadow_pixel_coord + shadow_soa_index);
            save_aligned(
                CVec4f(shadow_o, 0),
                soa_params.output_shadow_ray_o_t0 + shadow_soa_index);
            save_aligned(
                CVec4f(shadow_d, shadow_t1),
                soa_params.output_shadow_ray_d_t1 + shadow_soa_index);
            save_aligned(
                CVec2u(bitcast<u32>(ray_time), optix::RAY_MASK_ALL),
                soa_params.output_shadow_ray_time_mask + shadow_soa_index);

            var cos = cstd::abs(dot(normalize(shadow_d), inct.frame.z));
            var light_pdf = select_light.pdf * light_dir_pdf;
            var beta_li = li * beta * shadow_bsdf_val * cos / (shadow_bsdf_pdf + light_pdf);

            save_aligned(beta_li, soa_params.output_shadow_beta_li + shadow_soa_index);

            $if(dot(inct.frame.z, shadow_d) > 0)
            {
                soa_params.output_shadow_ray_medium_id[shadow_soa_index] = instance.outer_medium_id;
            }
            $else
            {
                soa_params.output_shadow_ray_medium_id[shadow_soa_index] = instance.inner_medium_id;
            };
        };

        $if(depth == 0)
        {
            std::vector<std::pair<std::string_view, Film::CValue>> values = {
                { Film::OUTPUT_WEIGHT, f32(1) },
                { Film::OUTPUT_ALBEDO, gbuffer_albedo },
                { Film::OUTPUT_NORMAL, gbuffer_normal }
            };
            film.splat_atomic(pixel_coord, values);
        };

        // emit next ray

        $if(!bsdf_sample.bsdf.is_zero())
        {
            bsdf_sample.dir = normalize(bsdf_sample.dir);
            var cos = cstd::abs(dot(bsdf_sample.dir, inct.frame.z));
            
            var beta_le = beta * cos * bsdf_sample.bsdf;
            beta = beta_le / bsdf_sample.pdf;

            var next_ray_o    = intersection_offset(inct.position, inct.frame.z, bsdf_sample.dir);
            var next_ray_t0   = 0.0f;
            var next_ray_d    = bsdf_sample.dir;
            var next_ray_t1   = btrc_max_float;
            var next_ray_time = ray_time;
            var next_ray_mask = optix::RAY_MASK_ALL;

            var output_index = cstd::atomic_add(active_state_counter, 1);

            soa_params.output_rng[output_index] = rng;
            
            save_aligned(pixel_coord, soa_params.output_pixel_coord + output_index);
            soa_params.output_depth[output_index] = depth + 1;
            save_aligned(path_radiance, soa_params.output_path_radiance + output_index);
            save_aligned(beta, soa_params.output_beta + output_index);

            save_aligned(
                CVec4f(next_ray_o, next_ray_t0),
                soa_params.output_new_ray_o_t0 + output_index);
            save_aligned(
                CVec4f(next_ray_d, next_ray_t1),
                soa_params.output_new_ray_d_t1 + output_index);
            save_aligned(
                CVec2u(bitcast<u32>(next_ray_time), u32(next_ray_mask)),
                soa_params.output_new_ray_time_mask + output_index);

            save_aligned(beta_le, soa_params.output_beta_le + output_index);

            var stored_pdf = cstd::select(is_bsdf_delta, -bsdf_sample.pdf, f32(bsdf_sample.pdf));
            soa_params.output_bsdf_pdf[output_index] = stored_pdf;

            soa_params.output_new_ray_medium_id[output_index] = medium_id;

            $if(dot(inct.frame.z, next_ray_d) > 0)
            {
                soa_params.output_shadow_ray_medium_id[output_index] = instance.outer_medium_id;
            }
            $else
            {
                soa_params.output_shadow_ray_medium_id[output_index] = instance.inner_medium_id;
            };
        }
        $else
        {
            film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, path_radiance.to_rgb());
            var output_index = total_state_count - 1 - cstd::atomic_add(inactive_state_counter, 1);
            soa_params.output_rng[output_index] = rng;
        };
    });
}

void ShadePipeline::initialize(RC<cuda::Module> cuda_module, RC<cuda::Buffer<StateCounters>> counters, const Scene &scene)
{
    kernel_ = std::move(cuda_module);
    geo_info_ = scene.get_device_geometry_info();
    inst_info_ = scene.get_device_instance_info();
    counters_ = std::move(counters);
}

ShadePipeline::ShadePipeline(ShadePipeline &&other) noexcept
    : ShadePipeline()
{
    swap(other);
}

ShadePipeline &ShadePipeline::operator=(ShadePipeline &&other) noexcept
{
    swap(other);
    return *this;
}

void ShadePipeline::swap(ShadePipeline &other) noexcept
{
    std::swap(kernel_,    other.kernel_);
    std::swap(geo_info_,  other.geo_info_);
    std::swap(inst_info_, other.inst_info_);
    std::swap(counters_,  other.counters_);
}

void ShadePipeline::shade(int total_state_count, const SOAParams &soa)
{
    assert(kernel_.is_linked());

    StateCounters *device_counters = counters_->get();
    int32_t *active_state_counter   = reinterpret_cast<int32_t *>(device_counters);
    int32_t *inactive_state_counter = active_state_counter + 1;
    int32_t *shadow_ray_counter     = active_state_counter + 2;

    constexpr int BLOCK_DIM = 256;
    const int thread_count = total_state_count;
    const int block_count = up_align(thread_count, BLOCK_DIM) / BLOCK_DIM;

    kernel_->launch(
        SHADE_KERNEL_NAME,
        { block_count, 1, 1 },
        { BLOCK_DIM, 1, 1 },
        total_state_count,
        inst_info_,
        geo_info_,
        active_state_counter,
        inactive_state_counter,
        shadow_ray_counter,
        soa);
}

void ShadePipeline::handle_miss(
    CompileContext     &cc,
    const LightSampler *light_sampler,
    ref<CSOAParams>     soa_params,
    u32                 soa_index,
    ref<CSpectrum>      path_rad)
{
    var time = bitcast<f32>(soa_params.ray_time_mask[soa_index].x);
    auto envir_light = light_sampler->get_envir_light();
    if(envir_light)
    {
        var ray_dir = load_aligned(cuj::bitcast<ptr<CVec3f>>(soa_params.ray_d_t1 + soa_index));
        var le = envir_light->eval_le(cc, ray_dir);
        var beta_le = load_aligned(soa_params.beta_le + soa_index);

        var bsdf_pdf = soa_params.bsdf_pdf[soa_index];
        $if(bsdf_pdf < 0) // delta
        {
            var rad = beta_le * le / -bsdf_pdf;
            path_rad = path_rad + rad;
        }
        $else
        {
            var o = load_aligned(cuj::bitcast<ptr<CVec3f>>(soa_params.ray_o_t0 + soa_index));
            var d = load_aligned(cuj::bitcast<ptr<CVec3f>>(soa_params.ray_d_t1 + soa_index));

            var select_light_pdf = light_sampler->pdf(
                o, time, light_sampler->get_envir_light_index());
            var envir_light_pdf = envir_light->pdf_li(cc, d);
            var light_pdf = select_light_pdf * envir_light_pdf;

            var rad = beta_le * le / (bsdf_pdf + light_pdf);
            path_rad = path_rad + rad;
        };
    }
}

BTRC_WFPT_END
