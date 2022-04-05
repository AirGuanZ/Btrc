#include <array>

#include <btrc/builtin/renderer/wavefront/helper.h>
#include <btrc/builtin/renderer/wavefront/shade.h>
#include <btrc/core/film.h>
#include <btrc/utils/intersection.h>
#include <btrc/utils/optix/device_funcs.h>

BTRC_WFPT_BEGIN

using namespace cuj;
using namespace shade_pipeline_detail;

namespace
{
    
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
        material_inct.position  = position;
        material_inct.frame     = geometry_frame;
        material_inct.interp_z  = interp_normal;
        material_inct.uv        = uv;
        material_inct.tex_coord = tex_coord;

        return material_inct;
    }

} // namespace anonymous

void ShadePipeline::record_device_code(
    CompileContext    &cc,
    Film              &film,
    const Scene       &scene,
    const ShadeParams &shade_params,
    float              world_diagonal)
{
    using namespace cuj;

    kernel(
        SHADE_KERNEL_NAME, [&](
            i32        total_state_count,
            ptr<i32>   active_state_counter,
            ptr<i32>   shadow_ray_counter,
            CSOAParams soa_params)
    {
        const WFPTScene wfpt_scene = { cc, scene, world_diagonal };

        var soa_index = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
        $if(soa_index >= total_state_count)
        {
            $return();
        };

        // basic path state

        var path_flag = soa_params.path_flag[soa_index];
        
        var beta = load_aligned(soa_params.beta + soa_index);

        var pixel_coord = load_aligned(soa_params.pixel_coord + soa_index);
        i32 depth = soa_params.depth[soa_index];

        var scattered = is_path_scattered(path_flag);

        var path_radiance = load_aligned(soa_params.path_radiance + soa_index);
        $if(scattered)
        {
            path_radiance = CSpectrum::zero();
        };

        $if(!is_path_intersected(path_flag))
        {
            var env_path_rad = CSpectrum::zero();

            if(scene.get_light_sampler()->get_envir_light())
            {
                var ray_ori = load_aligned(cuj::bitcast<ptr<CVec3f>>(soa_params.ray_o_medium_id + soa_index));
                var ray_dir = load_aligned(cuj::bitcast<ptr<CVec3f>>(soa_params.ray_d_t1 + soa_index));
                var beta_le = load_aligned(soa_params.beta_le_bsdf_pdf + soa_index);
                var bsdf_pdf = beta_le.additional_data;
                env_path_rad = wfpt::handle_miss(wfpt_scene, ray_ori, ray_dir, beta_le, bsdf_pdf);
            }

            $if(scattered)
            {
                film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, env_path_rad.to_rgb());
            }
            $else
            {
                film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, (path_radiance + env_path_rad).to_rgb());
            };
            $return();
        };

        var inst_id = extract_instance_id(path_flag);

        var ray_o_medium_id = load_aligned(soa_params.ray_o_medium_id + soa_index);
        var ray_o = ray_o_medium_id.xyz();
        var medium_id = bitcast<CMediumID>(ray_o_medium_id.w);
        var ray_d = load_aligned(cuj::bitcast<ptr<CVec3f>>(soa_params.ray_d_t1 + soa_index));

        var inct_t_prim_uv = load_aligned(soa_params.inct_t_prim_uv + soa_index);
        var inct_t = bitcast<f32>(inct_t_prim_uv.x);
        var prim_id = inct_t_prim_uv.y;
        var uv = CVec2f(bitcast<f32>(inct_t_prim_uv.z), bitcast<f32>(inct_t_prim_uv.w));

        var instances = const_data(std::span{
            scene.get_host_instance_info(), static_cast<size_t>(scene.get_instance_count()) });
        var geometries = const_data(std::span{
            scene.get_host_geometry_info(), static_cast<size_t>(scene.get_geometry_count()) });

        ref instance = instances[inst_id];
        ref geometry = geometries[instance.geometry_id];
        var inct = get_intersection(inct_t, ray_o, ray_d, instance, geometry, prim_id, uv);
        
        var beta_le = load_aligned(soa_params.beta_le_bsdf_pdf + soa_index);
        var bsdf_pdf = beta_le.additional_data;
        var le_rad = handle_intersected_light(
            wfpt_scene, ray_o, ray_d, inct, beta_le, bsdf_pdf, instance.light_id);

        $if(scattered)
        {
            film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, le_rad.to_rgb());
            $return();
        };

        path_radiance = path_radiance + le_rad;

        GlobalSampler sampler({ film.width(), film.height() }, soa_params.sampler_state[soa_index]);

        // rr

        var rr_exit = simple_russian_roulette(
            beta, depth, sampler, SimpleRussianRouletteParams{
                .min_depth      = shade_params.min_depth,
                .max_depth      = shade_params.max_depth,
                .beta_threshold = shade_params.rr_threshold,
                .cont_prob      = shade_params.rr_cont_prob
            });
        $if(rr_exit)
        {
            film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, path_radiance.to_rgb());
            $return();
        };

        // sample light

        CVec3f shadow_o, shadow_d;
        f32 shadow_t1, shadow_light_pdf;
        CSpectrum shadow_li;

        var shadow_bsdf_val = CSpectrum::zero();
        var shadow_bsdf_pdf = 0.0f;

        var emit_shadow_ray =  sample_light_li(
            wfpt_scene, inct.position, inct.frame.z, sampler,
            shadow_o, shadow_d, shadow_t1, shadow_light_pdf, shadow_li);

        // eval bsdf

        Shader::SampleResult bsdf_sample;

        CVec3f gbuffer_albedo;
        CVec3f gbuffer_normal;

        auto handle_material = [&](const Material *mat)
        {
            auto shader = mat->create_shader(cc, inct);

            // gbuffer

            $if(depth == 0)
            {
                gbuffer_albedo = shader->albedo(cc).to_rgb();
                gbuffer_normal = shader->normal(cc);
            };

            // sample bsdf

            bsdf_sample = shader->sample(cc, -ray_d, sampler.get3d(), TransportMode::Radiance);

            // shadow ray

            $if(emit_shadow_ray)
            {
                shadow_bsdf_val = shader->eval(cc, shadow_d, -ray_d, TransportMode::Radiance);
                emit_shadow_ray = !shadow_bsdf_val.is_zero();
                $if(emit_shadow_ray)
                {
                    shadow_bsdf_pdf = shader->pdf(cc, shadow_d, -ray_d, TransportMode::Radiance);
                };
            };
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

            CMediumID shadow_medium_id;
            var shadow_d_out = dot(inct.frame.z, shadow_d) > 0;
            var last_ray_out = dot(inct.frame.z, ray_d) < 0;
            $if(shadow_d_out == last_ray_out)
            {
                shadow_medium_id = medium_id;
            }
            $elif(shadow_d_out)
            {
                shadow_medium_id = instance.outer_medium_id;
            }
            $else
            {
                shadow_medium_id = instance.inner_medium_id;
            };

            save_aligned(pixel_coord, soa_params.output_shadow_pixel_coord + shadow_soa_index);

            save_aligned(
                CVec4f(shadow_o, bitcast<f32>(shadow_medium_id)),
                soa_params.output_shadow_ray_o_medium_id + shadow_soa_index);
            save_aligned(
                CVec4f(shadow_d, shadow_t1),
                soa_params.output_shadow_ray_d_t1 + shadow_soa_index);

            var cos = cstd::abs(dot(normalize(shadow_d), inct.frame.z));
            var beta_li = shadow_li * beta * shadow_bsdf_val * cos / (shadow_bsdf_pdf + shadow_light_pdf);

            save_aligned(beta_li, soa_params.output_shadow_beta_li + shadow_soa_index);
        };

        $if(depth == 0)
        {
            std::vector<std::pair<std::string_view, Film::CValue>> values = {
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
            var next_ray_d    = bsdf_sample.dir;
            var next_ray_t1   = btrc_max_float;

            var output_index = cstd::atomic_add(active_state_counter, 1);

            sampler.save(soa_params.output_sampler_state + output_index);
            
            save_aligned(pixel_coord, soa_params.output_pixel_coord + output_index);
            soa_params.output_depth[output_index] = depth + 1;
            save_aligned(path_radiance, soa_params.output_path_radiance + output_index);
            save_aligned(beta, soa_params.output_beta + output_index);

            var next_ray_out = dot(inct.frame.z, next_ray_d) > 0;
            var next_ray_medium_id = cstd::select(
                next_ray_out, instance.outer_medium_id, instance.inner_medium_id);

            save_aligned(
                CVec4f(next_ray_o, bitcast<f32>(next_ray_medium_id)),
                soa_params.output_new_ray_o_medium_id + output_index);
            save_aligned(
                CVec4f(next_ray_d, next_ray_t1),
                soa_params.output_new_ray_d_t1 + output_index);

            var stored_pdf = cstd::select(bsdf_sample.is_delta, -bsdf_sample.pdf, f32(bsdf_sample.pdf));
            beta_le.additional_data = stored_pdf;
            save_aligned(beta_le, soa_params.output_beta_le_bsdf_pdf + output_index);
        }
        $else
        {
            film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, path_radiance.to_rgb());
        };
    });
}

void ShadePipeline::initialize(RC<cuda::Module> cuda_module, RC<cuda::Buffer<StateCounters>> counters, const Scene &scene)
{
    kernel_ = std::move(cuda_module);
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
    std::swap(counters_,  other.counters_);
}

void ShadePipeline::shade(int total_state_count, const SOAParams &soa)
{
    assert(kernel_->is_linked());

    StateCounters *device_counters = counters_->get();
    int32_t *active_state_counter   = reinterpret_cast<int32_t *>(device_counters);
    int32_t *shadow_ray_counter     = active_state_counter + 2;

    constexpr int BLOCK_DIM = 256;
    const int thread_count = total_state_count;
    const int block_count = up_align(thread_count, BLOCK_DIM) / BLOCK_DIM;

    kernel_->launch(
        SHADE_KERNEL_NAME,
        { block_count, 1, 1 },
        { BLOCK_DIM, 1, 1 },
        total_state_count,
        active_state_counter,
        shadow_ray_counter,
        soa);
}

BTRC_WFPT_END
