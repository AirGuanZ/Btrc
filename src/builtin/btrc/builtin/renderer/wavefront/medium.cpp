#include <btrc/builtin/renderer/wavefront/medium.h>
#include <btrc/utils/intersection.h>
#include <btrc/utils/optix/device_funcs.h>

BTRC_WFPT_BEGIN

namespace
{

    const char *KERNEL = "sample_medium_scattering_kernel";

    CMediumID resolve_mediums(CMediumID medium_a, CMediumID medium_b)
    {
        return cstd::min(medium_a, medium_b);
    }

} // namespace anonymous

void MediumPipeline::record_device_code(CompileContext &cc, Film &film, const Scene &scene, const ShadeParams &shade_params)
{
    using namespace cuj;

    kernel(KERNEL, [&cc, &film, &shade_params, &scene](
        i32                total_state_count,
        ptr<CInstanceInfo> instances,
        ptr<CGeometryInfo> geometries,
        ptr<i32>           active_state_counter,
        ptr<i32>           inactive_state_counter,
        ptr<i32>           shadow_ray_counter,
        CSOAParams         soa)
    {
        var thread_index = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
        $if(thread_index >= total_state_count)
        {
            $return();
        };

        var inct_inst_launch_index = load_aligned(soa.inct_inst_launch_index + thread_index);
        var inst_id = inct_inst_launch_index.x;
        var soa_index = inct_inst_launch_index.y;

        $if(inst_id == INST_ID_MISS)
        {
            $return();
        };

        var ray_d_t1 = load_aligned(soa.ray_d_t1 + soa_index);
        var ray_d = ray_d_t1.xyz();

        var inct_t_prim_uv = load_aligned(soa.inct_t_prim_uv + soa_index);
        var prim_id = inct_t_prim_uv.y;

        ref instance = instances[inst_id];
        ref geometry = geometries[instance.geometry_id];
        var local_normal = load_aligned(geometry.geometry_ez_tex_coord_u_ca + prim_id).xyz();
        var inct_nor = instance.transform.apply_to_vector(local_normal);

        var inct_medium_id = cstd::select(dot(ray_d, inct_nor) < 0, instance.outer_medium_id, instance.inner_medium_id);
        var ray_medium_id = soa.ray_medium_id[soa_index];
        var medium_id = resolve_mediums(inct_medium_id, ray_medium_id);
        soa.ray_medium_id[soa_index] = medium_id;

        $if(medium_id == MEDIUM_ID_VOID)
        {
            $return();
        };

        var ray_o_t0 = load_aligned(soa.ray_o_t0 + soa_index);
        var ray_o = ray_o_t0.xyz();

        var inct_t = bitcast<f32>(inct_t_prim_uv.x);
        var inct_pos = intersection_offset(ray_o + inct_t * ray_d, inct_nor, -ray_d);

        var rng = soa.rng[soa_index];

        CVec3f shadow_o, shadow_d; f32 shadow_t1, shadow_light_pdf;
        CSpectrum shadow_li;

        var tr = CSpectrum::one();
        auto handle_medium = [&](const Medium *medium)
        {
            auto sample_medium = medium->sample(cc, ray_o, inct_pos, rng);

            $if(sample_medium.scattered)
            {
                var beta = soa.beta[soa_index];
                beta = beta * sample_medium.throughput;

                var path_radiance = load_aligned(soa.path_radiance + soa_index);
                var pixel_coord   = load_aligned(soa.pixel_coord + soa_index);
                var depth         = soa.depth[soa_index];

                // mark this path as scattered
                soa.inct_inst_launch_index[thread_index].x = inst_id | INST_ID_MEDIUM_MASK;

                $if(depth == 0)
                {
                    film.splat_atomic(pixel_coord, Film::OUTPUT_WEIGHT, f32(1));
                };

                // terminate

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
                    film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, path_radiance.to_rgb());

                    var output_index = total_state_count - 1 - cstd::atomic_add(inactive_state_counter, 1);
                    soa.output_rng[output_index] = rng;
                    $return();
                };

                // generate shadow ray

                Function sample_light_for_medium_direct_illum = [&cc, &scene](
                    ref<CVec3f>    scatter_pos,
                    ref<CRNG>      rng,
                    f32            time,
                    ref<CVec3f>    shadow_d,
                    ref<f32>       shadow_t1,
                    ref<f32>       shadow_light_pdf,
                    ref<CSpectrum> shadow_li)
                {
                    auto select_light = scene.get_light_sampler()->sample(scatter_pos, time, rng.uniform_float());
                    $if(select_light.light_idx >= 0)
                    {
                        $switch(select_light.light_idx)
                        {
                            for(int i = 0; i < scene.get_light_sampler()->get_light_count(); ++i)
                            {
                                $case(i)
                                {
                                    auto light = scene.get_light_sampler()->get_light(i);
                                    var sam = CVec3f(rng);
                                    if(auto area = light->as_area())
                                    {
                                        auto sample = area->sample_li(cc, scatter_pos, sam);
                                        var diff = sample.position - scatter_pos;
                                        var shadow_dst = intersection_offset(sample.position, sample.normal, -diff);
                                        shadow_d = shadow_dst - scatter_pos;
                                        shadow_t1 = 1;
                                        shadow_li = sample.radiance;
                                        shadow_light_pdf = select_light.pdf * sample.pdf;
                                    }
                                    else
                                    {
                                        assert(!light->is_area());
                                        auto sample = light->as_envir()->sample_li(cc, sam);
                                        shadow_d = sample.direction_to_light;
                                        shadow_t1 = btrc_max_float;
                                        shadow_li = sample.radiance;
                                        shadow_light_pdf = select_light.pdf * sample.pdf;
                                    }
                                };
                            }
                        };
                    };
                };

                shadow_o = sample_medium.position;

                var ray_time = bitcast<f32>(soa.ray_time_mask[soa_index].x);
                sample_light_for_medium_direct_illum(
                    sample_medium.position, rng, ray_time,
                    shadow_d, shadow_t1, shadow_light_pdf, shadow_li);
                
                $if(shadow_t1 > 1e-4f & !shadow_li.is_zero())
                {
                    var shadow_phase_val = sample_medium.shader->eval(cc, shadow_d, -ray_d);
                    var shadow_phase_pdf = sample_medium.shader->pdf(cc, shadow_d, -ray_d);

                    var shadow_soa_index = cstd::atomic_add(shadow_ray_counter, 1);

                    save_aligned(
                        pixel_coord,
                        soa.output_shadow_pixel_coord + shadow_soa_index);
                    save_aligned(
                        CVec4f(shadow_o, 0),
                        soa.output_shadow_ray_o_t0 + shadow_soa_index);
                    save_aligned(
                        CVec4f(shadow_d, shadow_t1),
                        soa.output_shadow_ray_d_t1 + shadow_soa_index);
                    save_aligned(
                        CVec2u(bitcast<u32>(ray_time), optix::RAY_MASK_ALL),
                        soa.output_shadow_ray_time_mask + shadow_soa_index);

                    $if(shadow_t1 > 1)
                    {
                        soa.output_shadow_medium_id[shadow_soa_index] = MEDIUM_ID_VOID;
                    }
                    $else
                    {
                        soa.output_shadow_medium_id[shadow_soa_index] = medium_id;
                    };

                    var beta_li = shadow_li * beta * shadow_phase_val / (shadow_phase_pdf + shadow_light_pdf);
                    save_aligned(
                        beta_li,
                        soa.output_shadow_beta_li + shadow_soa_index);
                };

                // generate next ray

                PhaseShader::SampleResult phase_sample = sample_medium.shader->sample(cc, -ray_d, CVec3f(rng));
                $if(!phase_sample.phase.is_zero())
                {
                    phase_sample.dir = normalize(phase_sample.dir);

                    var beta_le = beta * phase_sample.phase;
                    beta = beta_le / phase_sample.pdf;

                    var next_ray_o      = sample_medium.position;
                    var next_ray_t0     = 0.0f;
                    var next_ray_d      = phase_sample.dir;
                    var next_ray_t1     = btrc_max_float;
                    var next_ray_time   = ray_time;
                    var next_ray_mask   = optix::RAY_MASK_ALL;
                    var next_ray_medium = medium_id;

                    var output_soa_index = cstd::atomic_add(active_state_counter, 1);

                    soa.output_rng[output_soa_index] = rng;

                    save_aligned(
                        path_radiance,
                        soa.output_path_radiance + output_soa_index);
                    save_aligned(
                        pixel_coord,
                        soa.output_pixel_coord + output_soa_index);
                    save_aligned(
                        beta,
                        soa.output_beta + output_soa_index);

                    soa.output_depth[output_soa_index] = depth + 1;

                    save_aligned(
                        CVec4f(next_ray_o, next_ray_t0),
                        soa.output_new_ray_o_t0 + output_soa_index);
                    save_aligned(
                        CVec4f(next_ray_d, next_ray_t1),
                        soa.output_new_ray_d_t1 + output_soa_index);
                    save_aligned(
                        CVec2u(bitcast<u32>(next_ray_time), u32(next_ray_mask)),
                        soa.output_new_ray_time_mask + output_soa_index);

                    soa.output_new_ray_medium_id[output_soa_index] = next_ray_medium;

                    save_aligned(
                        beta_le,
                        soa.output_beta_le + output_soa_index);

                    soa.output_bsdf_pdf[output_soa_index] = phase_sample.pdf;
                }
                $else
                {
                    film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, path_radiance.to_rgb());
                    var output_index = total_state_count - 1 - cstd::atomic_add(inactive_state_counter, 1);
                    soa.output_rng[output_index] = rng;
                };
            }
            $else
            {
                tr = sample_medium.throughput;
            };

            $return();
        };

        $switch(medium_id)
        {
            for(int i = 0; i < scene.get_medium_count(); ++i)
            {
                $case(MediumID(i))
                {
                    handle_medium(scene.get_medium(i));
                };
            }
            $default
            {
                cstd::unreachable();
            };
        };

        var beta = load_aligned(soa.beta + soa_index);
        var beta_le = load_aligned(soa.beta_le + soa_index);

        beta = beta * tr;
        beta_le = beta_le * tr;

        save_aligned(beta, soa.beta + soa_index);
        save_aligned(beta_le, soa.beta_le + soa_index);

        soa.rng[soa_index] = rng;
    });
}

void MediumPipeline::initialize(RC<cuda::Module> cuda_module, RC<cuda::Buffer<StateCounters>> counters, const Scene &scene)
{
    cuda_module_ = std::move(cuda_module);
    state_counters_ = std::move(counters);
    geo_info_ = scene.get_device_geometry_info();
    inst_info_ = scene.get_device_instance_info();
}

MediumPipeline::MediumPipeline(MediumPipeline &&other) noexcept
    : MediumPipeline()
{
    swap(other);
}

MediumPipeline &MediumPipeline::operator=(MediumPipeline &&other) noexcept
{
    swap(other);
    return *this;
}

void MediumPipeline::swap(MediumPipeline &other) noexcept
{
    std::swap(cuda_module_, other.cuda_module_);
    std::swap(state_counters_, other.state_counters_);
    std::swap(geo_info_, other.geo_info_);
    std::swap(inst_info_, other.inst_info_);
}

void MediumPipeline::sample_scattering(int total_state_count, const SOAParams &soa)
{
    assert(cuda_module_->is_linked());

    StateCounters *device_counters = state_counters_->get();
    int32_t *active_state_counter = reinterpret_cast<int32_t *>(device_counters);
    int32_t *inactive_state_counter = active_state_counter + 1;
    int32_t *shadow_ray_counter = active_state_counter + 2;

    constexpr int BLOCK_DIM = 256;
    const int thread_count = total_state_count;
    const int block_count = up_align(thread_count, BLOCK_DIM) / BLOCK_DIM;

    cuda_module_->launch(
        KERNEL,
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

BTRC_WFPT_END
