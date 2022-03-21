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

void MediumPipeline::record_device_code(
    CompileContext      &cc,
    Film                &film,
    const VolumeManager &vols,
    const Scene         &scene,
    const ShadeParams   &shade_params,
    float                world_diagonal)
{
    using namespace cuj;

    kernel(KERNEL, [&cc, &film, &vols, &shade_params, &scene, world_diagonal, this](
        i32                total_state_count,
        ptr<CInstanceInfo> instances,
        ptr<CGeometryInfo> geometries,
        ptr<i32>           active_state_counter,
        ptr<i32>           inactive_state_counter,
        ptr<i32>           shadow_ray_counter,
        CSOAParams         soa)
    {
        var soa_index = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
        $if(soa_index >= total_state_count)
        {
            $return();
        };

        var path_flag = soa.path_flag[soa_index];

        var ray_o_medium_id = load_aligned(soa.ray_o_medium_id + soa_index);
        var ray_o = ray_o_medium_id.xyz();

        var ray_d_t1 = load_aligned(soa.ray_d_t1 + soa_index);
        var ray_d = ray_d_t1.xyz();

        var rng = soa.rng[soa_index];

        // resolve medium id

        CMediumID medium_id;
        CVec3f medium_end;
        $if(is_path_intersected(path_flag))
        {
            var inct_t_prim_uv = load_aligned(soa.inct_t_prim_uv + soa_index);
            var prim_id = inct_t_prim_uv.y;
            ref instance = instances[extract_instance_id(path_flag)];
            ref geometry = geometries[instance.geometry_id];
            var local_normal = load_aligned(geometry.geometry_ez_tex_coord_u_ca + prim_id).xyz();
            var inct_nor = instance.transform.apply_to_vector(local_normal);

            var inct_medium_id = cstd::select(dot(ray_d, inct_nor) < 0, instance.outer_medium_id, instance.inner_medium_id);
            var ray_medium_id = bitcast<CMediumID>(ray_o_medium_id.w);
            medium_id = resolve_mediums(inct_medium_id, ray_medium_id);

            var inct_t = bitcast<f32>(inct_t_prim_uv.x);
            medium_end = intersection_offset(ray_o + inct_t * ray_d, inct_nor, -ray_d);
        }
        $else
        {
            medium_id = MEDIUM_ID_VOID;
            medium_end = ray_o + normalize(ray_d) * world_diagonal;
        };
        soa.ray_o_medium_id[soa_index].w = bitcast<f32>(medium_id);

        boolean scattered = false;

        boolean emit_shadow = false;
        CVec3f shadow_o;
        CVec3f shadow_d;
        f32 shadow_t1;
        f32 shadow_light_pdf;
        CSpectrum shadow_beta;

        f32 scatter_ray_time;
        CSpectrum scatter_beta, scatter_path_radiance;
        i32 scatter_depth;

        PhaseShader::SampleResult phase_sample;
        CVec3f scatter_position;

        CSpectrum unscatter_tr;

        auto handle_medium = [&](const Medium::SampleResult &sample_medium)
        {
            $if(sample_medium.scattered)
            {
                scattered = true;
                scatter_position = sample_medium.position;

                scatter_beta = load_aligned(soa.beta + soa_index);
                scatter_beta = scatter_beta * sample_medium.throughput;

                scatter_path_radiance = load_aligned(soa.path_radiance + soa_index);
                scatter_depth = soa.depth[soa_index];

                soa.path_flag[soa_index] = path_flag | PATH_FLAG_HAS_SCATTERING;

                // terminate

                var rr_exit = false;
                $if(scatter_depth >= shade_params.min_depth)
                {
                    $if(scatter_depth >= shade_params.max_depth)
                    {
                        rr_exit = true;
                    }
                    $else
                    {
                        var sam = rng.uniform_float();
                        var lum = scatter_beta.get_lum();
                        $if(lum < shade_params.rr_threshold)
                        {
                            $if(sam > shade_params.rr_cont_prob)
                            {
                                rr_exit = true;
                            }
                            $else
                            {
                                scatter_beta = scatter_beta / shade_params.rr_cont_prob;
                            };
                        };
                    };
                };

                $if(rr_exit)
                {
                    var pixel_coord = load_aligned(soa.pixel_coord + soa_index);
                    film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, scatter_path_radiance.to_rgb());
                    var output_index = total_state_count - 1 - cstd::atomic_add(inactive_state_counter, 1);
                    soa.output_rng[output_index] = rng;
                    soa.next_state_index[soa_index] = output_index;
                    $return();
                };

                // generate shadow ray

                shadow_o = sample_medium.position;

                var ray_time = bitcast<f32>(soa.ray_time_mask[soa_index].x);
                CSpectrum shadow_li;
                sample_light(
                    cc, scene, sample_medium.position, rng, ray_time,
                    shadow_d, shadow_t1, shadow_light_pdf, shadow_li);
                
                $if(shadow_t1 > 1e-4f & !shadow_li.is_zero())
                {
                    var shadow_phase_val = sample_medium.shader->eval(cc, shadow_d, -ray_d);
                    var shadow_phase_pdf = sample_medium.shader->pdf(cc, shadow_d, -ray_d);
                    shadow_beta = shadow_li * scatter_beta * shadow_phase_val / (shadow_phase_pdf + shadow_light_pdf);
                    emit_shadow = true;
                };

                // generate next ray

                phase_sample = sample_medium.shader->sample(cc, -ray_d, CVec3f(rng));
            }
            $else
            {
                unscatter_tr = sample_medium.throughput;
            };
        };

        $switch(medium_id)
        {
            for(int i = 0; i < scene.get_medium_count(); ++i)
            {
                $case(MediumID(i))
                {
                    auto medium = scene.get_medium(i);
                    auto sample_medium = medium->sample(cc, ray_o, medium_end, ray_o, medium_end, rng);
                    handle_medium(sample_medium);
                };
            }
            $case(MEDIUM_ID_VOID)
            {
                auto sample_medium = vols.sample_scattering(cc, ray_o, medium_end, rng);
                handle_medium(sample_medium);
            };
            $default
            {
                cstd::unreachable();
            };
        };

        var pixel_coord = load_aligned(soa.pixel_coord + soa_index);

        $if(scattered)
        {
            $if(emit_shadow)
            {
                var shadow_soa_index = cstd::atomic_add(shadow_ray_counter, 1);
                var shadow_medium_id = cstd::select(shadow_t1 > 1, CMediumID(MEDIUM_ID_VOID), CMediumID(medium_id));

                save_aligned(pixel_coord, soa.output_shadow_pixel_coord + shadow_soa_index);
                save_aligned(shadow_beta, soa.output_shadow_beta_li + shadow_soa_index);

                save_aligned(
                    CVec4f(shadow_o, bitcast<f32>(shadow_medium_id)),
                    soa.output_shadow_ray_o_medium_id + shadow_soa_index);
                save_aligned(
                    CVec4f(shadow_d, shadow_t1),
                    soa.output_shadow_ray_d_t1 + shadow_soa_index);
                save_aligned(
                    CVec2u(bitcast<u32>(scatter_ray_time), optix::RAY_MASK_ALL),
                    soa.output_shadow_ray_time_mask + shadow_soa_index);
            };

            $if(!phase_sample.phase.is_zero())
            {
                phase_sample.dir = normalize(phase_sample.dir);

                var beta_le = scatter_beta * phase_sample.phase;
                scatter_beta = beta_le / phase_sample.pdf;

                var next_ray_o = scatter_position;
                var next_ray_d = phase_sample.dir;
                var next_ray_t1 = btrc_max_float;
                var next_ray_time = scatter_ray_time;
                var next_ray_mask = optix::RAY_MASK_ALL;
                var next_ray_medium = medium_id;

                var output_index = cstd::atomic_add(active_state_counter, 1);

                soa.output_rng[output_index] = rng;

                save_aligned(scatter_path_radiance, soa.output_path_radiance + output_index);
                save_aligned(pixel_coord, soa.output_pixel_coord + output_index);
                save_aligned(scatter_beta, soa.output_beta + output_index);

                soa.output_depth[output_index] = scatter_depth + 1;

                save_aligned(
                    CVec4f(next_ray_o, bitcast<f32>(next_ray_medium)),
                    soa.output_new_ray_o_medium_id + output_index);
                save_aligned(
                    CVec4f(next_ray_d, next_ray_t1),
                    soa.output_new_ray_d_t1 + output_index);
                save_aligned(
                    CVec2u(bitcast<u32>(next_ray_time), u32(next_ray_mask)),
                    soa.output_new_ray_time_mask + output_index);

                beta_le.additional_data = phase_sample.pdf;
                save_aligned(beta_le, soa.output_beta_le_bsdf_pdf + output_index);

                soa.next_state_index[soa_index] = output_index;
            }
            $else
            {
                film.splat_atomic(pixel_coord, Film::OUTPUT_RADIANCE, scatter_path_radiance.to_rgb());
                var output_index = total_state_count - 1 - cstd::atomic_add(inactive_state_counter, 1);
                soa.output_rng[output_index] = rng;
                soa.next_state_index[soa_index] = output_index;
            };
        }
        $else
        {
            var beta = load_aligned(soa.beta + soa_index);
            var beta_le = load_aligned(soa.beta_le_bsdf_pdf + soa_index);
            var bsdf_pdf = beta_le.additional_data;

            beta = beta * unscatter_tr;
            beta_le = beta_le * unscatter_tr;
            beta_le.additional_data = bsdf_pdf;

            save_aligned(beta, soa.beta + soa_index);
            save_aligned(beta_le, soa.beta_le_bsdf_pdf + soa_index);

            soa.rng[soa_index] = rng;
        };
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

void MediumPipeline::sample_light(
    CompileContext &cc,
    const Scene    &scene,
    ref<CVec3f>     scatter_pos,
    ref<CRNG>       rng,
    f32             time,
    ref<CVec3f>     shadow_d,
    ref<f32>        shadow_t1,
    ref<f32>        shadow_light_pdf,
    ref<CSpectrum>  shadow_li) const
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
}

BTRC_WFPT_END
