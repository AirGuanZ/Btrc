#include <btrc/builtin/renderer/wavefront/medium.h>
#include <btrc/builtin/renderer/wavefront/helper.h>
#include <btrc/utils/intersection.h>

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
    CompileContext    &cc,
    Film              &film,
    const Scene       &scene,
    const ShadeParams &shade_params,
    float              world_diagonal)
{
    using namespace cuj;

    kernel(KERNEL, [&cc, &film, &shade_params, &scene, world_diagonal](
        i32        total_state_count,
        ptr<i32>   active_state_counter,
        ptr<i32>   shadow_ray_counter,
        CSOAParams soa)
    {
        const WFPTScene wfpt_scene = { cc, scene, world_diagonal };

        var soa_index = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
        $if(soa_index >= total_state_count)
        {
            $return();
        };

        auto inct_flag = soa.inct.load_flag(soa_index);
        var intersected = inct_flag.is_intersected;
        var inst_id = inct_flag.instance_id;

        auto load_ray = soa.ray.load(soa_index);
        var ray = load_ray.ray;
        var ray_medium_id = load_ray.medium_id;
        
        // resolve medium id

        CMediumID medium_id;
        CVec3f medium_end;
        $if(intersected)
        {
            var instances = const_data(std::span{
                scene.get_host_instance_info(), static_cast<size_t>(scene.get_instance_count()) });
            var geometries = const_data(std::span{
                scene.get_host_geometry_info(), static_cast<size_t>(scene.get_geometry_count()) });

            auto inct_detail = soa.inct.load_detail(soa_index);
            ref instance = instances[inst_id];
            ref geometry = geometries[instance.geometry_id];
            var local_normal = load_aligned(geometry.geometry_ez_tex_coord_u_ca + inct_detail.prim_id).xyz();
            var inct_nor = instance.transform.apply_to_normal(local_normal);
            var inct_medium_id = cstd::select(dot(ray.d, inct_nor) < 0, instance.outer_medium_id, instance.inner_medium_id);

            medium_id = resolve_mediums(inct_medium_id, ray_medium_id);
            medium_end = intersection_offset(ray.o + inct_detail.t * ray.d, inct_nor, -ray.d);
        }
        $else
        {
            medium_id = scene.get_volume_primitive_medium_id();
            medium_end = ray.o + normalize(ray.d) * world_diagonal;
        };

        soa.ray.save(soa_index, ray, medium_id);

        boolean scattered = false;

        boolean emit_shadow = false;
        CVec3f shadow_o;
        CVec3f shadow_d;
        f32 shadow_t1;
        f32 shadow_light_pdf;
        CSpectrum shadow_beta;

        PhaseShader::SampleResult phase_sample;
        CVec3f scatter_position;

        CSpectrum unscatter_tr;

        auto path = soa.path.load(soa_index);
        GlobalSampler sampler({ film.width(), film.height() }, path.sampler_state);

        auto handle_medium = [&](const Medium::SampleResult &sample_medium)
        {
            $if(sample_medium.scattered)
            {
                scattered = true;
                scatter_position = sample_medium.position;
                path.beta = path.beta * sample_medium.throughput;
                soa.inct.save_flag(soa_index, intersected, true, inst_id);

                // terminate

                var rr_exit = simple_russian_roulette(
                    path.beta, path.depth, sampler,
                    SimpleRussianRouletteParams{
                        .min_depth = shade_params.min_depth,
                        .max_depth = shade_params.max_depth,
                        .beta_threshold = shade_params.rr_threshold,
                        .cont_prob = shade_params.rr_cont_prob
                    });

                $if(rr_exit)
                {
                    film.splat_atomic(path.pixel_coord, Film::OUTPUT_RADIANCE, path.path_radiance.to_rgb());
                    $return();
                };

                // generate shadow ray

                shadow_o = sample_medium.position;

                CSpectrum shadow_li;
                emit_shadow = sample_light_li(
                    wfpt_scene, sample_medium.position, sampler, shadow_d, shadow_t1, shadow_light_pdf, shadow_li);
                
                $if(emit_shadow)
                {
                    var shadow_phase_val = sample_medium.shader->eval(cc, shadow_d, -ray.d);
                    var shadow_phase_pdf = sample_medium.shader->pdf(cc, shadow_d, -ray.d);
                    shadow_beta = shadow_li * path.beta * shadow_phase_val / (shadow_phase_pdf + shadow_light_pdf);
                };

                // generate next ray

                phase_sample = sample_medium.shader->sample(cc, -ray.d, sampler.get3d());
            };
        };

        $switch(medium_id)
        {
            for(int i = 0; i < scene.get_medium_count(); ++i)
            {
                $case(MediumID(i))
                {
                    auto medium = scene.get_medium(i);
                    auto sample_medium = medium->sample(cc, ray.o, medium_end, sampler);
                    $if(sample_medium.scattered)
                    {
                        unscatter_tr = medium->tr(cc, ray.o, medium_end, sampler);
                    }
                    $else
                    {
                        unscatter_tr = sample_medium.throughput;
                    };
                    auto [beta_le, bsdf_pdf] = soa.bsdf_le.load(soa_index);
                    soa.bsdf_le.save(soa_index, beta_le *unscatter_tr, bsdf_pdf);
                    handle_medium(sample_medium);
                };
            }
            $default
            {
                cstd::unreachable();
            };
        };

        $if(scattered)
        {
            $if(emit_shadow)
            {
                var shadow_soa_index = cstd::atomic_add(shadow_ray_counter, 1);
                var shadow_medium_id = cstd::select(
                    shadow_t1 > 1, CMediumID(scene.get_volume_primitive_medium_id()), CMediumID(medium_id));

                soa.shadow_ray.save(shadow_soa_index, path.pixel_coord, shadow_beta, CRay(shadow_o, shadow_d, shadow_t1), shadow_medium_id);
            };

            $if(!phase_sample.phase.is_zero())
            {
                phase_sample.dir = normalize(phase_sample.dir);

                var beta_le = path.beta * phase_sample.phase;
                path.beta = beta_le / phase_sample.pdf;

                var next_ray_o = scatter_position;
                var next_ray_d = phase_sample.dir;
                var next_ray_t1 = btrc_max_float;
                var next_ray_medium = medium_id;

                var output_index = cstd::atomic_add(active_state_counter, 1);

                soa.output_path.save(output_index, path.depth + 1, path.pixel_coord, path.beta, path.path_radiance, sampler);
                soa.output_bsdf_le.save(output_index, beta_le, phase_sample.pdf);
                soa.output_ray.save(output_index, CRay(next_ray_o, next_ray_d, next_ray_t1), next_ray_medium);
            }
            $else
            {
                film.splat_atomic(path.pixel_coord, Film::OUTPUT_RADIANCE, path.path_radiance.to_rgb());
            };
        }
        $else
        {
            soa.path.save_sampler(soa_index, sampler);
        };
    });
}

void MediumPipeline::initialize(RC<cuda::Module> cuda_module, RC<cuda::Buffer<StateCounters>> counters, const Scene &scene)
{
    cuda_module_ = std::move(cuda_module);
    state_counters_ = std::move(counters);
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
}

void MediumPipeline::sample_scattering(int total_state_count, const SOAParams &soa)
{
    assert(cuda_module_->is_linked());

    StateCounters *device_counters = state_counters_->get();
    int32_t *active_state_counter = reinterpret_cast<int32_t *>(device_counters);
    int32_t *shadow_ray_counter = active_state_counter + 2;

    constexpr int BLOCK_DIM = 256;
    const int thread_count = total_state_count;
    const int block_count = up_align(thread_count, BLOCK_DIM) / BLOCK_DIM;

    cuda_module_->launch(
        KERNEL,
        { block_count, 1, 1 },
        { BLOCK_DIM, 1, 1 },
        total_state_count,
        active_state_counter,
        shadow_ray_counter,
        soa);
}

BTRC_WFPT_END
