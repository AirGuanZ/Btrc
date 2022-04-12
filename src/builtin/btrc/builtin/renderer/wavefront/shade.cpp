#include <array>

#include <btrc/builtin/renderer/wavefront/helper.h>
#include <btrc/builtin/renderer/wavefront/shade.h>
#include <btrc/core/film.h>
#include <btrc/utils/intersection.h>

BTRC_WFPT_BEGIN

using namespace cuj;
using namespace shade_pipeline_detail;

namespace
{
    
    constexpr char SHADE_KERNEL_NAME[] = "shade_kernel";
    
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
            CSOAParams soa)
    {
        const WFPTScene wfpt_scene = { cc, scene, world_diagonal };

        var soa_index = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
        $if(soa_index >= total_state_count)
        {
            $return();
        };

        // load basic path states

        auto inct_flag = soa.inct.load_flag(soa_index);
        auto path = soa.path.load(soa_index);
        auto load_ray = soa.ray.load(soa_index);
        auto bsdf_le = soa.bsdf_le.load(soa_index);

        GlobalSampler sampler({ film.width(), film.height() }, path.sampler_state);

        // handle miss le

        $if(!inct_flag.is_intersected)
        {
            var splat_path_rad = CSpectrum::zero();
            if(scene.get_light_sampler()->get_envir_light())
            {
                splat_path_rad = eval_miss_le(
                    wfpt_scene, load_ray.ray.o, load_ray.ray.d,
                    bsdf_le.beta_le, bsdf_le.bsdf_pdf);
            }
            $if(!inct_flag.is_scattered)
            {
                splat_path_rad = splat_path_rad + path.path_radiance;
            };
            film.splat_atomic(path.pixel_coord, Film::OUTPUT_RADIANCE, splat_path_rad.to_rgb());
            $return();
        };

        // get detailed intersection

        auto inct_detail = soa.inct.load_detail(soa_index);

        var instances = const_data(std::span{
            scene.get_host_instance_info(), static_cast<size_t>(scene.get_instance_count()) });
        var geometries = const_data(std::span{
            scene.get_host_geometry_info(), static_cast<size_t>(scene.get_geometry_count()) });

        ref<CInstanceInfo> instance = instances[inct_flag.instance_id];
        ref<CGeometryInfo> geometry = geometries[instance.geometry_id];
        var inct = get_intersection(
            load_ray.ray.o, load_ray.ray.d, instance, geometry,
            inct_detail.t, inct_detail.prim_id, inct_detail.uv);

        // handle le at intersection

        var le_rad = handle_intersected_light(
            wfpt_scene, load_ray.ray.o, load_ray.ray.d, inct,
            bsdf_le.beta_le, bsdf_le.bsdf_pdf, instance.light_id);
        $if(inct_flag.is_scattered)
        {
            film.splat_atomic(path.pixel_coord, Film::OUTPUT_RADIANCE, le_rad.to_rgb());
            $return();
        };
        path.path_radiance = path.path_radiance + le_rad;

        // rr

        var rr_exit = simple_russian_roulette(
            path.beta, path.depth, sampler, SimpleRussianRouletteParams{
                .min_depth      = shade_params.min_depth,
                .max_depth      = shade_params.max_depth,
                .beta_threshold = shade_params.rr_threshold,
                .cont_prob      = shade_params.rr_cont_prob
            });

        $if(rr_exit)
        {
            film.splat_atomic(path.pixel_coord, Film::OUTPUT_RADIANCE, path.path_radiance.to_rgb());
            $return();
        };

        // sample light

        var shadow_bsdf_val = CSpectrum::zero();
        var shadow_bsdf_pdf = 0.0f;
        auto sample_shadow = sample_surface_light_li(
            wfpt_scene, inct.position, inct.frame.z, sampler);

        // eval bsdf

        Shader::SampleResult bsdf_sample;

        CVec3f gbuffer_albedo;
        CVec3f gbuffer_normal;

        auto handle_material = [&](const Material *mat)
        {
            auto shader = mat->create_shader(cc, inct);

            // gbuffer

            $if(path.depth == 0)
            {
                gbuffer_albedo = shader->albedo(cc).to_rgb();
                gbuffer_normal = shader->normal(cc);
            };

            // sample bsdf

            bsdf_sample = shader->sample(
                cc, -load_ray.ray.d, sampler.get3d(), TransportMode::Radiance);

            // shadow ray

            $if(sample_shadow.success)
            {
                shadow_bsdf_val = shader->eval(
                    cc, sample_shadow.d, -load_ray.ray.d, TransportMode::Radiance);
                sample_shadow.success = !shadow_bsdf_val.is_zero();
                $if(sample_shadow.success)
                {
                    shadow_bsdf_pdf = shader->pdf(
                        cc, sample_shadow.d, -load_ray.ray.d, TransportMode::Radiance);
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

        // emit shadow ray

        $if(sample_shadow.success)
        {
            var shadow_soa_index = cstd::atomic_add(shadow_ray_counter, 1);

            CMediumID shadow_medium_id;
            var shadow_d_out = dot(inct.frame.z, sample_shadow.d) > 0;
            var last_ray_out = dot(inct.frame.z, load_ray.ray.d) < 0;
            $if(shadow_d_out == last_ray_out)
            {
                shadow_medium_id = load_ray.medium_id;
            }
            $elif(shadow_d_out)
            {
                shadow_medium_id = instance.outer_medium_id;
            }
            $else
            {
                shadow_medium_id = instance.inner_medium_id;
            };

            var cos = cstd::abs(dot(normalize(sample_shadow.d), inct.frame.z));
            var beta_li = sample_shadow.li * path.beta * shadow_bsdf_val * cos
                        / (shadow_bsdf_pdf + sample_shadow.light_pdf);

            soa.shadow_ray.save(
                shadow_soa_index,
                path.pixel_coord,
                beta_li,
                CRay(sample_shadow.o, sample_shadow.d, sample_shadow.t1),
                shadow_medium_id);
        };

        // write g-buffer

        $if(path.depth == 0)
        {
            std::vector<std::pair<std::string_view, Film::CValue>> values = {
                { Film::OUTPUT_ALBEDO, gbuffer_albedo },
                { Film::OUTPUT_NORMAL, gbuffer_normal }
            };
            film.splat_atomic(path.pixel_coord, values);
        };

        // emit next ray

        $if(!bsdf_sample.bsdf.is_zero())
        {
            bsdf_sample.dir = normalize(bsdf_sample.dir);
            var cos = cstd::abs(dot(bsdf_sample.dir, inct.frame.z));
            
            var new_beta_le = path.beta * cos * bsdf_sample.bsdf;
            path.beta = new_beta_le / bsdf_sample.pdf;

            var next_ray_o = intersection_offset(inct.position, inct.frame.z, bsdf_sample.dir);
            var next_ray_d = bsdf_sample.dir;

            var next_ray_out = dot(inct.frame.z, next_ray_d) > 0;
            var next_ray_medium_id = cstd::select(
                next_ray_out, instance.outer_medium_id, instance.inner_medium_id);

            var stored_bsdf_pdf = cstd::select(bsdf_sample.is_delta, -bsdf_sample.pdf, f32(bsdf_sample.pdf));

            var output_index = cstd::atomic_add(active_state_counter, 1);
            soa.output_path.save(output_index, path.depth + 1, path.pixel_coord, path.beta, path.path_radiance, sampler);
            soa.output_bsdf_le.save(output_index, new_beta_le, stored_bsdf_pdf);
            soa.output_ray.save(output_index, CRay(next_ray_o, next_ray_d), next_ray_medium_id);
        }
        $else
        {
            film.splat_atomic(path.pixel_coord, Film::OUTPUT_RADIANCE, path.path_radiance.to_rgb());
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
    int32_t *shadow_ray_counter     = active_state_counter + 1;

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
