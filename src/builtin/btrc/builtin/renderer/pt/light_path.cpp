#include <btrc/builtin/renderer/pt/light_path.h>
#include <btrc/utils/intersection.h>

BTRC_PT_BEGIN

void trace_light_path(
    CompileContext             &cc,
    const TraceUtils           &utils,
    const TraceLightPathParams &params,
    const Scene                &scene,
    GlobalSampler              &sampler,
    PhotonMap                  &photon_map,
    const WorldBound           &world_bound)
{
    $scope
    {
        // select light

        auto light_sampler = scene.get_light_sampler();
        auto select_light_result = light_sampler->sample_emit(sampler.get1d());
        var light_idx = select_light_result.light_idx;
        var select_light_pdf = select_light_result.pdf;

        $if(light_idx < 0)
        {
            $exit_scope;
        };

        // sample emission

        CVec3f emit_ori, emit_dir, emit_nor;
        f32 emit_pdf_pos, emit_pdf_dir;
        CSpectrum emit_radiance;
        boolean is_light_area;

        light_sampler->access_light(light_idx, [&](const Light *light)
        {
            if(auto area = light->as_area())
            {
                auto sample_emit = area->sample_emit(cc, sampler.get5d());

                emit_ori = intersection_offset(
                    sample_emit.point.position,
                    sample_emit.point.frame.z,
                    sample_emit.direction);
                emit_dir = sample_emit.direction;

                emit_pdf_pos = select_light_pdf * sample_emit.pdf_pos;
                emit_pdf_dir = sample_emit.pdf_dir;

                emit_radiance = sample_emit.radiance;

                is_light_area = true;
            }
            else
            {
                auto envir = light->as_envir();

                var emit_sam = sampler.get5d();
                var sample_disk = sample_disk_uniform(emit_sam[0], emit_sam[1]);
                auto sample_emit = envir->sample_emit(cc, make_sample(emit_sam[2], emit_sam[3], emit_sam[4]));

                var frame = CFrame::from_z(sample_emit.direction);
                emit_ori = CVec3f(world_bound.scene_center)
                  - frame.z * world_bound.scene_radius
                  + frame.x * world_bound.scene_radius * sample_disk.x
                  + frame.y * world_bound.scene_radius * sample_disk.y;
                emit_dir = frame.z;

                emit_pdf_pos = select_light_pdf * sample_emit.pdf_dir;
                emit_pdf_dir = 1 / (world_bound.scene_radius * world_bound.scene_radius) * pdf_sample_disk_uniform();

                emit_radiance = sample_emit.radiance;

                is_light_area = false;
            }
        });

        // trace light path

        var r = CRay(emit_ori, emit_dir);

        var partial_beta = emit_radiance * cstd::abs(cos(emit_dir, emit_nor));
        var partial_pdf = emit_pdf_pos * emit_pdf_dir;

        var p_y1_area = emit_pdf_pos, p_y2_l_y1_area = 0.0f, p_y2_r_y1_area = 0.0f, p_light_y1_y2_area = 0.0f;
        var p_y_t_r_tm1_sa = 0.0f, p_y_t_l_tm1_sa = 0.0f;

        CVec3f last_nor, last_pos;
        boolean is_last_projector = false;

        var dC = 1.0f;

        i32 depth = 1;
        $loop
        {
            var hit = utils.find_closest_intersection(r);
            $if(hit.miss())
            {
                $break;
            };

            var instances = cuj::import_pointer(scene.get_device_instance_info());
            var geometries = cuj::import_pointer(scene.get_device_geometry_info());
            ref instance = instances[hit.inst_id];
            ref geometry = geometries[instance.geometry_id];
            var hit_spt = get_hitinfo(r.o, r.d, instance, geometry, hit.t, hit.prim_id, hit.uv);

            Shader::SampleBidirResult sample_bsdf;
            scene.access_material(instance.material_id, [&](const Material *material)
            {
                auto shader = material->create_shader(cc, hit_spt);
                sample_bsdf = shader->sample_bidir(cc, -r.d, sampler.get3d(), TransportMode::Importance);
            });

            // hitting y2
            $if(depth == 1)
            {
                $if(is_light_area)
                {
                    var sa_to_area = cstd::abs(cos(r.d, hit_spt.frame.z)) / length_square(r.o - hit_spt.position);
                    p_y2_l_y1_area = emit_pdf_dir * sa_to_area;
                }
                $else
                {
                    p_y2_l_y1_area = emit_pdf_dir * cstd::abs(cos(r.d, hit_spt.frame.z));
                };

                var y2_r_y1_pdf_factor = cstd::abs(cos(emit_nor, r.d)) / length_square(emit_ori - hit_spt.position);
                p_y2_r_y1_area = sample_bsdf.pdf_rev * y2_r_y1_pdf_factor;

                light_sampler->access_light(light_idx, [&](const Light *light)
                {
                    if(auto area = light->as_area())
                    {
                        p_light_y1_y2_area = area->pdf_li(cc, hit_spt.position, emit_ori, emit_nor) * y2_r_y1_pdf_factor;
                    }
                    else
                    {
                        auto envir = light->as_envir();
                        p_light_y1_y2_area = envir->pdf_li(cc, -r.d);
                    }
                });
            };

            // hitting y3, y4, ...
            $if(depth >= 2)
            {
                var p_y_tm1_r_tm2_area = p_y_t_r_tm1_sa * cstd::abs(cos(last_nor, r.d))
                                       / length_square(last_pos - hit_spt.position);

                var p_y_t_l_tm1 = p_y_t_l_tm1_sa * cstd::abs(cos(hit_spt.frame.z, r.d))
                                / length_square(r.o - hit_spt.position);

                dC = dC * p_y_tm1_r_tm2_area / p_y_t_l_tm1;
            };

            // rr
            $if(depth >= params.min_depth)
            {
                $if(depth >= params.max_depth)
                {
                    $break;
                };

                var sam1 = sampler.get1d();
                $if(partial_beta.get_lum() < params.rr_threshold)
                {
                    $if(sam1 > params.rr_cont_prob)
                    {
                        $break;
                    };
                    partial_beta = partial_beta / params.rr_cont_prob;
                };
            };

            // caustic segment
            $if(depth >= 3 & is_last_projector & ((instance.flag & INSTANCE_FLAG_CAUSTICS_RECRIVER) != 0))
            {
                PhotonMap::CPhoton photon;
                photon.pos = hit_spt.position;
                photon.wr = -r.d;
                photon.beta = partial_beta;
                photon.pdf = partial_pdf;
                photon.pdf_factor_xc_r = cstd::abs(cos(r.d, last_nor)) / length_square(r.o - last_pos);
                photon.pdf_light = p_light_y1_y2_area;
                photon.pdf_y1 = p_y1_area;
                photon.pdf_y2_r = p_y2_r_y1_area;
                photon.pdf_y2_l = p_y2_l_y1_area;
                photon.dC = dC;
                photon_map.add_photon(photon);
            };

            partial_beta = partial_beta * sample_bsdf.bsdf * cstd::abs(cos(sample_bsdf.dir, hit_spt.frame.z));
            partial_pdf = partial_pdf * sample_bsdf.pdf;

            r.o = intersection_offset(hit_spt.position, hit_spt.frame.z, sample_bsdf.dir);
            r.d = normalize(sample_bsdf.dir);

            p_y_t_r_tm1_sa = sample_bsdf.pdf_rev;
            p_y_t_l_tm1_sa = sample_bsdf.pdf;

            last_pos = hit_spt.position;
            last_nor = hit_spt.frame.z;

            is_last_projector = (instance.flag & INSTANCE_FLAG_CAUSTICS_PROJECTOR) != 0;
        };
    };
}

BTRC_PT_END
