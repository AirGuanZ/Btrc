#include <chrono>
#include <iostream>

#include <btrc/builtin/camera/pinhole.h>
#include <btrc/builtin/geometry/triangle_mesh.h>
#include <btrc/builtin/light/gradient_sky.h>
#include <btrc/builtin/light/mesh_light.h>
#include <btrc/builtin/light_sampler/uniform_light_sampler.h>
#include <btrc/builtin/material/black.h>
#include <btrc/builtin/material/diffuse.h>
#include <btrc/builtin/material/glass.h>
#include <btrc/builtin/texture2d/array2d.h>
#include <btrc/builtin/texture2d/constant2d.h>
#include <btrc/builtin/renderer/wavefront.h>
#include <btrc/core/scene.h>
#include <btrc/utils/cuda/context.h>
#include <btrc/utils/optix/context.h>

using namespace btrc;
using namespace builtin;

RC<Scene> build_scene(optix::Context &optix_ctx, int width, int height)
{
    auto scene = newRC<Scene>();

    auto camera = newRC<PinholeCamera>();
    camera->set_eye({ 0, 0, 1.5f });
    camera->set_dst({ 0, 0, 0 });
    camera->set_up({ 0, 1, 0 });
    camera->set_fov_y_deg(60);
    camera->set_w_over_h(static_cast<float>(width) / height);
    scene->set_camera(std::move(camera));

    scene->set_light_sampler(newRC<UniformLightSampler>());

    auto glass_color = newRC<Constant2D>();
    glass_color->set_value(Spectrum::from_rgb(0.6f, 0.9f, 0.9f));
    auto glass_ior = newRC<Constant2D>();
    glass_ior->set_value(1.45f);
    auto glass = newRC<Glass>();
    glass->set_color(std::move(glass_color));
    glass->set_ior(std::move(glass_ior));
    auto teapot = newRC<TriangleMesh>(optix_ctx, "./asset/teapot.obj", true);
    scene->add_instance(
        Scene::Instance{
            .geometry = std::move(teapot),
            .material = std::move(glass),
            .transform = Transform{
                .translate = { 0, -0.29f, 0 },
                .scale = 0.5f,
                .rotate = Quaterion({ 1, 0, 0 }, 0)
            }
        });

    auto black = newRC<Black>();
    auto box = newRC<TriangleMesh>(optix_ctx, "./asset/cube.obj", true);
    auto box_trans = Transform{
        .translate = { 0, 0, 0 },
        .scale = 0.1f,
        .rotate = Quaterion({ -1, 1, -1 }, 1.6f)
    };
    auto light = newRC<MeshLight>(box, box_trans, 12 * Spectrum::from_rgb(1, 1, 1));
    scene->add_instance(
        Scene::Instance{
            .geometry = std::move(box),
            .material = std::move(black),
            .light = std::move(light),
            .transform = box_trans
        });

    auto tex_arr2d = newRC<Array2D>();
    tex_arr2d->initialize("./asset/tex.png", Texture::Description{
        .address_modes = {
            Texture::AddressMode::Clamp,
            Texture::AddressMode::Clamp,
            Texture::AddressMode::Clamp
        },
        .filter_mode = Texture::FilterMode::Point,
        .srgb_to_linear = false
    });
    auto tex_diffuse = newRC<Diffuse>();
    tex_diffuse->set_albedo(std::move(tex_arr2d));
    auto tex_box = newRC<TriangleMesh>(optix_ctx, "./asset/cube.obj", true);
    scene->add_instance(
        Scene::Instance{
            .geometry = std::move(tex_box),
            .material = std::move(tex_diffuse),
            .transform = Transform{
                .translate = { 0, 0.25f, 0 },
                .scale = 0.2f,
                .rotate = Quaterion({ 1, 1, 1 }, 0.6f)
            }
        });

    auto diffuse_albedo = newRC<Constant2D>();
    diffuse_albedo->set_value(0.8f);
    auto diffuse = newRC<Diffuse>();
    diffuse->set_albedo(std::move(diffuse_albedo));
    auto cbox = newRC<TriangleMesh>(optix_ctx, "./asset/cbox.obj", true);
    scene->add_instance(
        Scene::Instance{
            .geometry = std::move(cbox),
            .material = std::move(diffuse),
            .transform = Transform{
                .translate = {},
                .scale = 1,
                .rotate = Quaterion({ 1, 0, 0 }, 0)
            }
        });

    auto sky = newRC<GradientSky>();
    sky->set_up({ 0, 1, 0 });
    sky->set_lower(Spectrum::zero());
    sky->set_upper(Spectrum::one());
    //scene.set_envir_light(std::move(sky));

    scene->preprocess(optix_ctx);
    return scene;
}

void run()
{
    constexpr int width = 512, height = 512;

    cuda::Context cuda_ctx(0);
    optix::Context optix_ctx(cuda_ctx);

    CompileContext cc;
    CompileContext::push_context(&cc);
    BTRC_SCOPE_EXIT{ CompileContext::pop_context(); };

    auto scene = build_scene(optix_ctx, width, height);

    auto wfpt = newRC<WavefrontPathTracer>(optix_ctx);
    wfpt->set_params(WavefrontPathTracer::Params{
        .spp          = 128,
        .width        = width,
        .height       = height,
        .min_depth    = 20,
        .max_depth    = 40,
        .rr_threshold = 0.1f,
        .rr_cont_prob = 0.5f,
        .state_count  = 2000000,
        .albedo       = true,
        .normal       = true
    });
    wfpt->set_scene(scene);

    auto result = wfpt->render();
    result.value.save("output.exr");
    result.albedo.save("output_albedo.png");
    result.normal.save("output_normal.png");
}

int main()
{
    try
    {
        run();
    }
    catch(const std::exception &err)
    {
        std::cerr << "fatal error: " << err.what() << std::endl;
        return -1;
    }
}
