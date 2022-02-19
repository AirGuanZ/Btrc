#include <btrc/builtin/camera/pinhole.h>
#include <btrc/builtin/geometry/triangle_mesh.h>
#include <btrc/builtin/light/gradient_sky.h>
#include <btrc/builtin/light/ibl.h>
#include <btrc/builtin/light/mesh_light.h>
#include <btrc/builtin/light_sampler/uniform_light_sampler.h>
#include <btrc/builtin/material/black.h>
#include <btrc/builtin/material/diffuse.h>
#include <btrc/builtin/material/glass.h>
#include <btrc/builtin/renderer/wavefront.h>
#include <btrc/builtin/texture2d/array2d.h>
#include <btrc/builtin/register.h>

BTRC_BUILTIN_BEGIN

void register_builtin_creators(factory::Factory<Camera> &factory)
{
    factory.add_creator(newBox<PinholeCameraCreator>());
}

void register_builtin_creators(factory::Factory<Geometry> &factory)
{
    factory.add_creator(newBox<TriangleMeshCreator>());
}

void register_builtin_creators(factory::Factory<Light> &factory)
{
    factory.add_creator(newBox<GradientSkyCreator>());
    factory.add_creator(newBox<IBLCreator>());
    factory.add_creator(newBox<MeshLightCreator>());
}

void register_builtin_creators(factory::Factory<LightSampler> &factory)
{
    factory.add_creator(newBox<UniformLightSamplerCreator>());
}

void register_builtin_creators(factory::Factory<Material> &factory)
{
    factory.add_creator(newBox<BlackCreator>());
    factory.add_creator(newBox<DiffuseCreator>());
    factory.add_creator(newBox<GlassCreator>());
}

void register_builtin_creators(factory::Factory<Renderer> &factory)
{
    factory.add_creator(newBox<WavefrontPathTracerCreator>());
}

void register_builtin_creators(factory::Factory<Texture2D> &factory)
{
    factory.add_creator(newBox<Array2DCreator>());
}

void register_builtin_creators(factory::Context &context)
{
    register_builtin_creators(context.get_factory<Camera>());
    register_builtin_creators(context.get_factory<Geometry>());
    register_builtin_creators(context.get_factory<Light>());
    register_builtin_creators(context.get_factory<LightSampler>());
    register_builtin_creators(context.get_factory<Material>());
    register_builtin_creators(context.get_factory<Renderer>());
    register_builtin_creators(context.get_factory<Texture2D>());
}

BTRC_BUILTIN_END
