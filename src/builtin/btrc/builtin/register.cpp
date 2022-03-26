#include <btrc/builtin/camera/pinhole.h>
#include <btrc/builtin/film_filter/box.h>
#include <btrc/builtin/film_filter/gaussian.h>
#include <btrc/builtin/geometry/triangle_mesh.h>
#include <btrc/builtin/light/gradient_sky.h>
#include <btrc/builtin/light/ibl.h>
#include <btrc/builtin/light/mesh_light.h>
#include <btrc/builtin/material/black.h>
#include <btrc/builtin/material/diffuse.h>
#include <btrc/builtin/material/disney.h>
#include <btrc/builtin/material/glass.h>
#include <btrc/builtin/material/invisible.h>
#include <btrc/builtin/material/metal.h>
#include <btrc/builtin/material/mirror.h>
#include <btrc/builtin/medium/homogeneous.h>
#include <btrc/builtin/medium/hetergeneous.h>
#include <btrc/builtin/postprocess/optix_denoiser.h>
#include <btrc/builtin/postprocess/save_to_image.h>
#include <btrc/builtin/postprocess/tonemap.h>
#include <btrc/builtin/renderer/wavefront.h>
#include <btrc/builtin/texture2d/array2d.h>
#include <btrc/builtin/texture2d/transform.h>
#include <btrc/builtin/texture3d/array3d.h>
#include <btrc/builtin/texture3d/binary.h>
#include <btrc/builtin/register.h>

BTRC_BUILTIN_BEGIN

void register_builtin_creators(factory::Factory<Camera> &factory)
{
    factory.add_creator(newBox<PinholeCameraCreator>());
}

void register_builtin_creators(factory::Factory<FilmFilter> &factory)
{
    factory.add_creator(newBox<BoxFilterCreator>());
    factory.add_creator(newBox<GaussianFilterCreator>());
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

void register_builtin_creators(factory::Factory<Material> &factory)
{
    factory.add_creator(newBox<BlackCreator>());
    factory.add_creator(newBox<DiffuseCreator>());
    factory.add_creator(newBox<DisneyMaterialCreator>());
    factory.add_creator(newBox<GlassCreator>());
    factory.add_creator(newBox<InvisibleSurfaceCreator>());
    factory.add_creator(newBox<MetalCreator>());
    factory.add_creator(newBox<MirrorCreator>());
}

void register_builtin_creators(factory::Factory<Medium> &factory)
{
    factory.add_creator(newBox<HomogeneousMediumCreator>());
    factory.add_creator(newBox<HetergeneousMediumCreator>());
}

void register_builtin_creators(factory::Factory<PostProcessor> &factory)
{
    factory.add_creator(newBox<OptixAIDenoiserCreator>());
    factory.add_creator(newBox<SaveToImageCreator>());
    factory.add_creator(newBox<ACESToneMapCreator>());
}

void register_builtin_creators(factory::Factory<Renderer> &factory)
{
    factory.add_creator(newBox<WavefrontPathTracerCreator>());
}

void register_builtin_creators(factory::Factory<Texture2D> &factory)
{
    factory.add_creator(newBox<Array2DCreator>());
    factory.add_creator(newBox<TransformTexture2DCreator>());
}

void register_builtin_creators(factory::Factory<Texture3D> &factory)
{
    factory.add_creator(newBox<Array3DCreator>());
    factory.add_creator(newBox<Texture3DBinaryOperatorCreator<BinaryOp3D::Add>>());
    factory.add_creator(newBox<Texture3DBinaryOperatorCreator<BinaryOp3D::Mul>>());
}

void register_builtin_creators(factory::Context &context)
{
    register_builtin_creators(context.get_factory<Camera>());
    register_builtin_creators(context.get_factory<FilmFilter>());
    register_builtin_creators(context.get_factory<Geometry>());
    register_builtin_creators(context.get_factory<Light>());
    register_builtin_creators(context.get_factory<Material>());
    register_builtin_creators(context.get_factory<Medium>());
    register_builtin_creators(context.get_factory<PostProcessor>());
    register_builtin_creators(context.get_factory<Renderer>());
    register_builtin_creators(context.get_factory<Texture2D>());
    register_builtin_creators(context.get_factory<Texture3D>());
}

BTRC_BUILTIN_END
