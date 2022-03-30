#pragma once

#include <btrc/core/spectrum.h>
#include <btrc/utils/math/math.h>

BTRC_BEGIN

class Camera;
class Geometry;
class FilmFilter;
class Light;
class Material;
class Medium;
class PostProcessor;
class Renderer;
class Texture2D;
class Texture3D;

template<typename T>
class ObjectSlot;

class ObjectProxy
{
public:

    virtual ~ObjectProxy() = default;

    virtual void add_property(std::string name, float    &value, std::string tooltip = {});
    virtual void add_property(std::string name, Vec2f    &value, std::string tooltip = {});
    virtual void add_property(std::string name, Vec3f    &value, std::string tooltip = {});
    virtual void add_property(std::string name, Vec4f    &value, std::string tooltip = {});
    virtual void add_property(std::string name, Spectrum &value, std::string tooltip = {});

    virtual void add_object(std::string name, ObjectSlot<Camera>        &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<Geometry>      &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<FilmFilter>    &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<Light>         &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<Material>      &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<Medium>        &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<PostProcessor> &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<Renderer>      &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<Texture2D>     &slot, std::string tooltip = {});
    virtual void add_object(std::string name, ObjectSlot<Texture3D>     &slot, std::string tooltip = {});
};

BTRC_END
