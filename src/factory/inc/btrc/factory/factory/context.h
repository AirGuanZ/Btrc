#pragma once

#include <btrc/core/common/context.h>
#include <btrc/core/utils/uncopyable.h>
#include <btrc/factory/node/node.h>

#include <btrc/core/camera/camera.h>
#include <btrc/core/geometry/geometry.h>
#include <btrc/core/light/light.h>
#include <btrc/core/material/material.h>
#include <btrc/core/renderer/renderer.h>
#include <btrc/core/texture2d/texture2d.h>

BTRC_FACTORY_BEGIN

class Context;

template<typename T>
const char *get_object_typename();

#define REGISTER_OBJECT_TYPENAME(TYPE, NAME) \
    template<>                               \
    inline const char *get_object_typename<TYPE>() { return #NAME; }

REGISTER_OBJECT_TYPENAME(core::Camera,    camera)
REGISTER_OBJECT_TYPENAME(core::Geometry,  geometry)
REGISTER_OBJECT_TYPENAME(core::Light,     light)
REGISTER_OBJECT_TYPENAME(core::Material,  material)
REGISTER_OBJECT_TYPENAME(core::Renderer,  renderer)
REGISTER_OBJECT_TYPENAME(core::Texture2D, texture2d)

template<typename T> requires std::is_base_of_v<core::Object, T>
class Creator
{
public:

    virtual ~Creator() = default;

    virtual std::string get_name() const = 0;

    virtual RC<T> create(RC<const Node> node, Context &ctx) = 0;
};

template<typename T>
class Factory : public core::Uncopyable
{
public:

    void add_creator(Box<Creator<T>> creator);

    RC<T> create(RC<const Group> node, Context &ctx);

private:

    std::string object_typename_;
    std::map<std::string, Box<Creator<T>>, std::less<>> creators_;
};

class Context : public core::Uncopyable
{
public:

    Context();

    template<typename T>
    RC<T> create(RC<const Node> node);

private:

    template<typename...Ts>
    using FactoryTuple = std::tuple<Factory<Ts>...>;

    FactoryTuple<
        core::Camera,
        core::Geometry,
        core::Light,
        core::Material,
        core::Renderer,
        core::Texture2D
    > factorys_;

    RC<Node> root_node_;
    std::map<RC<const Node>, RC<core::Object>> object_pool_;
};

// ========================== impl ==========================

template<typename T>
void Factory<T>::add_creator(Box<Creator<T>> creator)
{
    auto name = creator->get_name();
    if(creators_.contains(name))
        throw BtrcFactoryException("repeated creator name: " + name);
    creators_.insert({ std::move(name), std::move(creator) });
}

template<typename T>
RC<T> Factory<T>::create(RC<const Group> node, Context &ctx)
{
    auto type = node->parse_child<std::string>("type");
    auto it = creators_.find(type);
    if(it == creators_.end())
    {
        throw BtrcFactoryException(std::format(
            "unknown {} type: {}", get_object_typename<T>(), type));
    }
    auto creator = it->second.get();
    return creator->create(node, ctx);
}

template<typename T>
RC<T> Context::create(RC<const Node> node)
{
    if(auto it = object_pool_.find(node); it != object_pool_.end())
        return std::dynamic_pointer_cast<T>(it->second);
    auto obj = std::get<Factory<T>>(factorys_).create(node, *this);
    object_pool_[node] = obj;
    return obj;
}

BTRC_FACTORY_END
