#pragma once

#include <btrc/core/camera.h>
#include <btrc/core/film_filter.h>
#include <btrc/core/geometry.h>
#include <btrc/core/light.h>
#include <btrc/core/material.h>
#include <btrc/core/medium.h>
#include <btrc/core/post_processor.h>
#include <btrc/core/renderer.h>
#include <btrc/core/texture2d.h>
#include <btrc/core/texture3d.h>
#include <btrc/factory/node/node.h>
#include <btrc/factory/path_resolver.h>
#include <btrc/utils/exception.h>
#include <btrc/utils/uncopyable.h>

BTRC_FACTORY_BEGIN

class Context;

template<typename T>
const char *get_object_typename();

#define REGISTER_OBJECT_TYPENAME(TYPE, NAME) \
    template<>                               \
    inline const char *get_object_typename<TYPE>() { return #NAME; }

REGISTER_OBJECT_TYPENAME(Camera,        camera)
REGISTER_OBJECT_TYPENAME(FilmFilter,    film_filter)
REGISTER_OBJECT_TYPENAME(Geometry,      geometry)
REGISTER_OBJECT_TYPENAME(Light,         light)
REGISTER_OBJECT_TYPENAME(Material,      material)
REGISTER_OBJECT_TYPENAME(Medium,        medium)
REGISTER_OBJECT_TYPENAME(PostProcessor, post_processor)
REGISTER_OBJECT_TYPENAME(Renderer,      renderer)
REGISTER_OBJECT_TYPENAME(Texture2D,     texture2d)
REGISTER_OBJECT_TYPENAME(Texture3D,     texture3d)

#undef REGISTER_OBJECT_TYPENAME

template<typename T> requires std::is_base_of_v<Object, T>
class Creator
{
public:

    virtual ~Creator() = default;

    virtual std::string get_name() const = 0;

    virtual RC<T> create(RC<const Node> node, Context &context) = 0;
};

template<typename T>
class Factory : public Uncopyable
{
public:

    void add_creator(Box<Creator<T>> creator);

    RC<T> create(RC<const Node> node, Context &ctx);

private:

    std::string object_typename_;
    std::map<std::string, Box<Creator<T>>, std::less<>> creators_;
};

class Context : public Uncopyable
{
public:

    explicit Context(optix::Context &optix_ctx);

    template<typename T>
    Factory<T> &get_factory();

    template<typename T>
    RC<T> create(RC<const Node> node);

    template<typename T>
    void add_creator(Box<Creator<T>> creator);

    optix::Context &get_optix_context();

    void add_path_mapping(std::string_view name, std::string value);

    std::filesystem::path resolve_path(std::string_view path) const;

private:

    template<typename...Ts>
    using FactoryTuple = std::tuple<Factory<Ts>...>;

    FactoryTuple<
        Camera,
        FilmFilter,
        Geometry,
        Light,
        Material,
        Medium,
        PostProcessor,
        Renderer,
        Texture2D,
        Texture3D
    > factorys_;

    optix::Context &optix_ctx_;
    RC<Node> root_node_;
    std::map<RC<const Node>, RC<Object>> object_pool_;
    PathResolver path_resolver_;
};

// ========================== impl ==========================

class Constant2DCreator : public Creator<Texture2D>
{
public:

    std::string get_name() const override { return "constant"; }

    RC<Texture2D> create(RC<const Node> node, Context &context) override
    {
        auto value = node->parse_child<Spectrum>("value");
        auto result = newRC<Constant2D>();
        result->set_value(value);
        return result;
    }
};

class Constant3DCreator : public Creator<Texture3D>
{
public:

    std::string get_name() const override { return "constant"; }

    RC<Texture3D> create(RC<const Node> node, Context &context) override
    {
        auto value = node->parse_child<Spectrum>("value");
        auto result = newRC<Constant3D>();
        result->set_value(value);
        return result;
    }
};

class TransformMediumCreator : public Creator<Medium>
{
public:

    std::string get_name() const override { return "transform"; }

    RC<Medium> create(RC<const Node> node, Context &context) override
    {
        Transform3D world_to_local;
        if(auto tnode = node->find_child_node("world_to_local"))
            world_to_local = tnode->parse<Transform3D>();
        else
            world_to_local = node->parse_child<Transform3D>("local_to_world").inverse();
        auto transformed = context.create<Medium>(node->child_node("medium"));
        auto result = newRC<TransformMedium>();
        result->set_transform(world_to_local);
        result->set_tranformed(std::move(transformed));
        return result;
    }
};

template<typename T>
void Factory<T>::add_creator(Box<Creator<T>> creator)
{
    auto name = creator->get_name();
    if(creators_.contains(name))
        throw BtrcException("repeated creator name: " + name);
    creators_.insert({ std::move(name), std::move(creator) });
}

template<typename T>
RC<T> Factory<T>::create(RC<const Node> node, Context &ctx)
{
    auto type = node->parse_child<std::string>("type");
    auto it = creators_.find(type);
    if(it == creators_.end())
    {
        throw BtrcException(fmt::format(
            "unknown {} type: {}", get_object_typename<T>(), type));
    }
    auto creator = it->second.get();
    return creator->create(node, ctx);
}

inline Context::Context(optix::Context &optix_ctx)
    : optix_ctx_(optix_ctx)
{
    std::get<Factory<Texture2D>>(factorys_).add_creator(newBox<Constant2DCreator>());
    std::get<Factory<Texture3D>>(factorys_).add_creator(newBox<Constant3DCreator>());
    std::get<Factory<Medium>>(factorys_).add_creator(newBox<TransformMediumCreator>());
}

template<typename T>
Factory<T> &Context::get_factory()
{
    return std::get<Factory<T>>(factorys_);
}

template<typename T>
RC<T> Context::create(RC<const Node> node)
{
    BTRC_HI_TRY

    if(auto it = object_pool_.find(node); it != object_pool_.end())
        return std::dynamic_pointer_cast<T>(it->second);
    if constexpr(std::is_same_v<T, Texture2D>)
    {
        if(node->get_type() == Node::Type::Value || node->get_type() == Node::Type::Array)
        {
            auto result = newRC<Constant2D>();
            auto value = node->parse<Spectrum>();
            result->set_value(value);
            return result;
        }
    }
    if constexpr(std::is_same_v<T, Texture3D>)
    {
        if(node->get_type() == Node::Type::Value || node->get_type() == Node::Type::Array)
        {
            auto result = newRC<Constant3D>();
            auto value = node->parse<Spectrum>();
            result->set_value(value);
            return result;
        }
    }
    auto obj = std::get<Factory<T>>(factorys_).create(node, *this);
    object_pool_[node] = obj;
    return obj;

    BTRC_HI_WRAP(fmt::format("in creating {}", get_object_typename<T>()))
}

template<typename T>
void Context::add_creator(Box<Creator<T>> creator)
{
    std::get<Factory<T>>(factorys_).add_creator(std::move(creator));
}

inline optix::Context &Context::get_optix_context()
{
    return optix_ctx_;
}

inline void Context::add_path_mapping(std::string_view name, std::string value)
{
    path_resolver_.add_env_value(name, std::move(value));
}

inline std::filesystem::path Context::resolve_path(std::string_view path) const
{
    return path_resolver_.resolve(path);
}

BTRC_FACTORY_END
