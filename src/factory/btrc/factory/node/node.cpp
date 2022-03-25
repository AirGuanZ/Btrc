#include <btrc/core/spectrum.h>
#include <btrc/factory/node/node.h>
#include <btrc/utils/math/vec3.h>
#include <btrc/utils/exception.h>

BTRC_FACTORY_BEGIN

namespace
{

    template<typename T>
    T parse_value_impl(const std::string &s);

    template<>
    int32_t parse_value_impl<int32_t>(const std::string &s)
    {
        return std::stoi(s);
    }

    template<>
    int64_t parse_value_impl<int64_t>(const std::string &s)
    {
        return std::stoll(s);
    }

    template<>
    float parse_value_impl<float>(const std::string &s)
    {
        return std::stof(s);
    }

    template<>
    double parse_value_impl<double>(const std::string &s)
    {
        return std::stod(s);
    }

    template<>
    bool parse_value_impl<bool>(const std::string &s)
    {
        return s == "true";
    }

    template<>
    std::string parse_value_impl<std::string>(const std::string &s)
    {
        return s;
    }

    Vec3f parse_vec3f(const RC<const Node> &node)
    {
        if(auto arr = node->as_array())
        {
            if(arr->get_size() == 1)
            {
                const float v = arr->get_element(0)->parse<float>();
                return Vec3f(v, v, v);
            }
            if(arr->get_size() == 3)
            {
                const float x = arr->get_element(0)->parse<float>();
                const float y = arr->get_element(1)->parse<float>();
                const float z = arr->get_element(2)->parse<float>();
                return Vec3f(x, y, z);
            }
            throw BtrcException(fmt::format("unexpected array size: {}", arr->get_size()));
        }
        if(auto val = node->as_value())
        {
            const float v = val->parse<float>();
            return Vec3f(v, v, v);
        }
        throw BtrcException("unexpected group node");
    }

    Degree parse_degree(const RC<const Node> &node)
    {
        auto grp = node->as_group();
        if(!grp)
            throw BtrcException("group expected");
        if(auto c = grp->find_child_node("deg"))
            return { c->parse<float>() };
        if(auto c = grp->find_child_node("rad"))
            return { 180.0f / btrc_pi * c->parse<float>() };
        throw BtrcException("invalid angle node");
    }

    Radian parse_radian(const RC<const Node> &node)
    {
        auto grp = node->as_group();
        if(!grp)
            throw BtrcException("group expected");
        if(auto c = grp->find_child_node("deg"))
            return { btrc_pi / 180.0f * c->parse<float>() };
        if(auto c = grp->find_child_node("rad"))
            return { c->parse<float>() };
        throw BtrcException("invalid angle node");
    }

    Spectrum parse_spectrum(const RC<const Node> &node)
    {
        if(auto arr = node->as_array())
        {
            if(arr->get_size() == 1)
            {
                const float v = arr->get_element(0)->parse<float>();
                return Spectrum::from_rgb(v, v, v);
            }
            if(arr->get_size() == 3)
            {
                const float r = arr->get_element(0)->parse<float>();
                const float g = arr->get_element(1)->parse<float>();
                const float b = arr->get_element(2)->parse<float>();
                return Spectrum::from_rgb(r, g, b);
            }
            throw BtrcException(fmt::format("unexpected array size: {}", arr->get_size()));
        }
        if(auto val = node->as_value())
        {
            const float v = val->parse<float>();
            return Spectrum::from_rgb(v, v, v);
        }
        throw BtrcException("unexpected group node");
    }

    Transform3D parse_transform(const RC<const Node> &node)
    {
        if(auto arr = node->as_array())
        {
            Transform3D ret;
            for(size_t i = 0; i < arr->get_size(); ++i)
                ret = parse_transform(arr->get_element(i)) * ret;
            return ret;
        }

        if(auto translate = node->find_child_node("translate"))
        {
            const Vec3f t = translate->parse<Vec3f>();
            return Transform3D::translate(t.x, t.y, t.z);
        }

        if(auto rotate = node->find_child_node("rotate"))
        {
            const Vec3f axis = rotate->parse_child<Vec3f>("axis");
            const Radian rad = rotate->parse_child<Radian>("angle");
            return Transform3D::rotate(axis, rad.value);
        }

        if(auto rotate_x = node->find_child_node("rotate_x"))
        {
            const float rad = rotate_x->parse<Radian>().value;
            return Transform3D::rotate_x(rad);
        }

        if(auto rotate_y = node->find_child_node("rotate_y"))
        {
            const float rad = rotate_y->parse<Radian>().value;
            return Transform3D::rotate_y(rad);
        }

        if(auto rotate_z = node->find_child_node("rotate_z"))
        {
            const float rad = rotate_z->parse<Radian>().value;
            return Transform3D::rotate_z(rad);
        }

        if(auto scale = node->find_child_node("scale"))
        {
            const Vec3f s = scale->parse<Vec3f>();
            return Transform3D::scale(s.x, s.y, s.z);
        }

        throw BtrcException("unknown transform type");
    }

} // namespace anonymous

RC<const Group> Node::as_group() const
{
    return std::dynamic_pointer_cast<const Group>(shared_from_this());
}

RC<Group> Node::as_group()
{
    return std::dynamic_pointer_cast<Group>(shared_from_this());
}

RC<const Array> Node::as_array() const
{
    return std::dynamic_pointer_cast<const Array>(shared_from_this());
}

RC<Array> Node::as_array()
{
    return std::dynamic_pointer_cast<Array>(shared_from_this());
}

RC<const Value> Node::as_value() const
{
    return std::dynamic_pointer_cast<const Value>(shared_from_this());
}

RC<Value> Node::as_value()
{
    return std::dynamic_pointer_cast<Value>(shared_from_this());
}

RC<Node> Node::find_child_node(std::string_view name)
{
    auto grp = as_group();
    if(!grp)
        throw BtrcException("group node expected");
    return grp->find_child_node(name);
}

RC<const Node> Node::find_child_node(std::string_view name) const
{
    auto grp = as_group();
    if(!grp)
        throw BtrcException("group node expected");
    return grp->find_child_node(name);
}

RC<Node> Node::child_node(std::string_view name)
{
    auto ret = find_child_node(name);
    if(!ret)
        throw BtrcException(fmt::format("child {} not found", name));
    return ret;
}

RC<const Node> Node::child_node(std::string_view name) const
{
    auto ret = find_child_node(name);
    if(!ret)
        throw BtrcException(fmt::format("child {} not found", name));
    return ret;
}

template<typename T>
T Node::parse() const
{
    BTRC_HI_TRY

    if constexpr(std::is_same_v<T, int32_t>     ||
                 std::is_same_v<T, int64_t>     ||
                 std::is_same_v<T, float>       ||
                 std::is_same_v<T, double>      ||
                 std::is_same_v<T, std::string> ||
                 std::is_same_v<T, bool>)
    {
        auto value = as_value();
        if(!value)
            throw BtrcException("value node expected");
        return parse_value_impl<T>(value->get_string());
    }
    else if constexpr(std::is_same_v<T, Vec3f>)
    {
        return parse_vec3f(shared_from_this());
    }
    else if constexpr(std::is_same_v<T, Degree>)
    {
        return parse_degree(shared_from_this());
    }
    else if constexpr(std::is_same_v<T, Radian>)
    {
        return parse_radian(shared_from_this());
    }
    else if constexpr(std::is_same_v<T, Spectrum>)
    {
        return parse_spectrum(shared_from_this());
    }
    else if constexpr(std::is_same_v<T, Transform3D>)
    {
        return parse_transform(shared_from_this());
    }
    else
    {
        throw BtrcException("unimplemented");
    }

    BTRC_HI_WRAP(fmt::format("in parsing {}", typeid(T).name()))
}

Node::Type Group::get_type() const
{
    return Type::Group;
}

void Group::insert(std::string value, RC<Node> node)
{
    auto it = children_.insert({ std::move(value), std::move(node) }).first;
    ordered_keys_.push_back(it->first);
}

RC<Node> Group::find_child_node(std::string_view name)
{
    auto it = children_.find(name);
    return it == children_.end() ? nullptr : it->second;
}

RC<const Node> Group::find_child_node(std::string_view name) const
{
    auto it = children_.find(name);
    return it == children_.end() ? nullptr : it->second;
}

RC<Node> Group::child_node(std::string_view name)
{
    auto ret = find_child_node(name);
    if(!ret)
        throw BtrcException(fmt::format("child {} not found", name));
    return ret;
}

RC<const Node> Group::child_node(std::string_view name) const
{
    auto ret = find_child_node(name);
    if(!ret)
        throw BtrcException(fmt::format("child {} not found", name));
    return ret;
}

Node::Type Array::get_type() const
{
    return Type::Array;
}

void Array::push_back(RC<Node> element)
{
    elements_.push_back(std::move(element));
}

size_t Array::get_size() const
{
    return elements_.size();
}

RC<Node> Array::get_element(size_t index)
{
    return elements_[index];
}

RC<const Node> Array::get_element(size_t index) const
{
    return elements_[index];
}

Node::Type Value::get_type() const
{
    return Type::Value;
}

void Value::set_string(std::string str)
{
    str_ = std::move(str);
}

const std::string &Value::get_string() const
{
    return str_;
}

template int32_t     Node::parse<int32_t>    () const;
template int64_t     Node::parse<int64_t>    () const;
template float       Node::parse<float>      () const;
template double      Node::parse<double>     () const;
template bool        Node::parse<bool>       () const;
template std::string Node::parse<std::string>() const;
template Vec3f       Node::parse<Vec3f>      () const;
template Degree      Node::parse<Degree>     () const;
template Radian      Node::parse<Radian>     () const;
template Spectrum    Node::parse<Spectrum>   () const;
template Transform3D Node::parse<Transform3D>() const;

BTRC_FACTORY_END
