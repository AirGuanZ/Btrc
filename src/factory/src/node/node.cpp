#include <btrc/factory/node/node.h>

BTRC_FACTORY_BEGIN

namespace
{

    template<typename T>
    T parse_impl(const std::string &s);

    template<>
    int32_t parse_impl<int32_t>(const std::string &s)
    {
        return std::stoi(s);
    }

    template<>
    int64_t parse_impl<int64_t>(const std::string &s)
    {
        return std::stoll(s);
    }

    template<>
    float parse_impl<float>(const std::string &s)
    {
        return std::stof(s);
    }

    template<>
    double parse_impl<double>(const std::string &s)
    {
        return std::stod(s);
    }

    template<>
    bool parse_impl<bool>(const std::string &s)
    {
        return s == "true";
    }

    template<>
    std::string parse_impl<std::string>(const std::string &s)
    {
        return s;
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

template<typename T>
T Value::parse() const
{
    return parse_impl<T>(str_);
}

template int32_t     Value::parse<int32_t>    () const;
template int64_t     Value::parse<int64_t>    () const;
template float       Value::parse<float>      () const;
template double      Value::parse<double>     () const;
template bool        Value::parse<bool>       () const;
template std::string Value::parse<std::string>() const;

BTRC_FACTORY_END
