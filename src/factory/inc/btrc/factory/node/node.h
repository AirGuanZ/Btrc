#pragma once

#include <format>
#include <map>
#include <span>
#include <vector>

#include <btrc/common.h>

BTRC_FACTORY_BEGIN

class Group;
class Array;
class Value;

class Node : public std::enable_shared_from_this<Node>
{
public:

    enum class Type
    {
        Group,
        Array,
        Value
    };

    virtual ~Node() = default;

    virtual Type get_type() const = 0;

    RC<const Group> as_group() const;
    RC<Group>       as_group();
    RC<const Array> as_array() const;
    RC<Array>       as_array();
    RC<const Value> as_value() const;
    RC<Value>       as_value();

    template<typename T>
    T parse_child(std::string_view name) const;

    template<typename T>
    T parse_child_or(std::string_view name, T default_value) const;
};

class Group : public Node
{
public:

    Type get_type() const override;

    void insert(std::string value, RC<Node> node);

    RC<Node>       find_child_node(std::string_view name);
    RC<const Node> find_child_node(std::string_view name) const;

    auto begin() { return children_.begin(); }
    auto end() { return children_.end(); }

    auto begin() const { return children_.begin(); }
    auto end() const { return children_.end(); }

    auto &get_ordered_keys() const { return ordered_keys_; }

private:

    std::map<std::string, RC<Node>, std::less<>> children_;
    std::vector<std::string_view> ordered_keys_;
};

class Array : public Node
{
public:

    Type get_type() const override;

    void push_back(RC<Node> element);

    size_t get_size() const;

    RC<Node>       get_element(size_t index);
    RC<const Node> get_element(size_t index) const;

    auto begin() { return elements_.begin(); }
    auto end() { return elements_.end(); }

    auto begin() const { return elements_.begin(); }
    auto end() const { return elements_.end(); }

private:

    std::vector<RC<Node>> elements_;
};

class Value : public Node
{
public:

    Type get_type() const override;

    void set_string(std::string str);

    const std::string &get_string() const;

    template<typename T>
    T parse() const;

private:

    std::string str_;
};

// ========================== impl ==========================

template<typename T>
T Node::parse_child(std::string_view name) const
{
    auto child = this->as_group()->find_child_node(name);
    if(!child)
    {
        throw BtrcException(std::format(
            "child {} is not found", name));
    }
    auto value = child->as_value();
    if(!value)
    {
        throw BtrcException(std::format(
            "child {} is expected to be of 'value' type", name));
    }
    return value->parse<T>();
}

template<typename T>
T Node::parse_child_or(std::string_view name, T default_value) const
{
    auto child = this->as_group()->find_child_node(name);
    if(!child)
        return default_value;
    auto value = child->as_value();
    if(!value)
    {
        throw BtrcException(std::format(
            "child {} is expected to be of 'value' type", name));
    }
    return value->parse<T>();
}

BTRC_FACTORY_END
