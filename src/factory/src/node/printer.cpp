#include <btrc/factory/node/printer.h>

BTRC_FACTORY_BEGIN

void JSONPrinter::set_root_node(RC<const Node> node)
{
    root_ = std::move(node);
}

void JSONPrinter::print()
{
    std::vector<std::string> node_path = { "$root"};
    const auto json = to_json(root_, node_path);
    result_ = json.dump(4, ' ');
}

const std::string &JSONPrinter::get_result() const
{
    return result_;
}

nlohmann::ordered_json JSONPrinter::to_json(
    const RC<const Node> &node, std::vector<std::string> &node_path)
{
    nlohmann::ordered_json ret;

    if(auto it = node_to_path_.find(node); it != node_to_path_.end())
    {
        auto &path = it->second;
        ret = path;
        return ret;
    }
    node_to_path_.insert({ node, merge_path(node_path) });

    if(auto grp = node->as_group())
    {
        ret = nlohmann::ordered_json::object({});
        for(auto &key : grp->get_ordered_keys())
        {
            auto child = grp->find_child_node(key);
            assert(child);

            node_path.push_back(std::string(key));
            ret[std::string(key)] = to_json(child , node_path);
            node_path.pop_back();
        }
        return ret;
    }

    if(auto arr = node->as_array())
    {
        ret = nlohmann::ordered_json::array({});
        for(size_t i = 0; i < arr->get_size(); ++i)
        {
            auto elem = arr->get_element(i);
            assert(elem);

            node_path.push_back(std::to_string(i));
            ret.push_back(to_json(elem, node_path));
            node_path.pop_back();
        }
        return ret;
    }

    ret = node->as_value()->get_string();
    return ret;
}

std::string JSONPrinter::merge_path(const std::vector<std::string> &path) const
{
    std::string ret = "$reference{";
    for(size_t i = 0; i < path.size(); ++i)
    {
        if(i > 0)
            ret += "/";
        ret += path[i];
    }
    ret += "}";
    return ret;
}

BTRC_FACTORY_END
