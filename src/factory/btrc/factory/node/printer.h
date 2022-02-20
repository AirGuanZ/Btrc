#pragma once

#include <nlohmann/json.hpp>

#include <btrc/factory/node/node.h>

BTRC_FACTORY_BEGIN

class JSONPrinter
{
public:

    void set_root_node(RC<const Node> node);

    void print();

    const std::string &get_result() const;

private:

    nlohmann::ordered_json to_json(
        const RC<const Node> &node, std::vector<std::string> &node_path);

    std::string merge_path(const std::vector<std::string> &path) const;

    RC<const Node> root_;
    std::string result_;

    std::map<RC<const Node>, std::string> node_to_path_;
};

BTRC_FACTORY_END
