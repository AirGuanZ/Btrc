#pragma once

#include <filesystem>
#include <set>

#include <btrc/factory/node/node.h>

BTRC_FACTORY_BEGIN

class JSONParser
{
public:

    void set_source(std::string src);

    void add_include_directory(const std::filesystem::path &dir);

    void parse();

    RC<Group> get_result();

private:

    std::filesystem::path get_absolute_included_file_path(
        const std::filesystem::path &included_file) const;

    void process_includes();

    void resolve_references(std::vector<RC<Node>> &current_path, RC<Node> &node);

    RC<Node> find_node(std::vector<RC<Node>> &current_path, const std::string &path);

    std::string src_;
    std::set<std::filesystem::path> include_dirs_;
    RC<Group> result_;
};

BTRC_FACTORY_END
