#pragma once

#include <btrc/core/context.h>

BTRC_BEGIN

std::vector<RC<Object>> topology_sort_object_tree(const std::set<RC<Object>> &entries);

inline std::vector<RC<Object>> topology_sort_object_tree(const std::vector<RC<Object>> &entries)
{
    return topology_sort_object_tree(std::set(entries.begin(), entries.end()));
}

BTRC_END
