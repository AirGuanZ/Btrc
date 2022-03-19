#include <btrc/builtin/volume/indexing.h>

BTRC_BUILTIN_BEGIN

namespace
{

    struct Node
    {
        int overlap_index;
        std::map<RC<VolumePrimitive>, Box<Node>> children;
    };

    // returns index where node begins
    int32_t fill_indices(const Node *node, std::vector<int32_t> &indices)
    {
        const int32_t result = static_cast<int32_t>(indices.size());
        indices.resize(indices.size() + 1 + node->children.size(), -1);
        indices[result] = node->overlap_index;
        size_t index_offset = 1;
        for(auto &child_it : node->children)
        {
            const auto child_index = fill_indices(child_it.second.get(), indices);
            indices[result + index_offset++] = child_index;
        }
        return result;
    }

} // namespace anonymous

volume::OverlapIndexer::OverlapIndexer(const std::vector<std::set<RC<VolumePrimitive>>> &overlaps)
{
    // build trie

    Node root_node = { -1, {} };
    for(size_t i = 0; i < overlaps.size(); ++i)
    {
        Node *node = &root_node;
        for(auto &vol : overlaps[i])
        {
            auto it = node->children.find(vol);
            if(it == node->children.end())
                it = node->children.insert({ vol, newBox<Node>() }).first;
            node = it->second.get();
        }
        node->overlap_index = static_cast<int>(i);
    }

    // create indices

    fill_indices(&root_node, indices_);
}

const std::vector<int32_t> &volume::OverlapIndexer::get_indices() const
{
    return indices_;
}

BTRC_BUILTIN_END
