#include <algorithm>

#include <btrc/builtin/volume/indexing.h>

BTRC_BUILTIN_BEGIN

namespace
{

    struct Node
    {
        int overlap_index = -2;
        std::map<RC<VolumePrimitive>, Box<Node>> children;
    };

    // returns index where node begins
    int32_t fill_indices(
        const Node *node, std::vector<int32_t> &indices,
        const std::vector<RC<VolumePrimitive>> &all_vols)
    {
        const int32_t result = static_cast<int32_t>(indices.size());
        indices.resize(indices.size() + 1 + all_vols.size(), -1);
        indices[result] = node->overlap_index;
        for(size_t i = 0; i < all_vols.size(); ++i)
        {
            auto &vol = all_vols[i];
            if(auto it = node->children.find(vol); it != node->children.end())
                indices[result + i + 1] = fill_indices(it->second.get(), indices, all_vols);
        }
        return result;
    }

} // namespace anonymous

volume::OverlapIndexer::OverlapIndexer(
    const std::vector<std::set<RC<VolumePrimitive>>> &overlaps,
    const std::vector<RC<VolumePrimitive>> &all_vols,
    const std::map<RC<VolumePrimitive>, int> &vol_to_id)
{
    // build trie

    Node root_node = { -1, {} };
    for(size_t i = 0; i < overlaps.size(); ++i)
    {
        std::vector<RC<VolumePrimitive>> vols{ overlaps[i].begin(), overlaps[i].end() };
        std::sort(vols.begin(), vols.end(), [&](const auto &l, const auto &r)
        {
            return vol_to_id.at(l) < vol_to_id.at(r);
        });

        Node *node = &root_node;
        for(auto &vol : vols)
        {
            auto it = node->children.find(vol);
            if(it == node->children.end())
                it = node->children.insert({ vol, newBox<Node>() }).first;
            node = it->second.get();
        }
        assert(node->overlap_index == -2);
        node->overlap_index = static_cast<int>(i);
    }

    // create indices

    fill_indices(&root_node, indices_, all_vols);
}

const std::vector<int32_t> &volume::OverlapIndexer::get_indices() const
{
    return indices_;
}

BTRC_BUILTIN_END
