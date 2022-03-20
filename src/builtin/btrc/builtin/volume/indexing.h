#pragma once

#include <btrc/core/volume.h>

BTRC_BUILTIN_BEGIN

namespace volume
{

    class OverlapIndexer
    {
    public:

        OverlapIndexer(
            const std::vector<std::set<RC<VolumePrimitive>>> &overlaps,
            const std::vector<RC<VolumePrimitive>> &all_vols,
            const std::map<RC<VolumePrimitive>, int> &vol_to_id);

        const std::vector<int32_t> &get_indices() const;

    private:

        std::vector<int32_t> indices_;
    };

} // namespace volume

BTRC_BUILTIN_END
