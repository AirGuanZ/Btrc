#pragma once

#include <btrc/core/volume.h>

BTRC_BUILTIN_BEGIN

namespace volume
{

    class OverlapIndexer
    {
    public:

        explicit OverlapIndexer(const std::vector<std::set<RC<VolumePrimitive>>> &overlaps);

        const std::vector<int32_t> &get_indices() const;

    private:

        std::vector<int32_t> indices_;
    };

} // namespace volume

BTRC_BUILTIN_END
