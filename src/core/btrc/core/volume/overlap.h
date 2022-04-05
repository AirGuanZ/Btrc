#pragma once

#include <btrc/core/volume.h>

BTRC_BEGIN

namespace volume
{

    class VolumeOverlapResolver
    {
    public:

        void add_volume(RC<VolumePrimitive> vol);

        std::vector<std::set<RC<VolumePrimitive>>> get_overlaps() const;

    private:

        struct Record
        {
            AABB3f bbox;
            std::set<RC<VolumePrimitive>> vols;

            auto operator<=>(const Record &rhs) const { return vols <=> rhs.vols; }

            auto operator==(const Record &rhs) const { return vols == rhs.vols; }
        };

        std::vector<std::set<Record>> size_to_overlap_;
    };

} // namespace volume

BTRC_END
