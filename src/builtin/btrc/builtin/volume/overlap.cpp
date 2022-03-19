#include <btrc/builtin/volume/overlap.h>

BTRC_BUILTIN_BEGIN

namespace volume
{

    void VolumeOverlapResolver::add_volume(RC<VolumePrimitive> vol)
    {
        AABB3f bbox = vol->get_bounding_box();
        const Vec3f extent = bbox.upper - bbox.lower;
        const Vec3f relax_factor = 0.05f * extent;
        bbox.lower = bbox.lower - relax_factor;
        bbox.upper = bbox.upper + relax_factor;

        size_to_overlap_.emplace_back();
        for(int old_size = static_cast<int>(size_to_overlap_.size()) - 2; old_size >= 0; --old_size)
        {
            auto &old_overlaps = size_to_overlap_[old_size];
            auto &new_overlaps = size_to_overlap_[old_size + 1];
            for(auto &old : old_overlaps)
            {
                auto new_bbox = intersect_aabb(old.bbox, bbox);
                if(!new_bbox.empty())
                {
                    auto new_vols = old.vols;
                    new_vols.insert(vol);
                    new_overlaps.insert(Record{
                        .bbox = new_bbox,
                        .vols = std::move(new_vols)
                    });
                }
            }
        }

        size_to_overlap_[0].insert(Record{
            .bbox = bbox,
            .vols = { std::move(vol) }
        });
    }

    std::vector<std::set<RC<VolumePrimitive>>> VolumeOverlapResolver::get_overlaps() const
    {
        std::vector<std::set<RC<VolumePrimitive>>> result;
        for(auto &overlap_set : size_to_overlap_)
        {
            for(auto &overlap : overlap_set)
                result.push_back(overlap.vols);
        }
        return result;
    }

} // namespace volume

BTRC_BUILTIN_END
