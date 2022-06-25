#include <btrc/builtin/renderer/pt/pm.h>

BTRC_PT_BEGIN

PhotonMap::PhotonMap(
    const Vec3f &world_lower,
    const Vec3f &world_upper,
    const Vec3i &grid_res,
    uint32_t     entry_count,
    uint32_t     max_record_count)
    : lower_(world_lower), upper_(world_upper), res_(grid_res)
{
    photon_record_counter_.initialize(1);
    hash_entries_.initialize(entry_count);
    photon_records_.initialize(max_record_count);

    clear();
}

void PhotonMap::clear()
{
    photon_record_counter_.clear_bytes(0);
    hash_entries_.clear(INVALID_PHOTON);
}

void PhotonMap::add_photon(const CPhoton &photon)
{
    var all_records = cuj::import_pointer(photon_records_.get());
    var record_counter = cuj::import_pointer(photon_record_counter_.get());
    var record_index = cstd::atomic_add(record_counter, 1);
    CUJ_ASSERT(record_index < static_cast<uint32_t>(photon_records_.get_size()));

    var record = all_records + record_index;
    *record = photon;

    var entry_index = pos_to_entry(photon.pos);
    var entry_ptr = cuj::import_pointer(hash_entries_.get()) + entry_index;

    var next = INVALID_PHOTON - 1;
    $loop
    {
        // the first try must fail
        var actual_entry = cstd::atomic_cmpxchg(entry_ptr, next, record_index);
        $if(actual_entry == next)
        {
            $break;
        };
        next = actual_entry;
    };
    record->next = next;
}

void PhotonMap::query_photons(const CVec3f &pos, f32 radius, const QueryCallback &callback)
{
    var lower_grid = pos_to_grid(pos - CVec3f(radius));
    var upper_grid = pos_to_grid(pos + CVec3f(radius)) + 1;
    $forrange(xi, lower_grid.x, upper_grid.x)
    {
        $forrange(yi, lower_grid.y, upper_grid.y)
        {
            $forrange(zi, lower_grid.z, upper_grid.z)
            {
                var entry_index = grid_to_entry(CVec3i(xi, yi, zi));
                var record_index = cuj::import_pointer(hash_entries_.get())[entry_index];
                $while(record_index != INVALID_PHOTON)
                {
                    ref record = cuj::import_pointer(photon_records_.get())[record_index];
                    $if(length_square(pos - record.pos) <= radius * radius)
                    {
                        callback(record);
                    };
                    record_index = record.next;
                };
            };
        };
    };
}

CVec3i PhotonMap::pos_to_grid(const CVec3f &pos) const
{
    const Vec3f grid_size = {
        (upper_.x - lower_.x) / res_.x,
        (upper_.y - lower_.y) / res_.y,
        (upper_.z - lower_.z) / res_.z,
    };
    return CVec3i(
        i32((pos.x - lower_.x) * (1.0f / grid_size.x)),
        i32((pos.y - lower_.y) * (1.0f / grid_size.y)),
        i32((pos.z - lower_.z) * (1.0f / grid_size.z)));
}

u32 PhotonMap::grid_to_entry(const CVec3i &grid) const
{
    var v = u32((grid.x * 73856093) ^ (grid.y * 19349663) ^ (grid.z * 83492791));
    return v % static_cast<unsigned>(hash_entries_.get_size());
}

u32 PhotonMap::pos_to_entry(const CVec3f &pos) const
{
    return grid_to_entry(pos_to_grid(pos));
}

BTRC_PT_END
