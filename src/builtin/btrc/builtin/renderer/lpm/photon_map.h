#pragma once

#include <btrc/builtin/renderer/lpm/common.h>
#include <btrc/core/spectrum.h>

BTRC_LPM_BEGIN

namespace photon_map_detail
{

    struct PhotonRecord
    {
        Vec3f    pos;
        Vec3f    wr;
        Spectrum partial_beta;
        float    partial_pdf;
        float    dVMT;
        float    dVM;
        uint32_t next;
    };

    CUJ_PROXY_CLASS(
        CPhotonRecord, PhotonRecord,
        pos, wr, partial_beta, partial_pdf, dVMT, dVM, next);

} // namespace photon_map_detail

class PhotonMap : public Uncopyable
{
public:

    using PhotonRecord = photon_map_detail::PhotonRecord;
    using CPhotonRecord = photon_map_detail::CPhotonRecord;
    using QueryCallback = std::function<void(ref<CPhotonRecord>)>;

    PhotonMap(
        const Vec3f &world_lower,
        const Vec3f &world_upper,
        const Vec3i &res,
        uint32_t     entry_count,
        uint32_t     max_record_count);

    void clear();

    void add_photon(const CPhotonRecord &photon);

    void query_photons(const CVec3f &pos, f32 radius, const QueryCallback &func) const;

private:

    static constexpr uint32_t INVALID_PHOTON = std::numeric_limits<uint32_t>::max();

    CVec3i pos_to_grid(const CVec3f &pos) const;

    u32 grid_to_entry(const CVec3i &grid) const;

    u32 pos_to_entry(const CVec3f &pos) const;

    Vec3f lower_;
    Vec3f upper_;
    Vec3i res_;

    std::vector<uint32_t> invalid_hash_entries_data_;

    cuda::Buffer<uint32_t>     photon_record_counter_;
    cuda::Buffer<uint32_t>     hash_entries_;
    cuda::Buffer<PhotonRecord> photon_records_;
};

BTRC_LPM_END
