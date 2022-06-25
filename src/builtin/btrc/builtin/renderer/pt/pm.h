#pragma once

#include <btrc/builtin/renderer/pt/common.h>
#include <btrc/core/scene.h>
#include <btrc/utils/optix/pipeline_mk.h>

BTRC_PT_BEGIN

namespace photon_map_impl
{

    struct Photon
    {
        Vec3f pos;
        Vec3f wr;

        Spectrum beta;
        float pdf;

        // |cos<n_{x_{c+1}}, x_cx_{c+1}>| / |x_c - x_{c+1}|^2
        // used to convert pdf_sa(x_c -> x_{c+1}) to pdf_area(x_c -> x_{c+1})
        float pdf_factor_xc_r;
        // pdf_light(y1|y2)
        float pdf_light;
        // pdf(y1)
        float pdf_y1;
        // pdf(y2 -> y1)
        float pdf_y2_r;
        // pdf(y2 <- y1)
        float pdf_y2_l;

        float dC;

        uint32_t next;
    };

    CUJ_PROXY_CLASS(
        CPhoton, Photon,
        pos,
        wr,
        beta,
        pdf,
        pdf_factor_xc_r,
        pdf_light,
        pdf_y1,
        pdf_y2_r,
        pdf_y2_l,
        dC,
        next);

} // namespace photon_map_impl

class PhotonMap
{
public:

    using Photon = photon_map_impl::Photon;
    using CPhoton = photon_map_impl::CPhoton;

    using QueryCallback = std::function<void(ref<CPhoton>)>;

    PhotonMap(
        const Vec3f &world_lower,
        const Vec3f &world_upper,
        const Vec3i &grid_res,
        uint32_t     entry_count,
        uint32_t     max_record_count);

    void clear();

    void add_photon(const CPhoton &photon);

    void query_photons(const CVec3f &pos, f32 radius, const QueryCallback &callback);

private:

    static constexpr uint32_t INVALID_PHOTON = std::numeric_limits<uint32_t>::max();

    CVec3i pos_to_grid(const CVec3f &pos) const;

    u32 grid_to_entry(const CVec3i &grid) const;

    u32 pos_to_entry(const CVec3f &pos) const;

    Vec3f lower_;
    Vec3f upper_;
    Vec3i res_;

    cuda::Buffer<uint32_t> photon_record_counter_;
    cuda::Buffer<uint32_t> hash_entries_;
    cuda::Buffer<Photon>   photon_records_;
};

BTRC_PT_END
