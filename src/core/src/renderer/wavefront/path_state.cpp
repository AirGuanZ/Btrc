#include <btrc/core/renderer/wavefront/path_state.h>

BTRC_WAVEFRONT_BEGIN

namespace
{

    template<typename...Ts>
    void init(int state_count, CUDABuffer<Ts> &...buffers)
    {
        ((buffers.initialize(state_count)), ...);
    }

} // namespace anonymous

void PathState::initialize(int state_count)
{
    {
        std::vector<cstd::LCGData> rng_init_data(state_count);
        for(int i = 0; i < state_count; ++i)
            rng_init_data[i].state = static_cast<uint32_t>(i + 1);
        rng.initialize(state_count, rng_init_data.data());
    }

    init(
        state_count,
        o_t0,
        d_t1,
        time_mask,
        beta_le,
        bsdf_pdf,
        beta,
        depth,
        pixel_coord,
        path_radiance,
        inct_t,
        inct_uv_id,
        active_state_indices,
        next_rng,
        next_beta,
        next_depth,
        next_pixel_coord,
        next_path_radiance,
        shadow_pixel_coord,
        shadow_o_t0,
        shadow_d_t1,
        shadow_time_mask,
        shadow_beta_li,
        next_o_t0, 
        next_d_t1,
        next_time_mask,
        next_beta_le,
        next_bsdf_pdf);
}

void PathState::next_iteration()
{
    rng.swap(next_rng);

    o_t0.swap(next_o_t0);
    d_t1.swap(next_d_t1);
    time_mask.swap(next_time_mask);
    pixel_coord.swap(next_pixel_coord);

    beta.swap(next_beta);
    beta_le.swap(next_beta_le);
    bsdf_pdf.swap(next_bsdf_pdf);
    depth.swap(next_depth);
    path_radiance.swap(next_path_radiance);
}

BTRC_WAVEFRONT_END
