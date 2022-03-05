#include <random>

#include "./path_state.h"

BTRC_WFPT_BEGIN

namespace
{

    template<typename...Ts>
    void init(int state_count, cuda::Buffer<Ts> &...buffers)
    {
        ((buffers.initialize(state_count)), ...);
    }

} // namespace anonymous

void PathState::initialize(int state_count)
{
    init(
        state_count,
        rng,
        o_t0,
        d_t1,
        time_mask,
        medium_id,
        beta_le,
        bsdf_pdf,
        beta,
        depth,
        pixel_coord,
        path_radiance,
        inct_inst_launch_index,
        inct_t_prim_uv,
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
        shadow_medium_id,
        next_o_t0, 
        next_d_t1,
        next_time_mask,
        next_medium_id,
        next_beta_le,
        next_bsdf_pdf);
}

void PathState::clear()
{
    const int state_count = static_cast<int>(rng.get_size());
    std::vector<RNG::Data> rng_init_data(state_count);
    for(int i = 0; i < state_count; ++i)
        rng_init_data[i] = RNG::Data(static_cast<uint32_t>(i));

    std::default_random_engine random_engine{ 42 };
    std::shuffle(rng_init_data.begin(), rng_init_data.end(), random_engine);
    rng.from_cpu(rng_init_data.data());
    next_rng.from_cpu(rng_init_data.data());
}

void PathState::next_iteration()
{
    rng.swap(next_rng);

    o_t0.swap(next_o_t0);
    d_t1.swap(next_d_t1);
    time_mask.swap(next_time_mask);
    medium_id.swap(next_medium_id);
    pixel_coord.swap(next_pixel_coord);

    beta.swap(next_beta);
    beta_le.swap(next_beta_le);
    bsdf_pdf.swap(next_bsdf_pdf);
    depth.swap(next_depth);
    path_radiance.swap(next_path_radiance);
}

BTRC_WFPT_END
