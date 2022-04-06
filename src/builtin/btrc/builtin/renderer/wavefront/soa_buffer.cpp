#include <random>

#include <btrc/builtin/renderer/wavefront/soa_buffer.h>

BTRC_WFPT_BEGIN

RayBuffer::RayBuffer(int state_count)
{
    o_medium_id_.initialize(state_count);
    d_t1_.initialize(state_count);
}

RayBuffer::operator RaySOA()
{
    return RaySOA{
        .o_med_id_buffer = o_medium_id_,
        .d_t1_buffer = d_t1_
    };
}

BSDFLeBuffer::BSDFLeBuffer(int state_count)
{
    beta_le_bsdf_pdf_.initialize(state_count);
}

BSDFLeBuffer::operator BSDFLeSOA()
{
    return BSDFLeSOA{
        .beta_le_bsdf_pdf_buffer = beta_le_bsdf_pdf_
    };
}

PathBuffer::PathBuffer(int state_count)
{
    pixel_coord_.initialize(state_count);
    beta_depth_.initialize(state_count);
    path_radiance_.initialize(state_count);
    sampler_state_.initialize(state_count);
}

PathBuffer::operator PathSOA()
{
    return PathSOA{
        .pixel_coord_buffer = pixel_coord_,
        .beta_depth_buffer = beta_depth_,
        .path_radiance_buffer = path_radiance_,
        .sampler_state_buffer = sampler_state_
    };
}

IntersectionBuffer::IntersectionBuffer(int state_count)
{
    path_flag_.initialize(state_count);
    t_prim_uv_.initialize(state_count);
}

IntersectionBuffer::operator IntersectionSOA()
{
    return IntersectionSOA{
        .path_flag_buffer = path_flag_,
        .t_prim_id_buffer = t_prim_uv_
    };
}

ShadowRayBuffer::ShadowRayBuffer(int state_count)
{
    pixel_coord_.initialize(state_count);
    beta_li_.initialize(state_count);
    ray_ = newRC<RayBuffer>(state_count);
}

ShadowRayBuffer::operator ShadowRaySOA()
{
    return ShadowRaySOA{
        .pixel_coord_buffer = pixel_coord_,
        .beta_li_buffer = beta_li_,
        .ray = *ray_
    };
}

ShadowSamplerBuffer::ShadowSamplerBuffer(int state_count)
{
    buffer_.initialize(state_count);
}

void ShadowSamplerBuffer::clear()
{
    const int state_count = static_cast<int>(buffer_.get_size());
    std::vector<IndependentSampler::State> rng_init_data(state_count);
    for(int i = 0; i < state_count; ++i)
        rng_init_data[i].rng = cstd::PCG::Data(static_cast<uint32_t>(i));

    std::default_random_engine random_engine{ 42 };
    std::shuffle(rng_init_data.begin(), rng_init_data.end(), random_engine);
    buffer_.from_cpu(rng_init_data.data());
}

ShadowSamplerBuffer::operator independent_sampler_detail::State*()
{
    return buffer_;
}

BTRC_WFPT_END
