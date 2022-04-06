#pragma once

#include <btrc/builtin/renderer/wavefront/soa.h>

BTRC_WFPT_BEGIN

class RayBuffer : public Uncopyable
{
public:

    explicit RayBuffer(int state_count);

    operator RaySOA();

private:

    cuda::Buffer<Vec4f> o_medium_id_;
    cuda::Buffer<Vec4f> d_t1_;
};

class BSDFLeBuffer : public Uncopyable
{
public:

    explicit BSDFLeBuffer(int state_count);

    operator BSDFLeSOA();

private:

    cuda::Buffer<Vec4f> beta_le_bsdf_pdf_;
};

class PathBuffer : public Uncopyable
{
public:

    explicit PathBuffer(int state_count);

    operator PathSOA();

private:

    cuda::Buffer<Vec2u>                pixel_coord_;
    cuda::Buffer<Vec4f>                beta_depth_;
    cuda::Buffer<Vec4f>                path_radiance_;
    cuda::Buffer<GlobalSampler::State> sampler_state_;
};

class IntersectionBuffer : public Uncopyable
{
public:

    explicit IntersectionBuffer(int state_count);

    operator IntersectionSOA();

private:

    cuda::Buffer<uint32_t> path_flag_;
    cuda::Buffer<Vec4u>    t_prim_uv_;
};

class ShadowRayBuffer : public Uncopyable
{
public:

    explicit ShadowRayBuffer(int state_count);

    operator ShadowRaySOA();

private:

    cuda::Buffer<Vec2u> pixel_coord_;
    cuda::Buffer<Vec4f> beta_li_;
    RC<RayBuffer> ray_;
};

class ShadowSamplerBuffer : public Uncopyable
{
public:

    explicit ShadowSamplerBuffer(int state_count);

    void clear();

    operator IndependentSampler::State*();

private:

    cuda::Buffer<IndependentSampler::State> buffer_;
};

BTRC_WFPT_END
