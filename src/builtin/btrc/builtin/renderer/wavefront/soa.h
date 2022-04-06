#pragma once

#include <btrc/builtin/renderer/wavefront/common.h>
#include <btrc/core/medium.h>

BTRC_WFPT_BEGIN

struct RaySOA
{
    Vec4f *o_med_id_buffer;
    Vec4f *d_t1_buffer;
};

struct BSDFLeSOA
{
    Vec4f *beta_le_bsdf_pdf_buffer;
};

struct PathSOA
{
    Vec2u                *pixel_coord_buffer;
    Vec4f                *beta_depth_buffer;
    Vec4f                *path_radiance_buffer;
    GlobalSampler::State *sampler_state_buffer;
};

struct IntersectionSOA
{
    uint32_t *path_flag_buffer;
    Vec4u    *t_prim_id_buffer;
};

struct ShadowRaySOA
{
    Vec2u *pixel_coord_buffer;
    Vec4f *beta_li_buffer;
    RaySOA ray;
};

CUJ_PROXY_CLASS_EX(CRaySOA, RaySOA, o_med_id_buffer, d_t1_buffer)
{
    CUJ_BASE_CONSTRUCTORS

    struct LoadResult
    {
        CRay ray;
        CMediumID medium_id;
    };

    void save(i32 index, const CRay &r, CMediumID medium_id);

    LoadResult load(i32 index) const;
};

CUJ_PROXY_CLASS_EX(CBSDFLeSOA, BSDFLeSOA, beta_le_bsdf_pdf_buffer)
{
    CUJ_BASE_CONSTRUCTORS

    struct LoadResult
    {
        CSpectrum beta_le;
        f32 bsdf_pdf;
    };

    void save(i32 index, const CSpectrum &beta_le, f32 bsdf_pdf);

    LoadResult load(i32 index) const;
};

CUJ_PROXY_CLASS_EX(
    CPathSOA, PathSOA,
    pixel_coord_buffer,
    beta_depth_buffer,
    path_radiance_buffer,
    sampler_state_buffer)
{
    CUJ_BASE_CONSTRUCTORS

    struct LoadResult
    {
        i32                   depth;
        CVec2u                pixel_coord;
        CSpectrum             beta;
        CSpectrum             path_radiance;
        GlobalSampler::CState sampler_state;
    };

    void save(
        i32                  index,
        i32                  depth,
        const CVec2u        &pixel_coord,
        const CSpectrum     &beta,
        const CSpectrum     &path_radiance,
        const GlobalSampler &sampler);

    void save_sampler(i32 index, const GlobalSampler &sampler);

    LoadResult load(i32 index) const;
};

CUJ_PROXY_CLASS_EX(
    CIntersectionSOA, IntersectionSOA,
    path_flag_buffer, t_prim_id_buffer)
{
    CUJ_BASE_CONSTRUCTORS

    struct LoadFlagResult
    {
        boolean is_intersected;
        boolean is_scattered;
        u32     instance_id;
    };

    struct LoadDetailResult
    {
        f32    t;
        u32    prim_id;
        CVec2f uv;
    };

    void save_flag(i32 index, boolean is_intersected, boolean is_scattered, u32 instance_id);

    void save_detail(i32 index, f32 t, u32 prim_id, const CVec2f &uv);

    LoadFlagResult load_flag(i32 index) const;

    LoadDetailResult load_detail(i32 index) const;
};

CUJ_PROXY_CLASS_EX(CShadowRaySOA, ShadowRaySOA, pixel_coord_buffer, beta_li_buffer, ray)
{
    CUJ_BASE_CONSTRUCTORS

    struct LoadBetaResult
    {
        CVec2u    pixel_coord;
        CSpectrum beta_li;
    };

    void save(i32 index, const CVec2u &pixel_coord, const CSpectrum &beta_li, const CRay &r, CMediumID medium_id);

    CRaySOA::LoadResult load_ray(i32 index) const;

    LoadBetaResult load_beta(i32 index) const;
};

BTRC_WFPT_END
