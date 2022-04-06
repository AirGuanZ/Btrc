#include <btrc/builtin/renderer/wavefront/soa.h>

BTRC_WFPT_BEGIN

void CRaySOA::save(i32 index, const CRay &r, CMediumID medium_id)
{
    save_aligned(CVec4f(r.o, cuj::bitcast<f32>(medium_id)), o_med_id_buffer + index);
    save_aligned(CVec4f(r.d, r.t), d_t1_buffer + index);
}

CRaySOA::LoadResult CRaySOA::load(i32 index) const
{
    var o_med_id = load_aligned(o_med_id_buffer + index);
    var o = o_med_id.xyz();
    var med_id = cuj::bitcast<CMediumID>(o_med_id.w);

    var d_t1 = load_aligned(d_t1_buffer + index);
    var d = d_t1.xyz();
    var t1 = d_t1.w;

    return LoadResult{ CRay(o, d, t1), med_id };
}

void CBSDFLeSOA::save(i32 index, const CSpectrum &beta_le, f32 bsdf_pdf)
{
    save_aligned(CVec4f(beta_le.r, beta_le.g, beta_le.b, bsdf_pdf), beta_le_bsdf_pdf_buffer + index);
}

CBSDFLeSOA::LoadResult CBSDFLeSOA::load(i32 index) const
{
    var v = load_aligned(beta_le_bsdf_pdf_buffer + index);
    return LoadResult{ CSpectrum::from_rgb(v.x, v.y, v.z), v.w };
}

void CPathSOA::save(
    i32                  index,
    i32                  depth,
    const CVec2u        &pixel_coord,
    const CSpectrum     &beta,
    const CSpectrum     &path_radiance,
    const GlobalSampler &sampler)
{
    save_aligned(pixel_coord, pixel_coord_buffer + index);
    save_aligned(CVec4f(beta.r, beta.g, beta.b, cuj::bitcast<f32>(depth)), beta_depth_buffer + index);
    save_aligned(CVec4f(path_radiance.r, path_radiance.g, path_radiance.b, 1), path_radiance_buffer + index);
    save_sampler(index, sampler);
}

void CPathSOA::save_sampler(i32 index, const GlobalSampler &sampler)
{
    sampler.save(sampler_state_buffer + index);
}

CPathSOA::LoadResult CPathSOA::load(i32 index) const
{
    var beta_depth = load_aligned(beta_depth_buffer + index);
    var path_rad = load_aligned(path_radiance_buffer + index);

    LoadResult result;
    result.depth = cuj::bitcast<i32>(beta_depth.w);
    result.pixel_coord = load_aligned(pixel_coord_buffer + index);
    result.beta = CSpectrum::from_rgb(beta_depth.x, beta_depth.y, beta_depth.z);
    result.path_radiance = CSpectrum::from_rgb(path_rad.x, path_rad.y, path_rad.z);
    result.sampler_state = sampler_state_buffer[index];
    return result;
}

void CIntersectionSOA::save_flag(i32 index, boolean is_intersected, boolean is_scattered, u32 instance_id)
{
    u32 flag = instance_id;
    $if(is_intersected)
    {
        flag = flag | PATH_FLAG_HAS_INTERSECTION;
    };
    $if(is_scattered)
    {
        flag = flag | PATH_FLAG_HAS_SCATTERING;
    };
    path_flag_buffer[index] = flag;
}

void CIntersectionSOA::save_detail(i32 index, f32 t, u32 prim_id, const CVec2f &uv)
{
    save_aligned(
        CVec4u(
            cuj::bitcast<u32>(t),
            prim_id,
            cuj::bitcast<u32>(uv.x),
            cuj::bitcast<u32>(uv.y)),
        t_prim_id_buffer + index);
}

CIntersectionSOA::LoadFlagResult CIntersectionSOA::load_flag(i32 index) const
{
    u32 flag = path_flag_buffer[index];
    LoadFlagResult result;
    result.is_intersected = is_path_intersected(flag);
    result.is_scattered = is_path_scattered(flag);
    result.instance_id = extract_instance_id(flag);
    return result;
}

CIntersectionSOA::LoadDetailResult CIntersectionSOA::load_detail(i32 index) const
{
    CVec4u v = load_aligned(t_prim_id_buffer + index);
    LoadDetailResult result;
    result.t = cuj::bitcast<f32>(v.x);
    result.prim_id = v.y;
    result.uv.x = cuj::bitcast<f32>(v.z);
    result.uv.y = cuj::bitcast<f32>(v.w);
    return result;
}

void CShadowRaySOA::save(i32 index, const CVec2u &pixel_coord, const CSpectrum &beta_li, const CRay &r, CMediumID medium_id)
{
    save_aligned(pixel_coord, pixel_coord_buffer + index);
    save_aligned(CVec4f(beta_li.r, beta_li.g, beta_li.b, 1), beta_li_buffer + index);
    ray.save(index, r, medium_id);
}

CRaySOA::LoadResult CShadowRaySOA::load_ray(i32 index) const
{
    return ray.load(index);
}

CShadowRaySOA::LoadBetaResult CShadowRaySOA::load_beta(i32 index) const
{
    var v = load_aligned(beta_li_buffer + index);
    LoadBetaResult result;
    result.pixel_coord = load_aligned(pixel_coord_buffer + index);
    result.beta_li = CSpectrum::from_rgb(v.x, v.y, v.z);
    return result;
}

BTRC_WFPT_END
