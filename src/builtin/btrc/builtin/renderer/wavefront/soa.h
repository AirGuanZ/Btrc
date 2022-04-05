#pragma once

#include <btrc/builtin/renderer/wavefront/common.h>
#include <btrc/core/medium.h>

BTRC_WFPT_BEGIN

class RaySOA : public Uncopyable
{
    cuda::Buffer<Vec4f> o_med_id_;
    cuda::Buffer<Vec4f> d_t1_;

public:

    struct LoadResult
    {
        CRay ray;
        CMediumID medium_id;
    };

    void initialize(int state_count);

    void save(i32 index, const CRay &r, CMediumID medium_id);

    LoadResult load(i32 index) const;
};

template<typename AdditionalData = int32_t>
    requires (sizeof(AdditionalData) == 4)
class SpectrumSOA
{
    cuda::Buffer<Vec4f> buffer_;

public:

    struct LoadResult
    {
        CSpectrum spectrum;
        cuj::cxx<AdditionalData> additional_data;
    };

    void initialize(int state_count);

    void save(i32 index, const CSpectrum &spec, cuj::cxx<AdditionalData> additional_data = 0);

    LoadResult load(i32 index) const;
};

// ========================== impl ==========================

template<typename AdditionalData> requires (sizeof(AdditionalData) == 4)
void SpectrumSOA<AdditionalData>::initialize(int state_count)
{
    buffer_.initialize(state_count);
}

template<typename AdditionalData> requires (sizeof(AdditionalData) == 4)
void SpectrumSOA<AdditionalData>::save(i32 index, const CSpectrum &spec, cuj::cxx<AdditionalData> additional_data)
{
    save_aligned(CVec4f(spec.r, spec.g, spec.b, cuj::bitcast<f32>(additional_data)), buffer_.get_cuj_ptr() + index);
}

template<typename AdditionalData> requires (sizeof(AdditionalData) == 4)
typename SpectrumSOA<AdditionalData>::LoadResult SpectrumSOA<AdditionalData>::load(i32 index) const
{
    var v4 = load_aligned(buffer_.get_cuj_ptr() + index);
    return LoadResult{
        CSpectrum::from_rgb(v4.x, v4.y, v4.z),
        cuj::bitcast<cuj::cxx<AdditionalData>>(v4.w)
    };
}

BTRC_WFPT_END
