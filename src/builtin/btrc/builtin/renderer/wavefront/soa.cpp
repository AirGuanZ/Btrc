#include <btrc/builtin/renderer/wavefront/soa.h>

BTRC_WFPT_BEGIN

void RaySOA::initialize(int state_count)
{
    o_med_id_.initialize(state_count);
    d_t1_.initialize(state_count);
}

void RaySOA::save(i32 index, const CRay &r, CMediumID medium_id)
{
    save_aligned(CVec4f(r.o, cuj::bitcast<f32>(medium_id)), o_med_id_.get_cuj_ptr() + index);
    save_aligned(CVec4f(r.d, r.t), d_t1_.get_cuj_ptr() + index);
}

RaySOA::LoadResult RaySOA::load(i32 index) const
{
    var o_med_id = load_aligned(o_med_id_.get_cuj_ptr() + index);
    var o = o_med_id.xyz();
    var med_id = cuj::bitcast<CMediumID>(o_med_id.w);

    var d_t1 = load_aligned(d_t1_.get_cuj_ptr() + index);
    var d = d_t1.xyz();
    var t1 = d_t1.w;

    return LoadResult{ CRay(o, d, t1), med_id };
}

BTRC_WFPT_END
