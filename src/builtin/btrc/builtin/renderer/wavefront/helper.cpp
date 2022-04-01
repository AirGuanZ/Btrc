#include <btrc/builtin/renderer/wavefront/helper.h>

BTRC_WFPT_BEGIN

boolean simple_russian_roulette(
    ref<CSpectrum>                     path_beta,
    i32                                depth,
    GlobalSampler                     &sampler,
    const SimpleRussianRouletteParams &params)
{
    boolean ret = false;
    $if(depth >= params.min_depth)
    {
        $if(depth >= params.max_depth)
        {
            ret = true;
        }
        $else
        {
            var sam = sampler.get1d();
            $if(path_beta.get_lum() < params.beta_threshold)
            {
                $if(sam > params.cont_prob)
                {
                    ret = true;
                }
                $else
                {
                    path_beta = path_beta / params.cont_prob;
                };
            };
        };
    };
    return ret;
}

CSpectrum estimate_medium_tr(
    CompileContext      &cc,
    const Scene         &scene,
    const VolumeManager &vols,
    CMediumID            medium_id,
    const CVec3f        &a,
    const CVec3f        &b,
    GlobalSampler       &sampler)
{
    CSpectrum tr;
    $switch(medium_id)
    {
        for(int i = 0; i < scene.get_medium_count(); ++i)
        {
            $case(i)
            {
                tr = scene.get_medium(i)->tr(cc, a, b, a, b, sampler);
            };
        }
        $case(MEDIUM_ID_VOID)
        {
            tr = vols.tr(cc, a, b, sampler);
        };
        $default
        {
            cstd::unreachable();
        };
    };
    return tr;
}

BTRC_WFPT_END
