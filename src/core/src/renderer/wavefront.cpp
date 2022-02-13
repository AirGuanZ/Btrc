#include <btrc/core/renderer/wavefront/generate.h>
#include <btrc/core/renderer/wavefront/path_state.h>
#include <btrc/core/renderer/wavefront/shade.h>
#include <btrc/core/renderer/wavefront/shadow.h>
#include <btrc/core/renderer/wavefront/sort.h>
#include <btrc/core/renderer/wavefront/trace.h>
#include <btrc/core/renderer/wavefront.h>

BTRC_CORE_BEGIN

struct WavefrontPathTracer::Impl
{
    Params          params;
    RC<const Scene> scene;

    bool is_dirty = true;

    Film                 film;
    wf::PathState        path_state;
    wf::GeneratePipeline generate;
    wf::TracePipeline    trace;
    wf::SortPipeline     sort;
    wf::ShadePipeline    shade;
    wf::ShadowPipeline   shadow;
};

WavefrontPathTracer::WavefrontPathTracer()
{
    impl_ = newBox<Impl>();
}

WavefrontPathTracer::~WavefrontPathTracer()
{
    
}

void WavefrontPathTracer::set_params(const Params &params)
{
    impl_->params = params;
    impl_->is_dirty = true;
}

void WavefrontPathTracer::set_scene(RC<const Scene> scene)
{
    impl_->scene = scene;
    impl_->is_dirty = true;
}

// TODO

BTRC_CORE_END
