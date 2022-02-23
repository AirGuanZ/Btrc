#include <btrc/core/renderer.h>
#include <btrc/utils/cuda/error.h>

BTRC_BEGIN

bool Renderer::is_waitable() const
{
    return waitable_;
}

bool Renderer::is_rendering() const
{
    return rendering_;
}

bool Renderer::should_stop() const
{
    return stop_;
}

void Renderer::render_async()
{
    CUcontext cuda_context;
    throw_on_error(cuCtxGetCurrent(&cuda_context));

    stop_ = false;
    rendering_ = true;
    async_future_ = std::async(
        std::launch::async, [this, cuda_context]
    {
        throw_on_error(cuCtxSetCurrent(cuda_context));
        BTRC_SCOPE_EXIT{ rendering_ = false; };
        return this->render();
    });
    waitable_ = true;
}

void Renderer::stop_async()
{
    if(waitable_)
    {
        stop_ = true;
        if(async_future_.valid())
            async_future_.wait();
        stop_ = false;
        waitable_ = false;
    }
}

Renderer::RenderResult Renderer::wait_async()
{
    assert(waitable_);
    BTRC_SCOPE_EXIT{ waitable_ = false; };
    return async_future_.get();
}

BTRC_END
