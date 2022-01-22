#include <btrc/core/utils/cmath/cmath.h>
#include <btrc/core/wavefront/generate.h>

BTRC_WAVEFRONT_BEGIN

namespace
{

    const char *GENERATE_KERNEL_NAME = "generate";

} // namespace anonymous

GeneratePipeline::GeneratePipeline()
    : pixel_count_(0), spp_(0), state_count_(0),
      finished_spp_(0), finished_pixel_(0)
{

}

GeneratePipeline::GeneratePipeline(
    const Camera &camera, const Vec2i &film_res, int spp, int state_count)
    : GeneratePipeline()
{
    initialize(camera, film_res, spp, state_count);
}

GeneratePipeline::GeneratePipeline(GeneratePipeline &&other) noexcept
    : GeneratePipeline()
{
    swap(other);
}

GeneratePipeline &GeneratePipeline::operator=(GeneratePipeline &&other) noexcept
{
    swap(other);
    return *this;
}

void GeneratePipeline::swap(GeneratePipeline &other) noexcept
{
    std::swap(film_res_, other.film_res_);
    std::swap(pixel_count_, other.pixel_count_);
    std::swap(spp_, other.spp_);
    std::swap(state_count_, other.state_count_);
    std::swap(finished_spp_, other.finished_spp_);
    std::swap(finished_pixel_, other.finished_pixel_);
    std::swap(kernel_, other.kernel_);
}

GeneratePipeline::operator bool() const
{
    return spp_ > 0;
}

bool GeneratePipeline::is_done() const
{
    return finished_spp_ >= spp_;
}

void GeneratePipeline::clear()
{
    finished_spp_ = 0;
    finished_pixel_ = 0;
}

void GeneratePipeline::initialize(
    const Camera &camera, const Vec2i &film_res, int spp, int state_count)
{
    using namespace cuj;

    assert(!spp_ && !state_count_);
    assert(spp > 0 && state_count > 0);

    film_res_       = film_res;
    pixel_count_    = film_res_.x * film_res_.y;
    spp_            = spp;
    state_count_    = state_count;
    finished_spp_   = 0;
    finished_pixel_ = 0;

    ScopedModule cuj_module;

    auto generate_kernel = kernel(
        GENERATE_KERNEL_NAME,
        [&camera, this](
            ptr<cstd::LCG> rngs,
            ptr<CVec2f>    output_pixel_coord,
            ptr<CVec4f>    output_ray_o_t0,
            ptr<CVec4f>    output_ray_d_t1,
            ptr<CVec2u>    output_ray_time_mask,
            i32            initial_pixel_index,
            i32            active_state_count)
    {
        i32 thread_idx = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
        i32 state_index = active_state_count + thread_idx;
        i32 pixel_index = (initial_pixel_index + thread_idx) % pixel_count_;

        ref rng = rngs[state_index];

        i32 pixel_x = pixel_index % film_res_.x;
        i32 pixel_y = pixel_index / film_res_.y;

        f32 pixel_xf = f32(pixel_x) + rng.uniform_float();
        f32 pixel_yf = f32(pixel_y) + rng.uniform_float();

        f32 film_x = pixel_xf / static_cast<float>(film_res_.x);
        f32 film_y = pixel_yf / static_cast<float>(film_res_.y);

        f32 time_sample = rng.uniform_float();

        auto sample_we_result = camera.generate_ray(
            CVec2f(film_x, film_y), time_sample);

        output_pixel_coord[state_index] = CVec2f(pixel_xf, pixel_yf);

        output_ray_o_t0[state_index] = CVec4f(
            sample_we_result.pos.x,
            sample_we_result.pos.y,
            sample_we_result.pos.z,
            0.0f);

        output_ray_d_t1[state_index] = CVec4f(
            sample_we_result.dir.x,
            sample_we_result.dir.y,
            sample_we_result.dir.z,
            btrc_max_float);

        output_ray_time_mask[state_index] = CVec2u(
            bitcast<u32>(sample_we_result.time), 0xff);
    });

    PTXGenerator ptx_gen;
    ptx_gen.set_options(Options{
        .opt_level        = OptimizationLevel::O3,
        .fast_math        = true,
        .approx_math_func = true
    });
    ptx_gen.generate(cuj_module);

    const std::string &ptx = ptx_gen.get_ptx();
    kernel_.load_ptx_from_memory(ptx.data(), ptx.size());
    kernel_.link();
}

int GeneratePipeline::generate(
    int            active_state_count,
    cstd::LCGData *rngs,
    float2        *output_pixel_coord,
    const RaySOA  &output_ray)
{
    if(is_done())
        return 0;
    
    const int new_state_count = (std::min)(
        state_count_ - active_state_count,
        (spp_ - finished_spp_) * pixel_count_ - finished_pixel_);

    constexpr int BLOCK_DIM = 256;
    const int thread_count = new_state_count;
    const int block_count = up_align(thread_count, BLOCK_DIM) / BLOCK_DIM;

    kernel_.launch(
        GENERATE_KERNEL_NAME,
        { block_count, 1, 1 },
        { BLOCK_DIM, 1, 1 },
        rngs,
        output_pixel_coord,
        output_ray.o_t0,
        output_ray.d_t1,
        output_ray.time_mask,
        finished_pixel_,
        active_state_count);

    finished_pixel_ += new_state_count;
    finished_spp_   += finished_pixel_ / pixel_count_;
    finished_pixel_ %= pixel_count_;

    return new_state_count;
}

BTRC_WAVEFRONT_END
