#include <btrc/core/medium.h>
#include <btrc/utils/cmath/cmath.h>

#include "./generate.h"

BTRC_WFPT_BEGIN

namespace
{

    const char *GENERATE_KERNEL_NAME = "generate_kernel";

} // namespace anonymous

GeneratePipeline::GeneratePipeline()
    : pixel_count_(0), spp_(0), state_count_(0),
      finished_spp_(0), finished_pixel_(0)
{

}

void GeneratePipeline::record_device_code(CompileContext &cc, const Camera &camera, const Vec2i &film_res)
{
    using namespace cuj;

    kernel(
        GENERATE_KERNEL_NAME,
        [&cc, &camera, &film_res](
            CSOAParams soa_params,
            i32        initial_pixel_index,
            i32        new_state_count,
            i32        active_state_count)
    {
        i32 thread_idx = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
        $if(thread_idx >= new_state_count)
        {
            $return();
        };

        i32 state_index = active_state_count + thread_idx;
        i32 pixel_index = (initial_pixel_index + thread_idx) % (film_res.x * film_res.y);

        ref rng = soa_params.rng[state_index];

        i32 pixel_x = pixel_index % film_res.x;
        i32 pixel_y = pixel_index / film_res.y;

        f32 pixel_xf = f32(pixel_x) + rng.uniform_float();
        f32 pixel_yf = f32(pixel_y) + rng.uniform_float();

        f32 film_x = pixel_xf / static_cast<float>(film_res.x);
        f32 film_y = pixel_yf / static_cast<float>(film_res.y);

        f32 time_sample = rng.uniform_float();

        auto sample_we_result = camera.generate_ray(
            cc, CVec2f(film_x, film_y), time_sample);

        soa_params.output_pixel_coord[state_index] = CVec2f(pixel_xf, pixel_yf);

        var o_t0 = CVec4f(
            sample_we_result.pos.x,
            sample_we_result.pos.y,
            sample_we_result.pos.z,
            0.0f);
        save_aligned(o_t0, soa_params.output_ray_o_t0 + state_index);

        var d_t1 = CVec4f(
            sample_we_result.dir.x,
            sample_we_result.dir.y,
            sample_we_result.dir.z,
            btrc_max_float);
        save_aligned(d_t1, soa_params.output_ray_d_t1 + state_index);

        var time_mask = CVec2u(bitcast<u32>(sample_we_result.time), 0xff);
        save_aligned(time_mask, soa_params.output_ray_time_mask + state_index);

        soa_params.output_depth[state_index] = 0;
        soa_params.output_bsdf_pdf[state_index] = -1;

        save_aligned(sample_we_result.throughput, soa_params.output_beta + state_index);
        save_aligned(sample_we_result.throughput, soa_params.output_beta_le + state_index);
        save_aligned(CSpectrum::zero(), soa_params.output_path_radiance + state_index);

        soa_params.output_ray_medium_id[state_index] = MEDIUM_ID_VOID;
    });
}

void GeneratePipeline::initialize(RC<cuda::Module> cuda_module, int spp, int state_count, const Vec2i &film_res)
{
    assert(!spp_ && !state_count_);
    assert(spp > 0 && state_count > 0);

    film_res_ = film_res;
    pixel_count_ = film_res_.x * film_res_.y;
    spp_ = spp;
    state_count_ = state_count;
    finished_spp_ = 0;
    finished_pixel_ = 0;

    cuda_module_ = std::move(cuda_module);
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
    std::swap(cuda_module_, other.cuda_module_);
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

int GeneratePipeline::generate(
    int              active_state_count,
    const SOAParams &launch_params,
    int64_t          limit_max_state_count)
{
    if(is_done())
        return 0;

    const int64_t total_state_count = (std::min)(state_count_, limit_max_state_count);
    const int64_t rest_state_count = total_state_count - active_state_count;
    if(rest_state_count <= 0)
        return 0;

    const int64_t unfinished_path_count = (spp_ - finished_spp_) * pixel_count_ - finished_pixel_;
    if(unfinished_path_count > rest_state_count &&
       rest_state_count + rest_state_count < total_state_count)
        return 0;

    const int new_state_count = static_cast<int>((std::min)(rest_state_count, unfinished_path_count));

    constexpr int BLOCK_DIM = 256;
    const int thread_count = new_state_count;
    const int block_count = up_align(thread_count, BLOCK_DIM) / BLOCK_DIM;

    cuda_module_->launch(
        GENERATE_KERNEL_NAME,
        { block_count, 1, 1 },
        { BLOCK_DIM, 1, 1 },
        launch_params,
        finished_pixel_,
        new_state_count,
        active_state_count);

    finished_pixel_ += new_state_count;
    finished_spp_   += finished_pixel_ / pixel_count_;
    finished_pixel_ %= pixel_count_;

    return new_state_count;
}

float GeneratePipeline::get_generated_percentage() const
{
    return 100.0f * finished_spp_ / spp_;
}

BTRC_WFPT_END
