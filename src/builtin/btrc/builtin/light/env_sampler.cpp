#include <numeric>

#include <btrc/builtin/light/env_sampler.h>
#include <btrc/utils/cuda/module.h>
#include <btrc/utils/local_angle.h>
#include <btrc/utils/math/hammersley.h>

BTRC_BUILTIN_BEGIN

namespace
{

    const char KERNEL[] = "generate_lum_sum_table";

    std::string generate_sample_texture_kernel(
        const RC<const Texture2D> &tex, const Vec2i &lut_res, int n_samples)
    {
        CompileContext cc;
        cuj::ScopedModule cuj_module;

        cuj::kernel(KERNEL, [&cc, &tex, lut_res, n_samples](ptr<f32> table)
        {
            $declare_scope;

            var xi = cstd::block_dim_x() * cstd::block_idx_x() + cstd::thread_idx_x();
            var yi = cstd::block_dim_y() * cstd::block_idx_y() + cstd::thread_idx_y();
            $if(xi >= lut_res.x | yi >= lut_res.y)
            {
                $exit_scope;
            };

            var x0 = f32(xi)     / lut_res.x;
            var x1 = f32(xi + 1) / lut_res.x;
            var y0 = f32(yi)     / lut_res.y;
            var y1 = f32(yi + 1) / lut_res.y;

            std::vector<Vec2f> local_samples_data(n_samples);
            for(int i = 0; i < n_samples; ++i)
                local_samples_data[i] = hammersley2d(i, n_samples);
            var local_samples = cuj::const_data(std::span{ local_samples_data });

            var i = 0;
            var lum_sum = 0.0f;
            $while(i < n_samples)
            {
                var local_sample = 1.4f * local_samples[i] - CVec2f(0.2f);
                i = i + 1;
                var x = lerp(x0, x1, local_sample.x);
                var y = lerp(y0, y1, local_sample.y);
                var value = tex->sample_spectrum(cc, CVec2f(x, y));
                lum_sum = lum_sum + value.get_lum();
            };

            var lum = lum_sum / n_samples;
            var delta_area = cstd::abs(
                2 * btrc_pi * (x1 - x0) * (cstd::cos(btrc_pi * y1) - cstd::cos(btrc_pi * y0)));
            table[yi * lut_res.x + xi] = lum * delta_area;
        });
        
        cuj::PTXGenerator gen;
        gen.set_options(cuj::Options{
            .opt_level = cuj::OptimizationLevel::O3,
            .fast_math = true,
            .approx_math_func = true
        });
        gen.generate(cuj_module);

        return gen.get_ptx();
    }

} // namespace anonymous

void EnvirLightSampler::preprocess(const RC<const Texture2D> &tex, const Vec2i &lut_res, int n_samples)
{
    const std::string ptx = generate_sample_texture_kernel(tex, lut_res, n_samples);

    CUDAModule cuda_module;
    cuda_module.load_ptx_from_memory(ptx.data(), ptx.size());
    cuda_module.link();

    cuda::CUDABuffer<float> device_table(lut_res.x * lut_res.y);
    constexpr int BLOCK_SIZE = 8;
    const int block_cnt_x = up_align(lut_res.x, BLOCK_SIZE) / BLOCK_SIZE;
    const int block_cnt_y = up_align(lut_res.y, BLOCK_SIZE) / BLOCK_SIZE;
    cuda_module.launch(
        KERNEL,
        { block_cnt_x, block_cnt_y, 1 },
        { BLOCK_SIZE, BLOCK_SIZE, 1 },
        device_table.get());
    throw_on_error(cudaStreamSynchronize(nullptr));

    std::vector<float> table(device_table.get_size());
    device_table.to_cpu(table.data());

    const float table_sum = std::accumulate(table.begin(), table.end(), 0.0f);
    if(table_sum > 1e-3f)
    {
        const float ratio = 1 / table_sum;
        for(auto &v : table)
            v *= ratio;
    }

    lut_res_ = lut_res;
    tile_probs_ = cuda::CUDABuffer<float>(table);
    tile_alias_ = CAliasTable(AliasTable(table));
}

EnvirLightSampler::SampleResult EnvirLightSampler::sample(ref<CVec3f> sam) const
{
    var tile_idx = tile_alias_.sample(sam.x);
    var tile_y = tile_idx / lut_res_.x;
    var tile_x = tile_idx % lut_res_.x;
    CUJ_ASSERT(tile_y < lut_res_.y);

    var tile_pdf_table = cuj::import_pointer(tile_probs_.get());
    var tile_pdf = tile_pdf_table[tile_idx];

    var u0 = f32(tile_x)     / lut_res_.x;
    var u1 = f32(tile_x + 1) / lut_res_.x;
    var v0 = f32(tile_y)     / lut_res_.y;
    var v1 = f32(tile_y + 1) / lut_res_.y;

    var cv0 = cstd::cos(btrc_pi * v0);
    var cv1 = cstd::cos(btrc_pi * v1);
    $if(cv0 > cv1)
    {
        var t = cv0;
        cv0 = cv1;
        cv1 = t;
    };

    var cos_theta = cv0 + sam.z * (cv1 - cv0);
    var sin_theta = local_angle::cos2sin(cos_theta);
    var phi = 2 * btrc_pi * lerp(u0, u1, sam.y);

    var dir = CVec3f(sin_theta * cstd::cos(phi), sin_theta * cstd::sin(phi), cos_theta);
    var in_tile_pdf = 1.0f / (2 * btrc_pi * (u1 - u0) * (cv1 - cv0));

    SampleResult result;
    result.to_light = dir;
    result.pdf = tile_pdf * in_tile_pdf;
    return result;
}

f32 EnvirLightSampler::pdf(ref<CVec3f> to_light) const
{
    var dir = normalize(to_light);
    var cos_theta = local_angle::cos_theta(dir);
    var theta = cstd::acos(cos_theta);
    var phi = local_angle::phi(dir);

    var u = phi / (2 * btrc_pi);
    var v = theta / btrc_pi;

    var tile_x = cstd::clamp(i32(cstd::floor(u * lut_res_.x)), 0, lut_res_.x - 1);
    var tile_y = cstd::clamp(i32(cstd::floor(v * lut_res_.y)), 0, lut_res_.y - 1);
    var tile_idx = tile_y * lut_res_.x + tile_x;

    var tile_pdf_table = cuj::import_pointer(tile_probs_.get());
    var tile_pdf = tile_pdf_table[tile_idx];

    var u0 = f32(tile_x)     / lut_res_.x;
    var u1 = f32(tile_x + 1) / lut_res_.x;
    var v0 = f32(tile_y)     / lut_res_.y;
    var v1 = f32(tile_y + 1) / lut_res_.y;
    var c0 = cstd::cos(btrc_pi * v0);
    var c1 = cstd::cos(btrc_pi * v1);
    var in_tile_pdf = 1.0f / cstd::abs(2 * btrc_pi * (u1 - u0) * (c1 - c0));

    return tile_pdf * in_tile_pdf;
}

BTRC_BUILTIN_END
