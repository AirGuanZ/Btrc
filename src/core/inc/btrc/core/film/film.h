#pragma once

#include <span>
#include <string_view>

#include <btrc/core/utils/cuda/buffer.h>
#include <btrc/core/utils/cmath/cmath.h>
#include <btrc/core/utils/uncopyable.h>
#include <btrc/core/utils/variant.h>

BTRC_CORE_BEGIN

class Film : public Uncopyable
{
public:

    enum Format
    {
        Float,
        Float3,
    };

    using CValue = Variant<f32, CVec3f>;

    static constexpr char OUTPUT_RADIANCE[] = "radiance";
    static constexpr char OUTPUT_WEIGHT[]   = "weight";
    static constexpr char OUTPUT_NORMAL[]   = "normal";
    static constexpr char OUTPUT_ALBEDO[]   = "albedo";

    Film();

    Film(int width, int height);

    Film(Film &&other) noexcept;

    Film &operator=(Film &&other) noexcept;

    void swap(Film &other) noexcept;

    operator bool() const;

    int width() const;

    int height() const;

    void add_output(std::string name, Format format);

    bool has_output(std::string_view name) const;

    void splat(
        const CVec2f &pixel_coord,
        std::span<std::pair<std::string_view, CValue>> values);

    void splat_atomic(
        const CVec2f &pixel_coord,
        std::span<std::pair<std::string_view, CValue>> values);
    
    void splat(
        const CVec2f    &pixel_coord,
        std::string_view name,
        const CValue    &value);

    void splat_atomic(
        const CVec2f    &pixel_coord,
        std::string_view name,
        const CValue    &value);

    void clear_output(std::string_view name);

    const CUDABuffer<float> &get_float_output(std::string_view name) const;

    const CUDABuffer<float> &get_float3_output(std::string_view name) const;

private:

    struct FilmBuffer
    {
        Format            format;
        CUDABuffer<float> buffer;
    };

    int width_;
    int height_;
    std::map<std::string, FilmBuffer, std::less<>> buffers_;
};

BTRC_CORE_END
