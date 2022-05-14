#pragma once

#include <span>

#include <btrc/utils/cuda/buffer.h>
#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/uncopyable.h>
#include <btrc/utils/variant.h>

BTRC_BEGIN

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

    Vec2i size() const;

    void add_output(std::string name, Format format);

    bool has_output(std::string_view name) const;

    void splat(
        const CVec2u &pixel_coord,
        std::span<std::pair<std::string_view, CValue>> values);

    void splat_atomic(
        const CVec2u &pixel_coord,
        std::span<std::pair<std::string_view, CValue>> values);
    
    void splat(
        const CVec2u &pixel_coord,
        std::string_view name,
        const CValue    &value);

    void splat_atomic(
        const CVec2u &pixel_coord,
        std::string_view name,
        const CValue    &value);

    void clear_output(std::string_view name);

    const cuda::Buffer<float> &get_float_output(std::string_view name) const;

    const cuda::Buffer<float> &get_float3_output(std::string_view name) const;

private:

    struct FilmBuffer
    {
        Format              format;
        cuda::Buffer<float> buffer;
    };

    int width_;
    int height_;
    std::map<std::string, FilmBuffer, std::less<>> buffers_;
};

BTRC_END
