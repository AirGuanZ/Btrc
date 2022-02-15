#pragma once

#include <span>
#include <vector>

#include <btrc/utils/math/math.h>
#include <btrc/utils/uncopyable.h>

BTRC_BEGIN

class TriangleMeshLoader : public Uncopyable
{
public:

    TriangleMeshLoader() = default;

    explicit TriangleMeshLoader(const std::string &filename);

    TriangleMeshLoader(TriangleMeshLoader &&other) noexcept;

    TriangleMeshLoader &operator=(TriangleMeshLoader &&other) noexcept;

    void swap(TriangleMeshLoader &other) noexcept;

    operator bool() const;

    void remove_indices();

    void transform_to_unit_cube();

    size_t get_primitive_count() const;

    std::span<const Vec3f> get_positions() const;

    std::span<const int16_t> get_indices_i16() const;

    std::span<const int32_t> get_indices_i32() const;

    std::span<const Vec2f> get_tex_coords() const;

    std::span<const Vec3f> get_geometry_exs() const;

    std::span<const Vec3f> get_geometry_ezs() const;

    std::span<const Vec3f> get_interp_ezs() const;

private:

    std::vector<Vec3f>   positions_;
    std::vector<int16_t> indices_i16_;
    std::vector<int32_t> indices_i32_;

    // per primitive * 3
    std::vector<Vec2f> tex_coords_;
    std::vector<Vec3f> interp_ezs_;

    // per primitive * 1
    std::vector<Vec3f> geometry_exs_;
    std::vector<Vec3f> geometry_ezs_;
};

BTRC_END
