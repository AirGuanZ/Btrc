#include <cassert>

#include <tiny_obj_loader.h>

#include <btrc/core/utils/math/triangle.h>
#include <btrc/core/utils/triangle_mesh_loader.h>

BTRC_CORE_BEGIN

TriangleMeshLoader::TriangleMeshLoader(const std::string &filename)
{
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;
    reader_config.vertex_color = false;
    if(!reader.ParseFromFile(filename, reader_config))
        throw BtrcException(reader.Error());

    auto &attrib = reader.GetAttrib();

    auto get_pos = [&](size_t index)
    {
        if(3 * index + 2 >= attrib.vertices.size())
            throw BtrcException("invalid obj vertex index: out of range");
        return Vec3f(
            attrib.vertices[3 * index],
            attrib.vertices[3 * index + 1],
            attrib.vertices[3 * index + 2]);
    };

    auto get_nor = [&](size_t index)
    {
        if(3 * index + 2 >= attrib.normals.size())
            throw BtrcException("invalid obj normal index: out of range");
        return Vec3f(
            attrib.normals[3 * index],
            attrib.normals[3 * index + 1],
            attrib.normals[3 * index + 2]);
    };

    auto get_tex_coord = [&](size_t index)
    {
        if(2 * index + 1 >= attrib.texcoords.size())
            throw BtrcException("invalid obj tex coord index: out of range");
        return Vec2f(
            attrib.texcoords[2 * index],
            attrib.texcoords[2 * index + 1]);
    };

    assert(attrib.vertices.size() % 3 == 0);
    for(size_t i = 0; i < attrib.vertices.size(); i += 3)
    {
        positions_.push_back({
            attrib.vertices[i],
            attrib.vertices[i + 1],
            attrib.vertices[i + 2],
        });
    }

    int32_t max_position_index_ = -1;
    for(auto &shape : reader.GetShapes())
    {
        for(auto fvc : shape.mesh.num_face_vertices)
        {
            if(fvc != 3)
            {
                throw BtrcException(
                    "invalid obj face vertex count: " +
                    std::to_string(+fvc));
            }
        }

        if(shape.mesh.indices.size() % 3 != 0)
        {
            throw BtrcException(
                "invalid obj index count: " +
                std::to_string(shape.mesh.indices.size()));
        }

        const size_t triangle_count = shape.mesh.indices.size() / 3;
        for(size_t i = 0, j = 0; i < triangle_count; ++i, j += 3)
        {
            const Vec3f pos_a = get_pos(shape.mesh.indices[j].vertex_index);
            const Vec3f pos_b = get_pos(shape.mesh.indices[j + 1].vertex_index);
            const Vec3f pos_c = get_pos(shape.mesh.indices[j + 2].vertex_index);
            const Vec3f geo_z = normalize(cross(pos_b - pos_a, pos_c - pos_a));

            for(size_t k = 0; k < 3; ++k)
            {
                const auto &index = shape.mesh.indices[j + k];

                indices_i32_.push_back(index.vertex_index);
                max_position_index_ =
                    (std::max)(max_position_index_, index.vertex_index);

                if(index.normal_index < 0)
                    interp_ezs_.push_back(geo_z);
                else
                {
                    interp_ezs_.push_back(get_nor(index.normal_index));
                    auto &int_z = interp_ezs_.back();
                    if(int_z.x == 0.0f && int_z.y == 0.0f && int_z.z == 0.0f)
                        int_z = geo_z;
                }

                if(index.texcoord_index < 0)
                    tex_coords_.emplace_back(0.0f);
                else
                    tex_coords_.push_back(get_tex_coord(index.texcoord_index));
            }

            const Vec2f &tex_coord_a = tex_coords_[tex_coords_.size() - 3];
            const Vec2f &tex_coord_b = tex_coords_[tex_coords_.size() - 2];
            const Vec2f &tex_coord_c = tex_coords_[tex_coords_.size() - 1];

            geometry_exs_.push_back(triangle_dpdu(
                pos_b - pos_a,
                pos_c - pos_a,
                tex_coord_b - tex_coord_a,
                tex_coord_c - tex_coord_a,
                geo_z));
            geometry_ezs_.push_back(geo_z);
        }
    }

    if(indices_i32_.size() == positions_.size())
    {
        bool remove_indices = true;
        for(size_t i = 0; i < indices_i32_.size(); ++i)
        {
            if(indices_i32_[i] != static_cast<int32_t>(i))
            {
                remove_indices = false;
                break;
            }
        }
        if(remove_indices)
            indices_i32_.clear();
    }

    if(!indices_i32_.empty() &&
        max_position_index_ <= (std::numeric_limits<int16_t>::max)())
    {
        indices_i16_.resize(indices_i32_.size());
        for(size_t i = 0; i < indices_i32_.size(); ++i)
            indices_i16_[i] = static_cast<int16_t>(indices_i32_[i]);
        indices_i32_.clear();
    }
}

TriangleMeshLoader::TriangleMeshLoader(TriangleMeshLoader &&other) noexcept
    : TriangleMeshLoader()
{
    swap(other);
}

TriangleMeshLoader &TriangleMeshLoader::operator=(TriangleMeshLoader &&other) noexcept
{
    swap(other);
    return *this;
}

void TriangleMeshLoader::swap(TriangleMeshLoader &other) noexcept
{
    positions_.swap(other.positions_);
    indices_i16_.swap(other.indices_i16_);
    indices_i32_.swap(other.indices_i32_);
    tex_coords_.swap(other.tex_coords_);
    interp_ezs_.swap(other.interp_ezs_);
    geometry_exs_.swap(other.geometry_exs_);
    geometry_ezs_.swap(other.geometry_ezs_);
}

TriangleMeshLoader::operator bool() const
{
    return !positions_.empty();
}

void TriangleMeshLoader::remove_indices()
{
    if(indices_i16_.empty() && indices_i32_.empty())
        return;

    std::vector<Vec3f> new_positions;
    for(auto i : indices_i16_)
        new_positions.push_back(positions_[i]);
    for(auto i : indices_i32_)
        new_positions.push_back(positions_[i]);

    positions_.swap(new_positions);
    indices_i16_.clear();
    indices_i32_.clear();
}

size_t TriangleMeshLoader::get_primitive_count() const
{
    return (std::max)(
        (std::max)(indices_i16_.size(), indices_i32_.size()),
        positions_.size()) / 3;
}

std::span<const Vec3f> TriangleMeshLoader::get_positions() const
{
    return positions_;
}

std::span<const int16_t> TriangleMeshLoader::get_indices_i16() const
{
    return indices_i16_;
}

std::span<const int32_t> TriangleMeshLoader::get_indices_i32() const
{
    return indices_i32_;
}

std::span<const Vec2f> TriangleMeshLoader::get_tex_coords() const
{
    return tex_coords_;
}

std::span<const Vec3f> TriangleMeshLoader::get_geometry_exs() const
{
    return geometry_exs_;
}

std::span<const Vec3f> TriangleMeshLoader::get_geometry_ezs() const
{
    return geometry_ezs_;
}

std::span<const Vec3f> TriangleMeshLoader::get_interp_ezs() const
{
    return interp_ezs_;
}

BTRC_CORE_END
