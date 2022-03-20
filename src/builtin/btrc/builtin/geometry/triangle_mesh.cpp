#include <numeric>

#include <btrc/builtin/geometry/triangle_mesh.h>
#include <btrc/utils/math/triangle.h>
#include <btrc/utils/triangle_mesh_loader.h>

BTRC_BUILTIN_BEGIN

void TriangleMesh::set_optix_context(optix::Context &optix_ctx)
{
    optix_ctx_ = &optix_ctx;
    set_need_commit();
}

void TriangleMesh::set_filename(std::string filename)
{
    filename_ = std::move(filename);
    set_need_commit();
}

void TriangleMesh::set_transform_to_unit_cube(bool transform)
{
    transform_to_unit_cube_ = transform;
    set_need_commit();
}

void TriangleMesh::commit()
{
    TriangleMeshLoader loader(filename_);
    if(transform_to_unit_cube_)
        loader.transform_to_unit_cube();
    if(!loader.get_indices_i32().empty())
    {
        as_ = optix_ctx_->create_triangle_as(
            loader.get_positions(), loader.get_indices_i32());
    }
    else
    {
        as_ = optix_ctx_->create_triangle_as(
            loader.get_positions(), loader.get_indices_i16());
    }

    std::vector<Vec4f> gx_tex_coord_u_a;
    std::vector<Vec4f> gy_tex_coord_u_ba;
    std::vector<Vec4f> gz_tex_coord_u_ca;
    std::vector<Vec4f> sz_tex_coord_v_a;
    std::vector<Vec4f> sz_tex_coord_v_ba;
    std::vector<Vec4f> sz_tex_coord_v_ca;

    for(size_t i = 0; i < loader.get_primitive_count(); ++i)
    {
        const Vec3f gx = loader.get_geometry_exs()[i];
        const Vec3f gz = loader.get_geometry_ezs()[i];
        const Vec3f gy = normalize(cross(gz, gx));

        const Vec2f uv_a = loader.get_tex_coords()[i * 3 + 0];
        const Vec2f uv_b = loader.get_tex_coords()[i * 3 + 1];
        const Vec2f uv_c = loader.get_tex_coords()[i * 3 + 2];

        const Vec3f sz_a = loader.get_interp_ezs()[i * 3 + 0];
        const Vec3f sz_b = loader.get_interp_ezs()[i * 3 + 1];
        const Vec3f sz_c = loader.get_interp_ezs()[i * 3 + 2];

        gx_tex_coord_u_a.push_back(Vec4f(gx, uv_a.x));
        gy_tex_coord_u_ba.push_back(Vec4f(gy, uv_b.x - uv_a.x));
        gz_tex_coord_u_ca.push_back(Vec4f(gz, uv_c.x - uv_a.x));

        sz_tex_coord_v_a.push_back(Vec4f(sz_a, uv_a.y));
        sz_tex_coord_v_ba.push_back(Vec4f(sz_b - sz_a, uv_b.y - uv_a.y));
        sz_tex_coord_v_ca.push_back(Vec4f(sz_c - sz_a, uv_c.y - uv_a.y));
    }

    geo_info_buf_.initialize(
        gx_tex_coord_u_a.size()  +
        gy_tex_coord_u_ba.size() +
        gz_tex_coord_u_ca.size() +
        sz_tex_coord_v_a.size()  +
        sz_tex_coord_v_ba.size() +
        sz_tex_coord_v_ca.size());

    size_t offset = 0;
    auto append = [&](const std::vector<Vec4f> &data)
    {
        geo_info_buf_.from_cpu(
            data.data(), offset, offset + data.size());
        offset += data.size();
    };
    append(gx_tex_coord_u_a);
    append(gy_tex_coord_u_ba);
    append(gz_tex_coord_u_ca);
    append(sz_tex_coord_v_a);
    append(sz_tex_coord_v_ba);
    append(sz_tex_coord_v_ca);

    offset = 0;
    auto get_ptr = [&](size_t cnt)
    {
        auto ret = geo_info_buf_.get() + offset;
        offset += cnt;
        return ret;
    };

    geo_info_.geometry_ex_tex_coord_u_a     = get_ptr(gx_tex_coord_u_a.size());
    geo_info_.geometry_ey_tex_coord_u_ba    = get_ptr(gy_tex_coord_u_ba.size());
    geo_info_.geometry_ez_tex_coord_u_ca    = get_ptr(gz_tex_coord_u_ca.size());
    geo_info_.shading_normal_tex_coord_v_a  = get_ptr(sz_tex_coord_v_a.size());
    geo_info_.shading_normal_tex_coord_v_ba = get_ptr(sz_tex_coord_v_ba.size());
    geo_info_.shading_normal_tex_coord_v_ca = get_ptr(sz_tex_coord_v_ca.size());

    const size_t prim_count = loader.get_primitive_count();
    std::vector<float> triangle_areas(prim_count);
    std::vector<float> positions(prim_count * 9);

    auto process_prim = [&](
        size_t prim_idx, const Vec3f &a, const Vec3f &b, const Vec3f &c)
    {
        const Vec3f ba = b - a, ca = c - a;
        triangle_areas[prim_idx] = triangle_area(ba, ca);
        positions[prim_idx * 9 + 0] = a.x;
        positions[prim_idx * 9 + 1] = a.y;
        positions[prim_idx * 9 + 2] = a.z;
        positions[prim_idx * 9 + 3] = ba.x;
        positions[prim_idx * 9 + 4] = ba.y;
        positions[prim_idx * 9 + 5] = ba.z;
        positions[prim_idx * 9 + 6] = ca.x;
        positions[prim_idx * 9 + 7] = ca.y;
        positions[prim_idx * 9 + 8] = ca.z;
    };

    if(!loader.get_indices_i32().empty())
    {
        for(size_t i = 0; i < prim_count; ++i)
        {
            process_prim(
                i,
                loader.get_positions()[loader.get_indices_i32()[3 * i + 0]],
                loader.get_positions()[loader.get_indices_i32()[3 * i + 1]],
                loader.get_positions()[loader.get_indices_i32()[3 * i + 2]]);
        }
    }
    else if(!loader.get_indices_i16().empty())
    {
        for(size_t i = 0; i < prim_count; ++i)
        {
            process_prim(
                i,
                loader.get_positions()[loader.get_indices_i16()[3 * i + 0]],
                loader.get_positions()[loader.get_indices_i16()[3 * i + 1]],
                loader.get_positions()[loader.get_indices_i16()[3 * i + 2]]);
        }
    }
    else
    {
        for(size_t i = 0; i < prim_count; ++i)
        {
            process_prim(
                i,
                loader.get_positions()[3 * i + 0],
                loader.get_positions()[3 * i + 1],
                loader.get_positions()[3 * i + 2]);
        }
    }

    positions_ = cuda::Buffer<float>(positions);

    AliasTable table(triangle_areas);
    alias_table_ = CAliasTable(table);

    total_area_ = std::accumulate(triangle_areas.begin(), triangle_areas.end(), 0.0f);

    bbox_ = AABB3f{};
    for(auto &p : loader.get_positions())
        bbox_ = union_aabb(bbox_, p);
}

OptixTraversableHandle TriangleMesh::get_blas() const
{
    return as_;
}

const GeometryInfo &TriangleMesh::get_geometry_info() const
{
    return geo_info_;
}

AABB3f TriangleMesh::get_bounding_box() const
{
    return bbox_;
}

Geometry::SampleResult TriangleMesh::sample_inline(ref<CVec3f> sam) const
{
    using namespace cuj;

    var prim_idx = alias_table_.sample(sam.x);
    var uv = sample_triangle_uniform(sam.y, sam.z);

    var pos_ptr = bitcast<ptr<CVec3f>>(import_pointer(positions_.get()));
    var a  = pos_ptr[prim_idx * 3 + 0];
    var ba = pos_ptr[prim_idx * 3 + 1];
    var ca = pos_ptr[prim_idx * 3 + 2];
    var pos = a + ba * uv.x + ca * uv.y;

    var ex_u_a  = load_aligned(import_pointer(geo_info_.geometry_ex_tex_coord_u_a)  + prim_idx);
    var ey_u_ba = load_aligned(import_pointer(geo_info_.geometry_ey_tex_coord_u_ba) + prim_idx);
    var ez_u_ca = load_aligned(import_pointer(geo_info_.geometry_ez_tex_coord_u_ca) + prim_idx);

    var ex = ex_u_a.xyz();
    var ey = ey_u_ba.xyz();
    var ez = ez_u_ca.xyz();

    var sz_v_a  = load_aligned(import_pointer(geo_info_.shading_normal_tex_coord_v_a)  + prim_idx);
    var sz_v_ba = load_aligned(import_pointer(geo_info_.shading_normal_tex_coord_v_ba) + prim_idx);
    var sz_v_ca = load_aligned(import_pointer(geo_info_.shading_normal_tex_coord_v_ca) + prim_idx);

    var tex_coord_a  = CVec2f(ex_u_a.w,  sz_v_a.w);
    var tex_coord_ba = CVec2f(ey_u_ba.w, sz_v_ba.w);
    var tex_coord_ca = CVec2f(ez_u_ca.w, sz_v_ca.w);

    var tex_coord = tex_coord_a + tex_coord_ba * uv.x + tex_coord_ca * uv.y;
    var interp_z = normalize(sz_v_a.xyz() + sz_v_ba.xyz() * uv.x + sz_v_ca.xyz() * uv.y);

    SampleResult result;
    result.point.position  = pos;
    result.point.uv        = uv;
    result.point.tex_coord = tex_coord;
    result.point.frame     = CFrame(ex, ey, ez);
    result.point.interp_z  = interp_z;
    result.pdf             = 1 / total_area_;
    return result;
}

f32 TriangleMesh::pdf_inline(ref<CVec3f> pos) const
{
    return 1 / total_area_;
}

Geometry::SampleResult TriangleMesh::sample_inline(ref<CVec3f> dst_pos, ref<CVec3f> sam) const
{
    return sample_inline(sam);
}

f32 TriangleMesh::pdf_inline(ref<CVec3f> dst_pos, ref<CVec3f> pos) const
{
    return pdf_inline(pos);
}

RC<Geometry> TriangleMeshCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const auto filename = context.resolve_path(node->parse_child<std::string>("filename")).string();
    const bool transform_to_unit_cube = node->parse_child_or<bool>("transform_to_unit_cube", false);
    auto mesh = newRC<TriangleMesh>();
    mesh->set_optix_context(context.get_optix_context());
    mesh->set_filename(filename);
    mesh->set_transform_to_unit_cube(transform_to_unit_cube);
    return mesh;
}

BTRC_BUILTIN_END
