#pragma once

#include <btrc/core/volume.h>

BTRC_BUILTIN_BEGIN

namespace volume
{

    constexpr int MAX_OVERLAP_COUNT = 4;

    struct BVHNode
    {
        Vec3f lower; int32_t prim_beg; // -1 for interior node
        Vec3f upper; int32_t prim_end; // left child index for interior node
    };

    CUJ_PROXY_CLASS(CBVHNode, BVHNode, lower, prim_beg, upper, prim_end);

    struct BVHPrimitive
    {
        Vec3f o; uint32_t vol_id;
        Vec3f x_div_x2; // x / |x|^2
        Vec3f y_div_y2; // y / |y|^2
        Vec3f z_div_z2; // z / |z|^2
    };

    CUJ_PROXY_CLASS(CBVHPrimitive, BVHPrimitive, o, vol_id, x_div_x2, y_div_y2, z_div_z2);

    class BVH
    {
    public:

        CUJ_CLASS_BEGIN(Overlap)
            using OverlapData = cuj::arr<u32, MAX_OVERLAP_COUNT>;
            CUJ_MEMBER_VARIABLE(OverlapData, data)
            CUJ_MEMBER_VARIABLE(i32, count)
        CUJ_CLASS_END

        explicit BVH(const std::vector<RC<VolumePrimitive>> &vols);

        bool is_empty() const;

        boolean find_closest_intersection(ref<CVec3f> a, ref<CVec3f> b, ref<CVec3f> &output_position) const;

        Overlap get_overlap(ref<CVec3f> position) const;

    private:

        boolean is_leaf_node(ref<CBVHNode> node) const;

        boolean is_in_prim(ref<CVec3f> pos, ref<CBVHPrimitive> prim) const;

        boolean is_in_aabb(ref<CVec3f> pos, ref<CVec3f> lower, ref<CVec3f> upper) const;

        boolean intersect_ray_aabb(ref<CVec3f> lower, ref<CVec3f> upper, ref<CVec3f> o, ref<CVec3f> inv_d, f32 t_max) const;

        // returns -1 when there is no intersection
        f32 find_closest_intersection(ref<CVec3f> o, ref<CVec3f> inv_d, f32 t_max, ref<CBVHPrimitive> prim) const;

        cuda::Buffer<BVHNode>      nodes_;
        cuda::Buffer<BVHPrimitive> prims_;
    };

} // namespace volume

BTRC_BUILTIN_END
