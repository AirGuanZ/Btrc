#include <bvh/bvh.hpp>
#include <bvh/leaf_collapser.hpp>
#include <bvh/locally_ordered_clustering_builder.hpp>
#include <bvh/parallel_reinsertion_optimizer.hpp>

#include <btrc/core/volume/bvh.h>
#include <btrc/utils/enumerate.h>

BTRC_BEGIN

namespace
{

    constexpr int TRAVERSAL_STACK_SIZE = 16;

    bvh::Vector3<float> convert(const Vec3f &v)
    {
        return bvh::Vector3<float>(v.x, v.y, v.z);
    }

    bvh::BoundingBox<float> convert(const AABB3f &bbox)
    {
        return bvh::BoundingBox(convert(bbox.lower), convert(bbox.upper));
    }

    f32 max(f32 a, f32 b, f32 c, f32 d)
    {
        return cstd::max(cstd::max(a, b), cstd::max(c, d));
    }

    f32 min(f32 a, f32 b, f32 c, f32 d)
    {
        return cstd::min(cstd::min(a, b), cstd::min(c, d));
    }

} // namespace anonymous

volume::BVH::BVH(const std::vector<RC<VolumePrimitive>> &vols)
{
    if(vols.empty())
        return;

    AABB3f global_bbox;
    std::vector<bvh::BoundingBox<float>> aabbs(vols.size());
    std::vector<bvh::Vector3<float>> centers(vols.size());
    for(auto &&[i, vol] : enumerate(vols))
    {
        const AABB3f aabb = vol->get_bounding_box();
        aabbs[i] = convert(aabb);
        centers[i] = convert(0.5f * (aabb.lower + aabb.upper));
        global_bbox = union_aabb(global_bbox, aabb);
    }

    bvh::Bvh<float> tree;

    bvh::LocallyOrderedClusteringBuilder<bvh::Bvh<float>, uint64_t> builder(tree);
    builder.build(convert(global_bbox), aabbs.data(), centers.data(), vols.size());

    bvh::ParallelReinsertionOptimizer bvh_optimizer(tree);
    bvh_optimizer.optimize();

    bvh::LeafCollapser leaf_collapser(tree);
    leaf_collapser.collapse();

    std::vector<BVHNode> nodes(tree.node_count);
    std::vector<BVHPrimitive> prims;

    for(size_t ni = 0; ni < tree.node_count; ++ni)
    {
        auto &src_node = tree.nodes[ni];
        auto &dst_node = nodes[ni];

        dst_node.lower.x = src_node.bounds[0];
        dst_node.lower.y = src_node.bounds[2];
        dst_node.lower.z = src_node.bounds[4];
        dst_node.upper.x = src_node.bounds[1];
        dst_node.upper.y = src_node.bounds[3];
        dst_node.upper.z = src_node.bounds[5];

        if(src_node.is_leaf())
        {
            const size_t prim_beg = prims.size();

            const size_t i_end = src_node.first_child_or_primitive + src_node.primitive_count;
            for(size_t i = src_node.first_child_or_primitive; i < i_end; ++i)
            {
                const size_t pi = tree.primitive_indices[i];
                auto &vol = vols[pi];
                const VolumePrimitive::VolumeGeometryInfo geo = vol->get_geometry_info();
                const BVHPrimitive prim_info = {
                    .o        = geo.o,
                    .vol_id   = static_cast<uint32_t>(pi),
                    .x_div_x2 = geo.x / length_square(geo.x),
                    .y_div_y2 = geo.y / length_square(geo.y),
                    .z_div_z2 = geo.z / length_square(geo.z)
                };
                prims.push_back(prim_info);
            }

            const size_t prim_end = prims.size();
            assert(prim_end > prim_beg);

            dst_node.prim_beg = static_cast<int32_t>(prim_beg);
            dst_node.prim_end = static_cast<int32_t>(prim_end);
        }
        else
        {
            dst_node.prim_beg = -1;
            dst_node.prim_end = static_cast<int32_t>(src_node.first_child_or_primitive);
        }
    }

    nodes_.swap(nodes);
    prims_.swap(prims);
}

bool volume::BVH::is_empty() const
{
    return nodes_.empty();
}

boolean volume::BVH::find_closest_intersection(ref<CVec3f> a, ref<CVec3f> b, ref<CVec3f> output_position) const
{
    if(nodes_.empty())
        return false;

    $declare_scope;
    boolean result;

    var nodes = cuj::const_data(std::span{ nodes_ });
    var prims = cuj::const_data(std::span{ prims_ });

    var dir = b - a;
    var inv_dir = 1.0f / dir;
    var final_t = btrc_max_float, t_max = 1.0f;

    cuj::arr<u32, TRAVERSAL_STACK_SIZE> traversal_stack;
    traversal_stack[0] = 0;

    var top = 1;
    $while(top > 0)
    {
        top = top - 1;
        var task_node_idx = traversal_stack[top];
        ref node = nodes[task_node_idx];

        $if(is_leaf_node(node))
        {
            $forrange(i, node.prim_beg, node.prim_end)
            {
                var t = find_closest_intersection(a, dir, t_max, prims[i]);
                $if(t >= 0.0f)
                {
                    final_t = cstd::min(final_t, t);
                    t_max = final_t;
                };
            };
        }
        $else
        {
            var left = nodes[node.prim_end];
            $if(intersect_ray_aabb(left.lower, left.upper, a, inv_dir, t_max))
            {
                traversal_stack[top] = u32(node.prim_end);
                top = top + 1;
            };

            var right = nodes[node.prim_end + 1];
            $if(intersect_ray_aabb(right.lower, right.upper, a, inv_dir, t_max))
            {
                traversal_stack[top] = u32(node.prim_end + 1);
                top = top + 1;
            };
        };
    };

    $if(final_t > 10)
    {
        result = false;
        $exit_scope;
    };

    CUJ_ASSERT(final_t >= 0.0f);
    CUJ_ASSERT(final_t <= 1.0f);
    output_position = a * (1.0f - final_t) + b * final_t;

    result = true;
    return result;
}

volume::BVH::Overlap volume::BVH::get_overlap(ref<CVec3f> position) const
{
    $declare_scope;
    Overlap result;
    result.count = 0;

    if(nodes_.empty())
        return result;

    var nodes = cuj::const_data(std::span{ nodes_ });
    var prims = cuj::const_data(std::span{ prims_ });

    cuj::arr<u32, TRAVERSAL_STACK_SIZE> traversal_stack;
    traversal_stack[0] = 0;

    auto sort_result = [&result]
    {
        $forrange(i, 0, result.count)
        {
            $forrange(j, 0, result.count - 1 - i)
            {
                $if(result.data[j] > result.data[j + 1])
                {
                    var t = result.data[j + 1];
                    result.data[j + 1] = result.data[j];
                    result.data[j] = t;
                };
            };
        };
    };

    var top = 1;
    $while(top > 0)
    {
        top = top - 1;
        ref node = nodes[traversal_stack[top]];

        $if(is_leaf_node(node))
        {
            $forrange(i, node.prim_beg, node.prim_end)
            {
                ref prim = prims[i];
                $if(is_in_prim(position, prim))
                {
                    result.data[result.count] = prim.vol_id;
                    result.count = result.count + 1;
                    $if(result.count >= MAX_OVERLAP_COUNT)
                    {
                        sort_result();
                        $exit_scope;
                    };
                };
            };
        }
        $else
        {
            var left = nodes[node.prim_end];
            $if(is_in_aabb(position, left.lower, left.upper))
            {
                traversal_stack[top] = u32(node.prim_end);
                top = top + 1;
            };

            var right = nodes[node.prim_end + 1];
            $if(is_in_aabb(position, right.lower, right.upper))
            {
                traversal_stack[top] = u32(node.prim_end + 1);
                top = top + 1;
            };
        };
    };

    sort_result();
    return result;
}

boolean volume::BVH::is_leaf_node(ref<CBVHNode> node) const
{
    return node.prim_beg >= 0;
}

boolean volume::BVH::is_in_prim(ref<CVec3f> pos, ref<CBVHPrimitive> prim) const
{
    var op = pos - prim.o;
    var u = dot(op, prim.x_div_x2);
    var v = dot(op, prim.y_div_y2);
    var w = dot(op, prim.z_div_z2);
    return 0.0f <= u & u <= 1.0f &
           0.0f <= v & v <= 1.0f &
           0.0f <= w & w <= 1.0f;
}

boolean volume::BVH::is_in_aabb(ref<CVec3f> pos, ref<CVec3f> lower, ref<CVec3f> upper) const
{
    return lower.x <= pos.x & pos.x <= upper.x &
           lower.y <= pos.y & pos.y <= upper.y &
           lower.z <= pos.z & pos.z <= upper.z;
}

boolean volume::BVH::intersect_ray_aabb(ref<CVec3f> lower, ref<CVec3f> upper, ref<CVec3f> o, ref<CVec3f> inv_d, f32 t_max) const
{
    var n = inv_d * (lower - o);
    var f = inv_d * (upper - o);
    var t0 = max(0.0f, cstd::min(n.x, f.x), cstd::min(n.y, f.y), cstd::min(n.z, f.z));
    var t1 = min(t_max, cstd::max(n.x, f.x), cstd::max(n.y, f.y), cstd::max(n.z, f.z));
    return t0 <= t1;
}

f32 volume::BVH::find_closest_intersection(ref<CVec3f> p, ref<CVec3f> d, f32 t_max, ref<CBVHPrimitive> prim) const
{
    var op = p - prim.o;
    var t0 = 0.0f, t1 = t_max;

    auto process_interval = [&](ref<CVec3f> d_div_d2)
    {
        var opd = dot(op, d_div_d2);
        var dd = dot(d, d_div_d2);
        var nd = (0.0f - opd) / dd;
        var fd = (1.0f - opd) / dd;
        t0 = cstd::max(t0, cstd::min(nd, fd));
        t1 = cstd::min(t1, cstd::max(nd, fd));
    };

    process_interval(prim.x_div_x2);
    process_interval(prim.y_div_y2);
    process_interval(prim.z_div_z2);

    $declare_scope;
    f32 result;

    $if(t0 > t1)
    {
        result = -1;
        $exit_scope;
    };

    $if(t0 > 0)
    {
        result = t0;
        $exit_scope;
    };

    $if(t1 < t_max)
    {
        result = t1;
        $exit_scope;
    };

    result = -1;
    return result;
}

BTRC_END
