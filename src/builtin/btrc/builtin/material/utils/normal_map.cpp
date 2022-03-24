#include <btrc/builtin/material/utils/normal_map.h>

BTRC_BUILTIN_BEGIN

void NormalMap::load(const RC<const factory::Node> &parent_node, factory::Context &context)
{
    if(auto n = parent_node->find_child_node("normal"))
        normal_tex_ = context.create<Texture2D>(n);
    else
        normal_tex_.reset();
}

CFrame NormalMap::adjust_frame(CompileContext &cc, const SurfacePoint &spt, const CFrame &frame) const
{
    if(!normal_tex_)
        return frame;
    var local_nor_spec = normal_tex_->sample_spectrum(cc, spt);
    var local_nor = 2.0f * CVec3f(local_nor_spec.r, local_nor_spec.g, local_nor_spec.b) - 1.0f;
    var world_nor = frame.local_to_global(normalize(local_nor));
    return frame.rotate_to_new_z(world_nor);
}

std::vector<RC<Object>> NormalMap::get_dependent_objects()
{
    if(normal_tex_)
        return { normal_tex_ };
    return {};
}

BTRC_BUILTIN_END
