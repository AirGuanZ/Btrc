#pragma once

#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class NormalMap : public Object
{
public:

    void load(const RC<const factory::Node> &parent_node, factory::Context &context);

    CFrame adjust_frame(CompileContext &cc, const SurfacePoint &spt, const CFrame &frame) const;

    std::vector<RC<Object>> get_dependent_objects() override;

private:
    
    RC<Texture2D> normal_tex_;
};

BTRC_BUILTIN_END
