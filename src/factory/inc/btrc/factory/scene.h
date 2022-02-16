#pragma once

#include <btrc/core/scene.h>
#include <btrc/factory/context.h>

BTRC_FACTORY_BEGIN

RC<Scene> create_scene(const RC<const Node> &scene_root, Context &context);

BTRC_FACTORY_END
