#include <btrc/builtin/renderer/wavefront/common.h>

BTRC_WFPT_BEGIN

cuj::boolean is_path_active(cuj::u32 path_flag)
{
    return (path_flag & PATH_FLAG_ACTIVE) != 0;
}

cuj::boolean is_path_intersected(cuj::u32 path_flag)
{
    return (path_flag & PATH_FLAG_HAS_INTERSECTION) != 0;
}

cuj::boolean is_path_scattered(cuj::u32 path_flag)
{
    return (path_flag & PATH_FLAG_HAS_SCATTERING) != 0;
}

cuj::u32 extract_instance_id(cuj::u32 path_flag)
{
    return path_flag & PATH_FLAG_INSTANCE_ID_MASK;
}

BTRC_WFPT_END
