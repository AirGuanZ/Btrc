#include <btrc/utils/optix/device_funcs.h>

BTRC_OPTIX_BEGIN

namespace
{
    void trace_impl(
        u64      handle,
        CVec3f   ori,
        CVec3f   dir,
        f32      tmin,
        f32      tmax,
        f32      time,
        u32      mask,
        u32      flags,
        u32      sbt_offset,
        u32      sbt_stride,
        u32      miss_sbt_index,
        u32      payload_count,
        ref<u32> p00 = u32(),
        ref<u32> p01 = u32(),
        ref<u32> p02 = u32(),
        ref<u32> p03 = u32(),
        ref<u32> p04 = u32(),
        ref<u32> p05 = u32(),
        ref<u32> p06 = u32(),
        ref<u32> p07 = u32())
    {
        u32 p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20;
        u32 p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31;
        cuj::inline_asm_volatile(
            "call"
            "($0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,"
            "$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31),"
            "_optix_trace_typed_32,"
            "($32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$46,$47,$48,"
            "$49,$50,$51,$52,$53,$54,$55,$56,$57,$58,$59,$60,$61,$62,$63,$64,$65,"
            "$66,$67,$68,$69,$70,$71,$72,$73,$74,$75,$76,$77,$78,$79,$80);",
            {
                { "=r", p00 }, { "=r", p01 }, { "=r", p02 }, { "=r", p03 }, { "=r", p04 },
                { "=r", p05 }, { "=r", p06 }, { "=r", p07 }, { "=r", p08 }, { "=r", p09 },
                { "=r", p10 }, { "=r", p11 }, { "=r", p12 }, { "=r", p13 }, { "=r", p14 },
                { "=r", p15 }, { "=r", p16 }, { "=r", p17 }, { "=r", p18 }, { "=r", p19 },
                { "=r", p20 }, { "=r", p21 }, { "=r", p22 }, { "=r", p23 }, { "=r", p24 },
                { "=r", p25 }, { "=r", p26 }, { "=r", p27 }, { "=r", p28 }, { "=r", p29 },
                { "=r", p30 }, { "=r", p31 }
            },
        {
            { "r", i32(0) }, { "l", handle },
            { "f", ori.x }, { "f", ori.y }, { "f", ori.z },
            { "f", dir.x }, { "f", dir.y }, { "f", dir.z },
            { "f", tmin }, { "f", tmax }, { "f", time }, { "r", mask },
            { "r", flags }, { "r", sbt_offset }, { "r", sbt_stride },
            { "r", miss_sbt_index }, { "r", payload_count },
            { "r", p00 }, { "r", p01 }, { "r", p02 }, { "r", p03 }, { "r", p04 },
            { "r", p05 }, { "r", p06 }, { "r", p07 }, { "r", p08 }, { "r", p09 },
            { "r", p10 }, { "r", p11 }, { "r", p12 }, { "r", p13 }, { "r", p14 },
            { "r", p15 }, { "r", p16 }, { "r", p17 }, { "r", p18 }, { "r", p19 },
            { "r", p20 }, { "r", p21 }, { "r", p22 }, { "r", p23 }, { "r", p24 },
            { "r", p25 }, { "r", p26 }, { "r", p27 }, { "r", p28 }, { "r", p29 },
            { "r", p30 }, { "r", p31 }
        }, {});
    }
}

u32 get_launch_index_x()
{
    u32 index;
    cuj::inline_asm(
        "call ($0), _optix_get_launch_index_x, ();",
        { { "=r", index } }, {}, {});
    return index;
}

u32 get_launch_dimension_x()
{
    u32 result;
    cuj::inline_asm(
        "call ($0), _optix_get_launch_dimension_x, ();",
        { { "=r", result } }, {}, {});
    return result;
}

void trace(
    u64    handle,
    CVec3f ori,
    CVec3f dir,
    f32    tmin,
    f32    tmax,
    f32    time,
    u32    mask,
    u32    flags,
    u32    sbt_offset,
    u32    sbt_stride,
    u32    miss_sbt_index)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        0);
}

void trace(
    u64      handle,
    CVec3f   ori,
    CVec3f   dir,
    f32      tmin,
    f32      tmax,
    f32      time,
    u32      mask,
    u32      flags,
    u32      sbt_offset,
    u32      sbt_stride,
    u32      miss_sbt_index,
    ref<u32> p00)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        1, p00);
}

void trace(
    u64      handle,
    CVec3f   ori,
    CVec3f   dir,
    f32      tmin,
    f32      tmax,
    f32      time,
    u32      mask,
    u32      flags,
    u32      sbt_offset,
    u32      sbt_stride,
    u32      miss_sbt_index,
    ref<u32> p00,
    ref<u32> p01)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        2, p00, p01);
}

void trace(
    u64      handle,
    CVec3f   ori,
    CVec3f   dir,
    f32      tmin,
    f32      tmax,
    f32      time,
    u32      mask,
    u32      flags,
    u32      sbt_offset,
    u32      sbt_stride,
    u32      miss_sbt_index,
    ref<u32> p00,
    ref<u32> p01,
    ref<u32> p02)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        3, p00, p01, p02);
}

void trace(
    u64      handle,
    CVec3f   ori,
    CVec3f   dir,
    f32      tmin,
    f32      tmax,
    f32      time,
    u32      mask,
    u32      flags,
    u32      sbt_offset,
    u32      sbt_stride,
    u32      miss_sbt_index,
    ref<u32> p00,
    ref<u32> p01,
    ref<u32> p02,
    ref<u32> p03)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        4, p00, p01, p02, p03);
}

void trace(
    u64      handle,
    CVec3f   ori,
    CVec3f   dir,
    f32      tmin,
    f32      tmax,
    f32      time,
    u32      mask,
    u32      flags,
    u32      sbt_offset,
    u32      sbt_stride,
    u32      miss_sbt_index,
    ref<u32> p00,
    ref<u32> p01,
    ref<u32> p02,
    ref<u32> p03,
    ref<u32> p04)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        5, p00, p01, p02, p03, p04);
}

void trace(
    u64      handle,
    CVec3f   ori,
    CVec3f   dir,
    f32      tmin,
    f32      tmax,
    f32      time,
    u32      mask,
    u32      flags,
    u32      sbt_offset,
    u32      sbt_stride,
    u32      miss_sbt_index,
    ref<u32> p00,
    ref<u32> p01,
    ref<u32> p02,
    ref<u32> p03,
    ref<u32> p04,
    ref<u32> p05)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        6, p00, p01, p02, p03, p04, p05);
}

void trace(
    u64      handle,
    CVec3f   ori,
    CVec3f   dir,
    f32      tmin,
    f32      tmax,
    f32      time,
    u32      mask,
    u32      flags,
    u32      sbt_offset,
    u32      sbt_stride,
    u32      miss_sbt_index,
    ref<u32> p00,
    ref<u32> p01,
    ref<u32> p02,
    ref<u32> p03,
    ref<u32> p04,
    ref<u32> p05,
    ref<u32> p06)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        7, p00, p01, p02, p03, p04, p05, p06);
}

void trace(
    u64      handle,
    CVec3f   ori,
    CVec3f   dir,
    f32      tmin,
    f32      tmax,
    f32      time,
    u32      mask,
    u32      flags,
    u32      sbt_offset,
    u32      sbt_stride,
    u32      miss_sbt_index,
    ref<u32> p00,
    ref<u32> p01,
    ref<u32> p02,
    ref<u32> p03,
    ref<u32> p04,
    ref<u32> p05,
    ref<u32> p06,
    ref<u32> p07)
{
    trace_impl(
        handle, ori, dir, tmin, tmax, time, mask, flags,
        sbt_offset, sbt_stride, miss_sbt_index,
        8, p00, p01, p02, p03, p04, p05, p06, p07);
}

void set_payload(i32 index, u32 value)
{
    cuj::inline_asm_volatile(
        "call _optix_set_payload, ($0, $1);",
        {}, { { "r", index }, { "r", value } }, {});
}

u32 get_payload(i32 index)
{
    u32 result;
    cuj::inline_asm_volatile(
        "call ($0), _optix_get_payload, ($1);",
        { { "=r", result } }, { { "r", index } }, {});
    return result;
}

f32 get_ray_tmax()
{
    f32 result;
    cuj::inline_asm(
        "call ($0), _optix_get_ray_tmax, ();",
        { { "=f", result } }, {}, {});
    return result;
}

CVec2f get_triangle_barycentrics()
{
    f32 u, v;
    cuj::inline_asm(
        "call ($0, $1), _optix_get_triangle_barycentrics, ();",
        { { "=f", u }, { "=f", v } }, {}, {});
    return CVec2f(u, v);
}

u32 get_primitive_index()
{
    u32 index;
    cuj::inline_asm(
        "call ($0), _optix_read_primitive_idx, ();",
        { { "=r", index } }, {}, {});
    return index;
}

u32 get_instance_id()
{
    u32 index;
    cuj::inline_asm(
        "call ($0), _optix_read_instance_id, ();",
        { { "=r", index } }, {}, {});
    return index;
}

BTRC_OPTIX_END
