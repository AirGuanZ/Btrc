#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_OPTIX_BEGIN

constexpr uint8_t RAY_MASK_ALL = 0xff;

u32 get_launch_index_x();

u32 get_launch_dimension_x();

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
    u32    miss_sbt_index);

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
    ref<u32> p00);

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
    ref<u32> p01);

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
    ref<u32> p02);

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
    ref<u32> p03);

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
    ref<u32> p04);

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
    ref<u32> p05);

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
    ref<u32> p06);

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
    ref<u32> p07);

void set_payload(i32 index, u32 value);

u32 get_payload(i32 index);

f32 get_ray_tmax();

CVec3f get_ray_o();

CVec3f get_ray_d();

CVec2f get_triangle_barycentrics();

u32 get_primitive_index();

u32 get_instance_id();

BTRC_OPTIX_END
