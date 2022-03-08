#pragma once

#include <btrc/core/texture3d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

enum class BinaryOp3D
{
    Add,
    Mul
};

template<BinaryOp3D OP>
class Texture3DBinaryOperator : public Texture3D
{
public:

    void set_lhs(RC<Texture3D> lhs);

    void set_rhs(RC<Texture3D> rhs);

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec3f> uvw) const override;

    f32 sample_float_inline(CompileContext &cc, ref<CVec3f> uvw) const override;

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<SurfacePoint> spt) const override;

    f32 sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const override;

    CSpectrum get_max_spectrum(CompileContext &cc) const override;

    CSpectrum get_min_spectrum(CompileContext &cc) const override;

    f32 get_max_float(CompileContext &cc) const override;

    f32 get_min_float(CompileContext &cc) const override;

private:

    BTRC_OBJECT(Texture3D, lhs_);
    BTRC_OBJECT(Texture3D, rhs_);
};

template<BinaryOp3D OP>
class Texture3DBinaryOperatorCreator : public factory::Creator<Texture3D>
{
public:

    std::string get_name() const override;

    RC<Texture3D> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
