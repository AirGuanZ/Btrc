#pragma once

#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Texture2DBinaryOperator : public Texture2D
{
public:

    enum class Type
    {
        Add,
        Sub,
        Mul,
        Div
    };

    void set_type(Type type);

    void set_lhs(RC<Texture2D> lhs);

    void set_rhs(RC<Texture2D> rhs);

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec2f> uv) const override;

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<SurfacePoint> spt) const override;

    f32 sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const override;

    f32 sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const override;

private:

    Type type_ = Type::Add;
    BTRC_OBJECT(Texture2D, lhs_);
    BTRC_OBJECT(Texture2D, rhs_);
};

class Texture2DBinaryOperatorCreator : public factory::Creator<Texture2D>
{
public:

    Texture2DBinaryOperatorCreator(std::string name, Texture2DBinaryOperator::Type type);

    std::string get_name() const override;

    RC<Texture2D> create(RC<const factory::Node> node, factory::Context &context) override;

private:

    std::string name_;
    Texture2DBinaryOperator::Type type_;

};

BTRC_BUILTIN_END
