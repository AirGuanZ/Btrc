#include <btrc/builtin/texture2d/math.h>

BTRC_BUILTIN_BEGIN

void Texture2DBinaryOperator::set_type(Type type)
{
    type_ = type;
}

void Texture2DBinaryOperator::set_lhs(RC<Texture2D> lhs)
{
    lhs_ = std::move(lhs);
}

void Texture2DBinaryOperator::set_rhs(RC<Texture2D> rhs)
{
    rhs_ = std::move(rhs);
}

CSpectrum Texture2DBinaryOperator::sample_spectrum_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    var lhs = lhs_->sample_spectrum(cc, uv);
    var rhs = rhs_->sample_spectrum(cc, uv);
    switch(type_)
    {
    case Type::Add: return lhs + rhs;
    case Type::Sub: return lhs - rhs;
    case Type::Mul: return lhs * rhs;
    case Type::Div: return lhs / rhs;
    }
    unreachable();
}

CSpectrum Texture2DBinaryOperator::sample_spectrum_inline(CompileContext &cc, ref<SurfacePoint> spt) const
{
    var lhs = lhs_->sample_spectrum(cc, spt);
    var rhs = rhs_->sample_spectrum(cc, spt);
    switch(type_)
    {
    case Type::Add: return lhs + rhs;
    case Type::Sub: return lhs - rhs;
    case Type::Mul: return lhs * rhs;
    case Type::Div: return lhs / rhs;
    }
    unreachable();
}

f32 Texture2DBinaryOperator::sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    var lhs = lhs_->sample_float(cc, uv);
    var rhs = rhs_->sample_float(cc, uv);
    switch(type_)
    {
    case Type::Add: return lhs + rhs;
    case Type::Sub: return lhs - rhs;
    case Type::Mul: return lhs * rhs;
    case Type::Div: return lhs / rhs;
    }
    unreachable();
}

f32 Texture2DBinaryOperator::sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const
{
    var lhs = lhs_->sample_float(cc, spt);
    var rhs = rhs_->sample_float(cc, spt);
    switch(type_)
    {
    case Type::Add: return lhs + rhs;
    case Type::Sub: return lhs - rhs;
    case Type::Mul: return lhs * rhs;
    case Type::Div: return lhs / rhs;
    }
    unreachable();
}

Texture2DBinaryOperatorCreator::Texture2DBinaryOperatorCreator(std::string name, Texture2DBinaryOperator::Type type)
    : name_(std::move(name)), type_(type)
{
    
}

std::string Texture2DBinaryOperatorCreator::get_name() const
{
    return name_;
}

RC<Texture2D> Texture2DBinaryOperatorCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto lhs = context.create<Texture2D>(node->child_node("lhs"));
    auto rhs = context.create<Texture2D>(node->child_node("rhs"));
    auto ret = newRC<Texture2DBinaryOperator>();
    ret->set_type(type_);
    ret->set_lhs(std::move(lhs));
    ret->set_rhs(std::move(rhs));
    return ret;
}

BTRC_BUILTIN_END
