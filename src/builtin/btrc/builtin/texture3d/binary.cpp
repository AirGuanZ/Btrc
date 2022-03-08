#include <btrc/builtin/texture3d/binary.h>

BTRC_BUILTIN_BEGIN

template<BinaryOp3D OP>
void Texture3DBinaryOperator<OP>::set_lhs(RC<Texture3D> lhs)
{
    lhs_ = std::move(lhs);
}

template<BinaryOp3D OP>
void Texture3DBinaryOperator<OP>::set_rhs(RC<Texture3D> rhs)
{
    rhs_ = std::move(rhs);
}

template<BinaryOp3D OP>
CSpectrum Texture3DBinaryOperator<OP>::sample_spectrum_inline(CompileContext &cc, ref<CVec3f> uvw) const
{
    var lhs = lhs_->sample_spectrum(cc, uvw);
    var rhs = rhs_->sample_spectrum(cc, uvw);
    switch(OP)
    {
    case BinaryOp3D::Add: return lhs + rhs;
    case BinaryOp3D::Mul: return lhs * rhs;
    }
    unreachable();
}

template<BinaryOp3D OP>
f32 Texture3DBinaryOperator<OP>::sample_float_inline(CompileContext &cc, ref<CVec3f> uvw) const
{
    var lhs = lhs_->sample_float(cc, uvw);
    var rhs = rhs_->sample_float(cc, uvw);
    switch(OP)
    {
    case BinaryOp3D::Add: return lhs + rhs;
    case BinaryOp3D::Mul: return lhs * rhs;
    }
    unreachable();
}

template<BinaryOp3D OP>
CSpectrum Texture3DBinaryOperator<OP>::sample_spectrum_inline(CompileContext &cc, ref<SurfacePoint> spt) const
{
    var lhs = lhs_->sample_spectrum(cc, spt);
    var rhs = rhs_->sample_spectrum(cc, spt);
    switch(OP)
    {
    case BinaryOp3D::Add: return lhs + rhs;
    case BinaryOp3D::Mul: return lhs * rhs;
    }
    unreachable();
}

template<BinaryOp3D OP>
f32 Texture3DBinaryOperator<OP>::sample_float_inline(CompileContext &cc, ref<SurfacePoint> spt) const
{
    var lhs = lhs_->sample_float(cc, spt);
    var rhs = rhs_->sample_float(cc, spt);
    switch(OP)
    {
    case BinaryOp3D::Add: return lhs + rhs;
    case BinaryOp3D::Mul: return lhs * rhs;
    }
    unreachable();
}

template<BinaryOp3D OP>
CSpectrum Texture3DBinaryOperator<OP>::get_max_spectrum(CompileContext &cc) const
{
    var lmax = lhs_->get_max_spectrum(cc);
    var lmin = lhs_->get_min_spectrum(cc);
    var rmax = rhs_->get_max_spectrum(cc);
    var rmin = rhs_->get_min_spectrum(cc);
    switch(OP)
    {
    case BinaryOp3D::Add: return lmax + rmax;
    case BinaryOp3D::Mul: return max(max(lmax * rmax, lmax * rmin), max(lmin * rmax, lmin * rmin));
    }
    unreachable();
}

template<BinaryOp3D OP>
f32 Texture3DBinaryOperator<OP>::get_max_float(CompileContext &cc) const
{
    var lmax = lhs_->get_max_float(cc);
    var lmin = lhs_->get_min_float(cc);
    var rmax = rhs_->get_max_float(cc);
    var rmin = rhs_->get_min_float(cc);
    switch(OP)
    {
    case BinaryOp3D::Add: return lmax + rmax;
    case BinaryOp3D::Mul: return cstd::max(cstd::max(lmax * rmax, lmax * rmin), cstd::max(lmin * rmax, lmin * rmin));
    }
    unreachable();
}

template<BinaryOp3D OP>
CSpectrum Texture3DBinaryOperator<OP>::get_min_spectrum(CompileContext &cc) const
{
    var lmax = lhs_->get_max_spectrum(cc);
    var lmin = lhs_->get_min_spectrum(cc);
    var rmax = rhs_->get_max_spectrum(cc);
    var rmin = rhs_->get_min_spectrum(cc);
    switch(OP)
    {
    case BinaryOp3D::Add: return lmin + rmin;
    case BinaryOp3D::Mul: return min(min(lmax * rmax, lmax * rmin), min(lmin * rmax, lmin * rmin));
    }
    unreachable();
}

template<BinaryOp3D OP>
f32 Texture3DBinaryOperator<OP>::get_min_float(CompileContext &cc) const
{
    var lmax = lhs_->get_max_float(cc);
    var lmin = lhs_->get_min_float(cc);
    var rmax = rhs_->get_max_float(cc);
    var rmin = rhs_->get_min_float(cc);
    switch(OP)
    {
    case BinaryOp3D::Add: return lmin + rmin;
    case BinaryOp3D::Mul: return cstd::min(cstd::min(lmax * rmax, lmax * rmin), cstd::min(lmin * rmax, lmin * rmin));
    }
    unreachable();
}

template<BinaryOp3D OP>
std::string Texture3DBinaryOperatorCreator<OP>::get_name() const
{
    switch(OP)
    {
    case BinaryOp3D::Add: return "add";
    case BinaryOp3D::Mul: return "mul";
    }
    unreachable();
}

template<BinaryOp3D OP>
RC<Texture3D> Texture3DBinaryOperatorCreator<OP>::create(RC<const factory::Node> node, factory::Context &context)
{
    auto lhs = context.create<Texture3D>(node->child_node("lhs"));
    auto rhs = context.create<Texture3D>(node->child_node("rhs"));
    auto result = newRC<Texture3DBinaryOperator<OP>>();
    result->set_lhs(std::move(lhs));
    result->set_rhs(std::move(rhs));
    return result;
}

template class Texture3DBinaryOperatorCreator<BinaryOp3D::Add>;
template class Texture3DBinaryOperatorCreator<BinaryOp3D::Mul>;

BTRC_BUILTIN_END
