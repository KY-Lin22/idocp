#include "idocp/complementarity/force_inequality.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <assert.h>

namespace idocp {

inline ForceInequality::ForceInequality(
    const Robot& robot, const double mu, const double barrier, 
    const double fraction_to_boundary_rate)
  : num_point_contacts_(robot.num_point_contacts()),
    dimc_(6*robot.num_point_contacts()),
    mu_(mu), 
    barrier_(barrier), 
    fraction_to_boundary_rate_(fraction_to_boundary_rate) {
}


inline ForceInequality::ForceInequality() 
  : num_point_contacts_(0),
    dimc_(0),
    mu_(0), 
    barrier_(0), 
    fraction_to_boundary_rate_(0) {
}


inline ForceInequality::~ForceInequality() {
}


inline bool ForceInequality::isFeasible(const Robot& robot, 
                                        const SplitSolution& s) {
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    constexpr int kDimf = 5;
    constexpr int kDimf_verbose = 7;
    if (s.f_verbose.segment<kDimf>(kDimf_verbose*i).minCoeff() < 0) {
      return false;
    }
    const double fx = s.f.coeff(3*i  );
    const double fy = s.f.coeff(3*i+1);
    const double fz = s.f.coeff(3*i+2);
    assert(fx == s.f_verbose(kDimf_verbose*i  )-s.f_verbose(kDimf_verbose*i+1));
    assert(fy == s.f_verbose(kDimf_verbose*i+2)-s.f_verbose(kDimf_verbose*i+3));
    assert(fz == s.f_verbose(kDimf_verbose*i+4));
    const double friction_res = mu_*mu_*fz*fz - fx*fx - fy*fy;
    if (friction_res < 0) {
      return false;
    }
  }
  return true;
}


inline void ForceInequality::setSlackAndDual(const Robot& robot, 
                                             const double dtau, 
                                             const SplitSolution& s,
                                             ConstraintComponentData& data) {
  assert(dtau > 0);
  constexpr int kDimf = 5;
  constexpr int kDimc = 6;
  constexpr int kDimf_verbose = 7;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.slack.segment<kDimf>(kDimc*i) = dtau * s.f_verbose.segment<kDimf>(kDimf_verbose*i);
    const double fx = s.f.coeff(3*i  );
    const double fy = s.f.coeff(3*i+1);
    const double fz = s.f.coeff(3*i+2);
    assert(fx == s.f_verbose(kDimf_verbose*i  )-s.f_verbose(kDimf_verbose*i+1));
    assert(fy == s.f_verbose(kDimf_verbose*i+2)-s.f_verbose(kDimf_verbose*i+3));
    assert(fz == s.f_verbose(kDimf_verbose*i+4));
    data.slack.coeffRef(kDimc*i+kDimf) = dtau * (mu_*mu_*fz*fz-fx*fx-fy*fy);
  }
  pdipmfunc::SetSlackAndDualPositive(barrier_, data.slack, data.dual);
}


inline void ForceInequality::computePrimalResidual(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    ConstraintComponentData& data) {
  assert(dtau > 0);
  constexpr int kDimf = 5;
  constexpr int kDimc = 6;
  constexpr int kDimf_verbose = 7;
  data.residual = data.slack;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.residual.segment<kDimf>(kDimc*i).noalias() -= dtau * s.f_verbose.segment<kDimf>(kDimf_verbose*i);
    const double fx = s.f.coeff(3*i  );
    const double fy = s.f.coeff(3*i+1);
    const double fz = s.f.coeff(3*i+2);
    assert(fx == s.f_verbose(kDimf_verbose*i  )-s.f_verbose(kDimf_verbose*i+1));
    assert(fy == s.f_verbose(kDimf_verbose*i+2)-s.f_verbose(kDimf_verbose*i+3));
    assert(fz == s.f_verbose(kDimf_verbose*i+4));
    const double friction_cone_residual = (mu_*mu_*fz*fz-fx*fx-fy*fy);
    data.residual.coeffRef(kDimc*i+kDimf) -= dtau * friction_cone_residual;
  }
}


inline void ForceInequality::augmentDualResidual(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const ConstraintComponentData& data, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  constexpr int kDimf = 5;
  constexpr int kDimc = 6;
  constexpr int kDimf_verbose = 7;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_residual.lf().segment<kDimf>(kDimf_verbose*i).noalias() -= dtau * data.dual.segment<kDimf>(i*kDimc);
    const double fx = s.f.coeff(3*i  );
    const double fy = s.f.coeff(3*i+1);
    const double fz = s.f.coeff(3*i+2);
    assert(fx == s.f_verbose(kDimf_verbose*i  )-s.f_verbose(kDimf_verbose*i+1));
    assert(fy == s.f_verbose(kDimf_verbose*i+2)-s.f_verbose(kDimf_verbose*i+3));
    assert(fz == s.f_verbose(kDimf_verbose*i+4));
    const double friction_cone_dual = data.dual.coeff(kDimc*i+kDimf);
    kkt_residual.lf().coeffRef(kDimf*i  ) += 2 * dtau * fx * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf*i+1) -= 2 * dtau * fx * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf*i+2) += 2 * dtau * fy * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf*i+3) -= 2 * dtau * fy * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf*i+4) -= 2 * dtau * mu_ * mu_ * fz * friction_cone_dual;
  }
}


} // namespace idocp