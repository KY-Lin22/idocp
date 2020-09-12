#include "idocp/complementarity/force_inequality.hpp"

namespace idocp {

inline ForceInequality::ForceInequality(
    const Robot& robot, const double mu, const double barrier, 
    const double fraction_to_boundary_rate)
  : num_point_contacts_(robot.max_point_contacts()),
    dimc_(robot.max_point_contacts()*6)
    mu_(mu), 
    barrier_(barrier), 
    fraction_to_boundary_rate_(fraction_to_boundary_rate) {
}


inline ForceInequality::ForceInequality() {
}


inline ForceInequality::~ForceInequality() {
}


inline bool ForceInequality::isFeasible(Robot& robot, const SplitSolution& s) {
  if (s.f.minCoeff() >= 0) {
    return true;
  }
  return false;
}


inline void ForceInequality::setSlackAndDual(Robot& robot, const double dtau, 
                                             const SplitSolution& s,
                                             ConstraintComponentData& data) {
  assert(dtau > 0);
  constexpr int kDimf = 5;
  constexpr int kDimc = 6;
  constexpr int kDimfg = 7;
  for (int i=0; i<num_point_contacts_; ++i) {
    data.slack.segment<kDimf>(i*kDimc) = dtau * s.f.segment<kDimf>(i*kDimfg);
    const double fx = s.f.coeff(i*kDimf  ) - s.f.coeff(i*kDimf+1);
    const double fy = s.f.coeff(i*kDimf+2) - s.f.coeff(i*kDimf+3);
    const double fz = s.f.coeff(i*kDimf+4);
    data.slack.coeff(i*kDimc+kDimf) = dtau * (mu_*fz*fz-fx*fx-fy*fy);
  }
  pdipmfunc::SetSlackAndDualPositive(barrier_, data.slack, data.dual)
}


inline void ForceInequality::computePrimalResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const ConstraintComponentData& data, Eigen::VectorXd& residual) {
  assert(dtau > 0);
  constexpr int kDimf = 5;
  constexpr int kDimc = 6;
  constexpr int kDimfg = 7;
  for (int i=0; i<num_point_contacts_; ++i) {
    residual.segment<kDimf>(i*kDimc) 
        = data.slack.segment<kDimf>(i*kDimc) - dtau * s.f.segment<kDimf>(i*kDimfg);
    const double fx = s.f.coeff(i*kDimf  ) - s.f.coeff(i*kDimf+1);
    const double fy = s.f.coeff(i*kDimf+2) - s.f.coeff(i*kDimf+3);
    const double fz = s.f.coeff(i*kDimf+4);
    const double friction_cone_residual = dtau * (mu_*fz*fz-fx*fx-fy*fy);
    residual.coeffRef(i*kDimc+kDimf) 
        = data.slack.coeff(i*kDimc+kDimf) - friction_cone_residual;
  }
}


inline void ForceInequality::augmentDualResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const ConstraintComponentData& data, KKTResidual& kkt_residual) const {
  constexpr int kDimf = 5;
  constexpr int kDimc = 6;
  constexpr int kDimfg = 7;
  for (int i=0; i<num_point_contacts_; ++i) {
    kkt_residual.lf().segment<kDimf>(i*kDimf).noalias() 
        -= dtau * data.dual.segment<kDimf>(i*kDimc);
    const double fx = s.f.coeff(i*kDimf  ) - s.f.coeff(i*kDimf+1);
    const double fy = s.f.coeff(i*kDimf+2) - s.f.coeff(i*kDimf+3);
    const double fz = s.f.coeff(i*kDimf+4);
    const double friction_cone_dual = data.dual.coeff(i*kDimc+kDimf);
    kkt_residual.lf().coeffRef(i*kDimf  ) += 2 * dtau * fx * friction_cone_dual;
    kkt_residual.lf().coeffRef(i*kDimf+1) -= 2 * dtau * fx * friction_cone_dual;
    kkt_residual.lf().coeffRef(i*kDimf+2) += 2 * dtau * fy * friction_cone_dual;
    kkt_residual.lf().coeffRef(i*kDimf+3) -= 2 * dtau * fy * friction_cone_dual;
    kkt_residual.lf().coeffRef(i*kDimf+4) -= 2 * dtau * mu_ * fz * friction_cone_dual;
  }
}


// inline double ForceInequality::residualL1Nrom(
//     const Robot& robot, const double dtau, const SplitSolution& s) const {
//   data.residual = dtau * (amin_-s.a.tail(dimc_)) + data.slack;
//   return data.residual.lpNorm<1>();
// }


// inline double ForceInequality::squaredKKTErrorNorm(
//     Robot& robot, const double dtau, const SplitSolution& s) const {
//   data.residual = dtau * (amin_-s.a.tail(dimc_)) + data.slack;
//   computeDuality(data.slack, data.dual, data.duality);
//   double error = 0;
//   error += data.residual.squaredNorm();
//   error += data.duality.squaredNorm();
//   return error;
// }

} // namespace idocp