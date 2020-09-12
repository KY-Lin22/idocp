#include "idocp/complementarity/force_inequality.hpp"

namespace idocp {

inline ForceInequality::ForceInequality(
    const Robot& robot, const double mu, const double barrier, 
    const double fraction_to_boundary_rate)
  : num_point_contacts_(robot.max_point_contacts()),
    dimc_(robot.max_point_contacts()*7)
    mu_(mu), 
    barrier_(barrier), 
    fraction_to_boundary_rate_(fraction_to_boundary_rate),
    data_(dimc_) {
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
                                             const SplitSolution& s) {
  assert(dtau > 0);
  data_.slack = dtau * s.f;
  pdipmfunc::SetSlackAndDualPositive(barrier_, data_.slack, data_.dual)
}


inline void ForceInequality::augmentDualResidual(
    Robot& robot, const double dtau, KKTResidual& kkt_residual) const {
  kkt_residual.lf()noalias() -= dtau * data.dual;
}


inline void ForceInequality::condenseSlackAndDual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    KKTMatrix& kkt_matrix, KKTResidual& kkt_residual) const {
  kkt_matrix.Qaa().diagonal().tail(dimc_).array()
      += dtau * dtau * data.dual.array() / data.slack.array();
  data.residual = dtau * (amin_-s.a.tail(dimc_)) + data.slack;
  computeDuality(data.slack, data.dual, data.duality);
  kkt_residual.la().tail(dimc_).array() 
      -= dtau * (data.dual.array()*data.residual.array()-data.duality.array()) 
              / data.slack.array();
}


inline void ForceInequality::computeSlackAndDualDirection(
    const Robot& robot, const double dtau, const SplitDirection& d) const {
  data.dslack = dtau * d.da().tail(dimc_) - data.residual;
  computeDualDirection(data.slack, data.dual, data.dslack, data.duality, 
                       data.ddual);
}


inline double ForceInequality::residualL1Nrom(
    const Robot& robot, const double dtau, const SplitSolution& s) const {
  data.residual = dtau * (amin_-s.a.tail(dimc_)) + data.slack;
  return data.residual.lpNorm<1>();
}


inline double ForceInequality::squaredKKTErrorNorm(
    Robot& robot, const double dtau, const SplitSolution& s) const {
  data.residual = dtau * (amin_-s.a.tail(dimc_)) + data.slack;
  computeDuality(data.slack, data.dual, data.duality);
  double error = 0;
  error += data.residual.squaredNorm();
  error += data.duality.squaredNorm();
  return error;
}

} // namespace idocp