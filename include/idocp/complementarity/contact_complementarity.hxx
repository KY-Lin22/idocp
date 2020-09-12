#include "idocp/ocp/contact_complementarity.hpp"


namespace idocp {

inline ContactComplementarity::ContactComplementarity(
    const Robot& robot, const double mu, const double max_complementarity, 
    const double barrier, const double fraction_to_boundary_rate) 
  : num_point_contacts_(robot.max_point_contacts()),
    dimf_(robot.max_dimf()),
    dimc_(robot.max_point_contacts()*7)
    mu_(mu), 
    max_complementarity_(max_complementarity), 
    barrier_(barrier), 
    fraction_to_boundary_rate_(fraction_to_boundary_rate),
    g_data_(dimc_),
    h_data_(dimc_),
    gh_data_(dimc_),
    contact_residual_(Eigen::VectorXd::Zero(robot.max_dimf())),
    contact_derivatives_(Eigen::MatrixXd::Zero(robot.max_dimf(), robot.dimv())) {
}


inline ContactComplementarity::ContactComplementarity() {
}


inline ContactComplementarity::~ContactComplementarity() {
}


inline bool ContactComplementarity::useKinematics() const {
  return true;
}


inline bool ContactComplementarity::isFeasible(Robot& robot, 
                                               const SplitSolution& s) {
  robot.updateKinematics(s.q. s.v, s.a);
  robot.computeBaumgarteResidual(contact_residual_);
  if (contact_residual_.minCoeff() < 0) {
    return false;
  }
  if (s.fxplus < 0) {
    return false;
  }
  if (s.fxminus < 0) {
    return false;
  }
  if (s.fyplus < 0) {
    return false;
  }
  if (s.fyminus < 0) {
    return false;
  }
  if (s.fz < 0) {
    return false;
  }
  return true;
}


inline void ContactComplementarity::setSlackAndDual(Robot& robot, 
                                                    const double dtau, 
                                                    const SplitSolution& s) {
  assert(dtau > 0);
  data.slack = dtau * (s.a.tail(dimc_)-amin_);
  setSlackAndDualPositive(data.slack, data.dual);
}


inline void ContactComplementarity::augmentDualResidual(
    Robot& robot, const double dtau, KKTResidual& kkt_residual) const {
  kkt_residual.la().tail(dimc_).noalias() -= dtau * data.dual;
}max_complementarity


inline void ContactComplementarity::condenseSlackAndDual(
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


inline void ContactComplementarity::computeSlackAndDualDirection(
    const Robot& robot, const double dtau, const SplitDirection& d) const {
  data.dslack = dtau * d.da().tail(dimc_) - data.residual;
  computeDualDirection(data.slack, data.dual, data.dslack, data.duality, 
                       data.ddual);
}


inline double ContactComplementarity::residualL1Nrom(
    const Robot& robot, const double dtau, const SplitSolution& s) const {
  data.residual = dtau * (amin_-s.a.tail(dimc_)) + data.slack;
  return data.residual.lpNorm<1>();
}


inline double ContactComplementarity::squaredKKTErrorNorm(
    Robot& robot, const double dtau, const SplitSolution& s) const {
  data.residual = dtau * (amin_-s.a.tail(dimc_)) + data.slack;
  computeDuality(data.slack, data.dual, data.duality);
  double error = 0;
  error += data.residual.squaredNorm();
  error += data.duality.squaredNorm();
  return error;
}

} // namespace idocp