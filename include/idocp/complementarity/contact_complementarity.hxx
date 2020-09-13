#include "idocp/complementarity/contact_complementarity.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <assert.h>

namespace idocp {

inline ContactComplementarity::ContactComplementarity(
    const Robot& robot, const double mu, 
    const double max_complementarity_violation, 
    const double barrier, const double fraction_to_boundary_rate) 
  : dimc_(6*robot.num_point_contacts()),
    max_complementarity_violation_(max_complementarity_violation), 
    barrier_(barrier), 
    fraction_to_boundary_rate_(fraction_to_boundary_rate),
    contact_force_inequality_(robot, mu, barrier, fraction_to_boundary_rate),
    baumgarte_inequality_(robot, barrier, fraction_to_boundary_rate),
    force_data_(dimc_),
    baumgarte_data_(dimc_),
    complementarity_data_(dimc_),
    s_g_(Eigen::VectorXd::Zero(dimc_)), 
    s_h_(Eigen::VectorXd::Zero(dimc_)), 
    g_w_(Eigen::VectorXd::Zero(dimc_)), 
    g_ss_(Eigen::VectorXd::Zero(dimc_)), 
    g_st_(Eigen::VectorXd::Zero(dimc_)), 
    g_tt_(Eigen::VectorXd::Zero(dimc_)) {
}


inline ContactComplementarity::ContactComplementarity() 
  : dimc_(0),
    max_complementarity_violation_(0), 
    barrier_(0), 
    fraction_to_boundary_rate_(0),
    contact_force_inequality_(),
    baumgarte_inequality_(),
    force_data_(),
    baumgarte_data_(),
    complementarity_data_(),
    s_g_(), 
    s_h_(), 
    g_w_(), 
    g_ss_(), 
    g_st_(), 
    g_tt_() {
}


inline ContactComplementarity::~ContactComplementarity() {
}


inline bool ContactComplementarity::isFeasible(Robot& robot, 
                                               const SplitSolution& s) {
  if(!contact_force_inequality_.isFeasible(robot, s)) {
    return false;
  }
  if(!baumgarte_inequality_.isFeasible(robot, s)) {
    return false;
  }
  return true;
}


inline void ContactComplementarity::setSlackAndDual(Robot& robot, 
                                                    const double dtau, 
                                                    const SplitSolution& s) {
  assert(dtau > 0);
  contact_force_inequality_.setSlack(robot, dtau, s, force_data_);
  baumgarte_inequality_.setSlack(robot, dtau, s, baumgarte_data_);
  assert(force_data_.slack.minCoeff() > 0);
  assert(baumgarte_data_.slack.minCoeff() > 0);
  complementarity_data_.slack.array() 
      = max_complementarity_violation_ 
          - force_data_.slack.array() * baumgarte_data_.slack.array();
  pdipmfunc::SetSlackAndDualPositive(barrier_, complementarity_data_.slack, 
                                     complementarity_data_.dual);
  force_data_.dual.array() 
      = barrier_ / force_data_.slack.array() 
          - baumgarte_data_.slack.array() * complementarity_data_.dual.array();
  for (int i=0; i<force_data_.dual.size(); ++i) {
    while (force_data_.dual.coeff(i) < barrier_) {
      force_data_.dual.coeffRef(i) += barrier_;
    }
  }
  baumgarte_data_.dual.array() 
      = barrier_ / baumgarte_data_.slack.array() 
          - force_data_ .slack.array() * complementarity_data_.dual.array();
  for (int i=0; i<baumgarte_data_.dual.size(); ++i) {
    while (baumgarte_data_.dual.coeff(i) < barrier_) {
      baumgarte_data_.dual.coeffRef(i) += barrier_;
    }
  }
}


inline void ContactComplementarity::computeResidual(
    Robot& robot, const double dtau, const SplitSolution& s) {
  assert(dtau > 0);
  contact_force_inequality_.computePrimalResidual(robot, dtau, s, force_data_);
  baumgarte_inequality_.computePrimalResidual(robot, dtau, s, baumgarte_data_);
  complementarity_data_.residual.array()
      = complementarity_data_.slack.array() 
          + force_data_.slack.array() * baumgarte_data_.slack.array() 
          - max_complementarity_violation_;
  force_data_.duality.array() 
      = force_data_.slack.array() * force_data_.dual.array()
          + force_data_.slack.array() * baumgarte_data_.slack.array() * complementarity_data_.dual.array()
          - barrier_;
  baumgarte_data_.duality.array() 
      = baumgarte_data_.slack.array() * baumgarte_data_.dual.array()
          + force_data_.slack.array() * baumgarte_data_.slack.array() * complementarity_data_.dual.array()
          - barrier_;
  complementarity_data_.duality.array() 
      = complementarity_data_.slack.array() * complementarity_data_.dual.array() - barrier_;
}


inline void ContactComplementarity::augmentDualResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    KKTResidual& kkt_residual) {
  assert(dtau > 0);
  contact_force_inequality_.augmentDualResidual(robot, dtau, s, force_data_, kkt_residual);
  baumgarte_inequality_.augmentDualResidual(robot, dtau, s, baumgarte_data_, kkt_residual);
}


inline void ContactComplementarity::condenseSlackAndDual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    KKTMatrix& kkt_matrix, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  s_g_.array() = (force_data_.dual.array()+baumgarte_data_.slack.array()*complementarity_data_.dual.array()) / force_data_.slack.array();
  s_h_.array() = (baumgarte_data_.dual.array()+force_data_.slack.array()*complementarity_data_.dual.array()) / baumgarte_data_.slack.array();
  g_w_.array() = complementarity_data_.dual.array() / complementarity_data_.slack.array();
  g_ss_.array() = force_data_.slack.array() * g_w_.array() * force_data_.slack.array();
  g_st_.array() = force_data_.slack.array() * g_w_.array() * baumgarte_data_.slack.array();
  g_tt_.array() = baumgarte_data_.slack.array() * g_w_.array() * baumgarte_data_.slack.array();
}


inline void ContactComplementarity::computeSlackAndDualDirection(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const SplitDirection& d) {
  assert(dtau > 0);
  contact_force_inequality_.computeSlackDirection(robot, dtau, s, d, force_data_);
  baumgarte_inequality_.computeSlackDirection(robot, dtau, s, d, baumgarte_data_);
  complementarity_data_.dslack.array() 
      = - force_data_.slack.array() * baumgarte_data_.dslack.array() 
        - baumgarte_data_.slack.array() * force_data_.dslack.array()
        - complementarity_data_.residual.array();
  complementarity_data_.ddual.array() 
      = - g_w_.array() * complementarity_data_.dslack.array()
        - complementarity_data_.duality.array() / complementarity_data_.slack.array();
  force_data_.ddual.array()
      = - s_g_.array() * force_data_.dslack.array()
        - complementarity_data_.dual.array() * baumgarte_data_.dslack.array()
        - complementarity_data_.ddual.array() * baumgarte_data_.slack.array()
        - force_data_.duality.array() / force_data_.slack.array();
  baumgarte_data_.ddual.array()
      = - s_h_.array() * baumgarte_data_.dslack.array()
        - complementarity_data_.dual.array() * force_data_.dslack.array()
        - complementarity_data_.ddual.array() * force_data_.slack.array()
        - baumgarte_data_.duality.array() / baumgarte_data_.slack.array();
}


inline double ContactComplementarity::residualL1Nrom(
    const Robot& robot, const double dtau, const SplitSolution& s) const {
  double error = 0;
  // error += data.residual.squaredNorm();
  // error += data.duality.squaredNorm();
  return error;
  // return data.residual.lpNorm<1>();
}


inline double ContactComplementarity::squaredKKTErrorNorm(
    Robot& robot, const double dtau, const SplitSolution& s) const {
  double error = 0;
  // error += data.residual.squaredNorm();
  // error += data.duality.squaredNorm();
  return error;
}

} // namespace idocp