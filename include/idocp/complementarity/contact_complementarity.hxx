#include "idocp/complementarity/contact_complementarity.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <algorithm>
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
    contact_force_inequality_(robot, mu),
    baumgarte_inequality_(robot),
    force_data_(dimc_),
    baumgarte_data_(dimc_),
    complementarity_data_(dimc_),
    s_g_(Eigen::VectorXd::Zero(dimc_)), 
    s_h_(Eigen::VectorXd::Zero(dimc_)), 
    g_w_(Eigen::VectorXd::Zero(dimc_)), 
    g_ss_(Eigen::VectorXd::Zero(dimc_)), 
    g_st_(Eigen::VectorXd::Zero(dimc_)), 
    g_tt_(Eigen::VectorXd::Zero(dimc_)),
    condensed_force_residual_(Eigen::VectorXd::Zero(dimc_)),
    condensed_baumgarte_residual_(Eigen::VectorXd::Zero(dimc_)) {
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
    g_tt_(),
    condensed_force_residual_(),
    condensed_baumgarte_residual_() {
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
  for (int i=0; i<dimc_; ++i) {
    if (force_data_.slack.coeff(i) < barrier_) {
      force_data_.slack.coeffRef(i) = barrier_;
    }
    if (baumgarte_data_.slack.coeff(i) < barrier_) {
      baumgarte_data_.slack.coeffRef(i) = barrier_;
    }
  }
  force_data_.dual.array() 
      = barrier_ / force_data_.slack.array() 
          - baumgarte_data_.slack.array() * complementarity_data_.dual.array();
  baumgarte_data_.dual.array() 
      = barrier_ / baumgarte_data_.slack.array() 
          - force_data_ .slack.array() * complementarity_data_.dual.array();
  for (int i=0; i<dimc_; ++i) {
    if (force_data_.dual.coeff(i) < barrier_) {
      force_data_.dual.coeffRef(i) = barrier_;
    }
    if (baumgarte_data_.dual.coeff(i) < barrier_) {
      baumgarte_data_.dual.coeffRef(i) = barrier_;
    }
  }
  complementarity_data_.slack.array() 
      = max_complementarity_violation_ 
          - force_data_.slack.array() * baumgarte_data_.slack.array();
  pdipmfunc::SetSlackAndDualPositive(barrier_, complementarity_data_.slack, 
                                     complementarity_data_.dual);
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
          + force_data_.slack.array() * baumgarte_data_.slack.array() 
                                      * complementarity_data_.dual.array()
          - barrier_;
  baumgarte_data_.duality.array() 
      = baumgarte_data_.slack.array() * baumgarte_data_.dual.array()
          + force_data_.slack.array() * baumgarte_data_.slack.array() 
                                      * complementarity_data_.dual.array()
          - barrier_;
  complementarity_data_.duality.array() 
      = complementarity_data_.slack.array() 
          * complementarity_data_.dual.array() - barrier_;
}


inline void ContactComplementarity::augmentDualResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    KKTResidual& kkt_residual) {
  assert(dtau > 0);
  contact_force_inequality_.augmentDualResidual(robot, dtau, s, force_data_, 
                                                kkt_residual);
  baumgarte_inequality_.augmentDualResidual(robot, dtau, s, baumgarte_data_, 
                                            kkt_residual);
}


inline void ContactComplementarity::condenseSlackAndDual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    KKTMatrix& kkt_matrix, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  s_g_.array() 
      = (force_data_.dual.array()
          + baumgarte_data_.slack.array()*complementarity_data_.dual.array()) 
        / force_data_.slack.array();
  s_h_.array() 
      = (baumgarte_data_.dual.array()
          + force_data_.slack.array()*complementarity_data_.dual.array()) 
        / baumgarte_data_.slack.array();
  g_w_.array() 
      = complementarity_data_.dual.array() / complementarity_data_.slack.array();
  g_ss_.array() 
      = force_data_.slack.array() * g_w_.array() * force_data_.slack.array();
  g_st_.array() 
      = force_data_.slack.array() * g_w_.array() * baumgarte_data_.slack.array();
  g_tt_.array() 
      = baumgarte_data_.slack.array() * g_w_.array() * baumgarte_data_.slack.array();
  g_ss_.noalias() += s_h_;
  g_st_.noalias() += complementarity_data_.dual;
  g_tt_.noalias() += s_g_;
  contact_force_inequality_.augmentCondensedHessian(robot, dtau, s, g_tt_, 
                                                    kkt_matrix);
  baumgarte_inequality_.augmentCondensedHessian(robot, dtau, s, g_ss_, 
                                                kkt_matrix);
  baumgarte_inequality_.augmentComplementarityCondensedHessian(
      robot, dtau, s, contact_force_inequality_, g_st_, kkt_matrix);
  condensed_force_residual_.array() 
      = g_tt_.array() * force_data_.residual.array() 
         + g_st_.array() * baumgarte_data_.residual.array()
         - baumgarte_data_.slack.array() * g_w_.array() 
                                         * complementarity_data_.residual.array()
         + baumgarte_data_.slack.array() * complementarity_data_.duality.array() 
                                         / complementarity_data_.slack.array();
         - force_data_.duality.array() / force_data_.slack.array();
  condensed_baumgarte_residual_.array() 
      = g_st_.array() * force_data_.residual.array() 
         + g_ss_.array() * baumgarte_data_.residual.array()
         - force_data_.slack.array() * g_w_.array() 
                                     * complementarity_data_.residual.array()
         + force_data_.slack.array() * complementarity_data_.duality.array() 
                                     / complementarity_data_.slack.array()
         - baumgarte_data_.duality.array() / baumgarte_data_.slack.array();
  contact_force_inequality_.augmentCondensedResidual(robot, dtau, s, 
                                                     condensed_force_residual_, 
                                                     kkt_residual);
  baumgarte_inequality_.augmentCondensedResidual(robot, dtau, s, 
                                                 condensed_baumgarte_residual_, 
                                                 kkt_residual);
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


inline double ContactComplementarity::maxSlackStepSize() const {
  const double step_size_force 
      = pdipmfunc::FractionToBoundary(dimc_, fraction_to_boundary_rate_, 
                                      force_data_.slack, force_data_.dslack);
  const double step_size_baumgarte
      = pdipmfunc::FractionToBoundary(dimc_, fraction_to_boundary_rate_, 
                                      baumgarte_data_.slack, 
                                      baumgarte_data_.dslack);
  const double step_size_complementarity
      = pdipmfunc::FractionToBoundary(dimc_, fraction_to_boundary_rate_, 
                                      complementarity_data_.slack, 
                                      complementarity_data_.dslack);
  return std::min({step_size_force, step_size_baumgarte, 
                   step_size_complementarity});
}


inline double ContactComplementarity::maxDualStepSize() const {
  const double step_size_force 
      = pdipmfunc::FractionToBoundary(dimc_, fraction_to_boundary_rate_, 
                                      force_data_.dual, force_data_.ddual);
  const double step_size_baumgarte
      = pdipmfunc::FractionToBoundary(dimc_, fraction_to_boundary_rate_, 
                                      baumgarte_data_.dual, 
                                      baumgarte_data_.ddual);
  const double step_size_complementarity
      = pdipmfunc::FractionToBoundary(dimc_, fraction_to_boundary_rate_, 
                                      complementarity_data_.dual, 
                                      complementarity_data_.ddual);
  return std::min({step_size_force, step_size_baumgarte, 
                   step_size_complementarity});
}


inline void ContactComplementarity::updateSlack(const double step_size) { 
  assert(step_size > 0);
  assert(step_size <= 1);
  force_data_.slack.noalias() += step_size * force_data_.dslack;
  baumgarte_data_.slack.noalias() += step_size * baumgarte_data_.dslack;
  complementarity_data_.slack.noalias() += step_size * complementarity_data_.dslack;
}


inline void ContactComplementarity::updateDual(const double step_size) {
  assert(step_size > 0);
  assert(step_size <= 1);
  force_data_.dual.noalias() += step_size * force_data_.ddual;
  baumgarte_data_.dual.noalias() += step_size * baumgarte_data_.ddual;
  complementarity_data_.dual.noalias() += step_size * complementarity_data_.ddual;
}


inline double ContactComplementarity::costSlackBarrier() const {
  double cost = 0;
  cost -= barrier_ * force_data_.slack.array().log().sum();
  cost -= barrier_ * baumgarte_data_.slack.array().log().sum();
  cost -= barrier_ * complementarity_data_.slack.array().log().sum();
  return cost;
}


inline double ContactComplementarity::costSlackBarrier(const double step_size) const {
  assert(step_size > 0);
  assert(step_size <= 1);
  double cost = 0;
  cost -= barrier_ * (force_data_.slack+step_size*force_data_.dslack).array().log().sum();
  cost -= barrier_ * (baumgarte_data_.slack+step_size*baumgarte_data_.dslack).array().log().sum();
  cost -= barrier_ * (complementarity_data_.slack+step_size*complementarity_data_.dslack).array().log().sum();
  return cost;
}


inline double ContactComplementarity::residualL1Nrom() const {
  double error = 0;
  error += force_data_.residual.template lpNorm<1>();
  error += force_data_.duality.template lpNorm<1>();
  error += baumgarte_data_.residual.template lpNorm<1>();
  error += baumgarte_data_.duality.template lpNorm<1>();
  error += complementarity_data_.residual.template lpNorm<1>();
  error += complementarity_data_.duality.template lpNorm<1>();
  return error;
}


inline double ContactComplementarity::computeResidualL1Nrom(
    Robot& robot, const double dtau, const SplitSolution& s) {
  computeResidual(robot, dtau, s);
  return residualL1Nrom();
}


inline double ContactComplementarity::squaredKKTErrorNorm() const {
  double error = 0;
  error += force_data_.residual.squaredNorm();
  error += force_data_.duality.squaredNorm();
  error += baumgarte_data_.residual.squaredNorm();
  error += baumgarte_data_.duality.squaredNorm();
  error += complementarity_data_.residual.squaredNorm();
  error += complementarity_data_.duality.squaredNorm();
  return error;
}


inline double ContactComplementarity::computeSquaredKKTErrorNorm(
    Robot& robot, const double dtau, const SplitSolution& s) { 
  computeResidual(robot, dtau, s);
  return squaredKKTErrorNorm();
}

} // namespace idocp