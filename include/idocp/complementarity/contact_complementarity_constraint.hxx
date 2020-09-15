#ifndef IDOCP_CONTACT_COMPLEMENTARITY_CONSTRAINT_HXX_
#define IDOCP_CONTACT_COMPLEMENTARITY_CONSTRAINT_HXX_

#include "idocp/complementarity/contact_complementarity_constraint.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <assert.h>


namespace idocp {

inline ContactComplementarityConstraint::ContactComplementarityConstraint(
    const Robot& robot, const double mu, 
    const double max_complementarity_violation, 
    const double barrier, const double fraction_to_boundary_rate) 
  : dimc_(6*robot.num_point_contacts()),
    max_complementarity_violation_(max_complementarity_violation), 
    barrier_(barrier), 
    fraction_to_boundary_rate_(fraction_to_boundary_rate),
    contact_force_constraint_(robot, mu),
    baumgarte_constraint_(robot),
    complementarity_constraint_(6*robot.num_point_contacts(), 
                                max_complementarity_violation_, barrier_),
    contact_force_data_(dimc_),
    baumgarte_data_(dimc_),
    complementarity_data_(dimc_) {
  try {
    if (mu <= 0) {
      throw std::out_of_range("invalid argment: mu must be positive");
    }
    if (max_complementarity_violation_ <= 0) {
      throw std::out_of_range(
          "invalid argment: max_complementarity_violation must be positive");
    }
    if (max_complementarity_violation_ > 1) {
      throw std::out_of_range(
          "invalid argment: max_complementarity_violation must be less than 1");
    }
    if (barrier <- 0) {
      throw std::out_of_range(
          "invalid argment: barrirer must be positive");
    }
    if (fraction_to_boundary_rate < 0) {
      throw std::out_of_range(
          "invalid argment: fraction_to_boundary_rate must be positive");
    }
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    std::exit(EXIT_FAILURE);
  }
}


inline ContactComplementarityConstraint::ContactComplementarityConstraint() 
  : dimc_(0),
    max_complementarity_violation_(0), 
    barrier_(0), 
    fraction_to_boundary_rate_(0),
    contact_force_constraint_(),
    baumgarte_constraint_(),
    complementarity_constraint_(),
    contact_force_data_(),
    baumgarte_data_(),
    complementarity_data_() {
}


inline ContactComplementarityConstraint::~ContactComplementarityConstraint() {
}


inline bool ContactComplementarityConstraint::isFeasible(
    Robot& robot, const SplitSolution& s) {
  if(!contact_force_constraint_.isFeasible(robot, s)) {
    return false;
  }
  if(!baumgarte_constraint_.isFeasible(robot, s)) {
    return false;
  }
  return true;
}


inline void ContactComplementarityConstraint::setSlackAndDual(
    Robot& robot, const double dtau, const SplitSolution& s) {
  assert(dtau > 0);
  contact_force_constraint_.setSlack(robot, dtau, s, contact_force_data_);
  baumgarte_constraint_.setSlack(robot, dtau, s, baumgarte_data_);
  complementarity_constraint_.setSlackAndDual(contact_force_data_, 
                                              baumgarte_data_, 
                                              complementarity_data_);
}


inline void ContactComplementarityConstraint::computeResidual(
    Robot& robot, const double dtau, const SplitSolution& s) {
  assert(dtau > 0);
  contact_force_constraint_.computePrimalResidual(robot, dtau, s, 
                                                  contact_force_data_);
  baumgarte_constraint_.computePrimalResidual(robot, dtau, s, baumgarte_data_);
  complementarity_constraint_.computeComplementarityResidual(
      contact_force_data_, baumgarte_data_, complementarity_data_);
  complementarity_constraint_.computeDualities(contact_force_data_, 
                                               baumgarte_data_, 
                                               complementarity_data_);
}


inline void ContactComplementarityConstraint::augmentDualResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    KKTResidual& kkt_residual) {
  assert(dtau > 0);
  contact_force_constraint_.augmentDualResidual(robot, dtau, s, 
                                                contact_force_data_, 
                                                kkt_residual);
  baumgarte_constraint_.augmentDualResidual(robot, dtau, s, baumgarte_data_, 
                                            kkt_residual);
}


inline void ContactComplementarityConstraint::condenseSlackAndDual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    KKTMatrix& kkt_matrix, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  complementarity_constraint_.condenseSlackAndDual(
      contact_force_data_, baumgarte_data_, complementarity_data_, 
      condensed_hessian_diagonal_contact_force_, 
      condensed_hessian_diagonal_contact_force_baumgarte_, 
      condensed_hessian_diagonal_baumgarte_, 
      condensed_dual_contact_force_, condensed_dual_baumgarte_);
  contact_force_constraint_.augmentCondensedHessian(
      robot, dtau, s, condensed_hessian_diagonal_contact_force_, kkt_matrix);
  baumgarte_constraint_.augmentCondensedHessian(
      robot, dtau, s, condensed_hessian_diagonal_baumgarte_, kkt_matrix);
  baumgarte_constraint_.augmentComplementarityCondensedHessian(
      robot, dtau, s, contact_force_constraint_, 
      condensed_hessian_diagonal_contact_force_baumgarte_, kkt_matrix);
  contact_force_constraint_.augmentCondensedResidual(
      robot, dtau, s, condensed_dual_contact_force_, kkt_residual);
  baumgarte_constraint_.augmentCondensedResidual(
      robot, dtau, s, condensed_dual_baumgarte_, kkt_residual);
}


inline void ContactComplementarityConstraint::computeSlackAndDualDirection(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const SplitDirection& d) {
  assert(dtau > 0);
  contact_force_constraint_.computeSlackDirection(robot, dtau, s, d, 
                                                  contact_force_data_);
  baumgarte_constraint_.computeSlackDirection(robot, dtau, s, d, 
                                              baumgarte_data_);
  complementarity_constraint_.computeDirections(contact_force_data_, 
                                                baumgarte_data_, 
                                                complementarity_data_);
}


inline double ContactComplementarityConstraint::maxSlackStepSize() const {
  const double step_size_force 
      = pdipmfunc::FractionToBoundary(dimc_, fraction_to_boundary_rate_, 
                                      contact_force_data_.slack, 
                                      contact_force_data_.dslack);
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


inline double ContactComplementarityConstraint::maxDualStepSize() const {
  const double step_size_force 
      = pdipmfunc::FractionToBoundary(dimc_, fraction_to_boundary_rate_, 
                                      contact_force_data_.dual, 
                                      contact_force_data_.ddual);
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


inline void ContactComplementarityConstraint::updateSlack(
    const double step_size) { 
  assert(step_size > 0);
  assert(step_size <= 1);
  contact_force_data_.slack.noalias() += step_size * contact_force_data_.dslack;
  baumgarte_data_.slack.noalias() += step_size * baumgarte_data_.dslack;
  complementarity_data_.slack.noalias() 
      += step_size * complementarity_data_.dslack;
}


inline void ContactComplementarityConstraint::updateDual(
    const double step_size) {
  assert(step_size > 0);
  assert(step_size <= 1);
  contact_force_data_.dual.noalias() += step_size * contact_force_data_.ddual;
  baumgarte_data_.dual.noalias() += step_size * baumgarte_data_.ddual;
  complementarity_data_.dual.noalias() 
      += step_size * complementarity_data_.ddual;
}


inline double ContactComplementarityConstraint::costSlackBarrier() const {
  double cost = 0;
  cost -= barrier_ * contact_force_data_.slack.array().log().sum();
  cost -= barrier_ * baumgarte_data_.slack.array().log().sum();
  cost -= barrier_ * complementarity_data_.slack.array().log().sum();
  return cost;
}


inline double ContactComplementarityConstraint::costSlackBarrier(
    const double step_size) const {
  assert(step_size > 0);
  assert(step_size <= 1);
  double cost = 0;
  cost -= barrier_ * (contact_force_data_.slack
                      + step_size
                          * contact_force_data_.dslack).array().log().sum();
  cost -= barrier_ * (baumgarte_data_.slack
                      + step_size
                          * baumgarte_data_.dslack).array().log().sum();
  cost -= barrier_ * (complementarity_data_.slack
                      + step_size
                          * complementarity_data_.dslack).array().log().sum();
  return cost;
}


inline double ContactComplementarityConstraint::residualL1Nrom() const {
  double error = 0;
  error += contact_force_data_.residual.template lpNorm<1>();
  error += contact_force_data_.duality.template lpNorm<1>();
  error += baumgarte_data_.residual.template lpNorm<1>();
  error += baumgarte_data_.duality.template lpNorm<1>();
  error += complementarity_data_.residual.template lpNorm<1>();
  error += complementarity_data_.duality.template lpNorm<1>();
  return error;
}


inline double ContactComplementarityConstraint::computeResidualL1Nrom(
    Robot& robot, const double dtau, const SplitSolution& s) {
  assert(dtau > 0);
  computeResidual(robot, dtau, s);
  return residualL1Nrom();
}


inline double ContactComplementarityConstraint::squaredKKTErrorNorm() const {
  double error = 0;
  error += contact_force_data_.residual.squaredNorm();
  error += contact_force_data_.duality.squaredNorm();
  error += baumgarte_data_.residual.squaredNorm();
  error += baumgarte_data_.duality.squaredNorm();
  error += complementarity_data_.residual.squaredNorm();
  error += complementarity_data_.duality.squaredNorm();
  return error;
}


inline double ContactComplementarityConstraint::computeSquaredKKTErrorNorm(
    Robot& robot, const double dtau, const SplitSolution& s) { 
  assert(dtau > 0);
  computeResidual(robot, dtau, s);
  return squaredKKTErrorNorm();
}


inline void ContactComplementarityConstraint::set_mu(const double mu) {
  try {
    if (mu <= 0) {
      throw std::out_of_range("invalid argment: mu must be positive");
    }
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    std::exit(EXIT_FAILURE);
  }
  contact_force_constraint_.set_mu(mu);
}


inline void ContactComplementarityConstraint::set_barrier(
    const double barrier) {
  try {
    if (barrier <- 0) {
      throw std::out_of_range(
          "invalid argment: barrirer must be positive");
    }
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    std::exit(EXIT_FAILURE);
  }
  barrier_ = barrier;
  complementarity_constraint_.set_barrier(barrier);
}


inline void ContactComplementarityConstraint::set_max_complementarity_violation(
    const double max_complementarity_violation) {
  try {
    if (max_complementarity_violation_ <= 0) {
      throw std::out_of_range(
          "invalid argment: max_complementarity_violation must be positive");
    }
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    std::exit(EXIT_FAILURE);
  }
  complementarity_constraint_.set_max_complementarity_violation(
      max_complementarity_violation);
}


inline void ContactComplementarityConstraint::set_fraction_to_boundary_rate(
    const double fraction_to_boundary_rate) {
  try {
    if (fraction_to_boundary_rate <= 0) {
      throw std::out_of_range(
          "invalid argment: max_complementarity_violation must be positive");
    }
    if (max_complementarity_violation_ > 1) {
      throw std::out_of_range(
          "invalid argment: max_complementarity_violation must be less than 1");
    }
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    std::exit(EXIT_FAILURE);
  }
  fraction_to_boundary_rate_ = fraction_to_boundary_rate;
}

} // namespace idocp

#endif // IDOCP_CONTACT_COMPLEMENTARITY_CONSTRAINT_HXX_