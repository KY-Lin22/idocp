#ifndef IDOCP_COMPLEMENTARITY_CONSTRAINT_HXX_
#define IDOCP_COMPLEMENTARITY_CONSTRAINT_HXX_ 

#include "idocp/complementarity/complementarity_constraint.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <exception>
#include <iostream>
#include <assert.h>

namespace idocp {

inline ComplementarityConstraint::ComplementarityConstraint(
    const int dimc, const double max_complementarity_violation, 
    const double barrier, const double fraction_to_boundary_rate) 
  : dimc_(dimc), 
    max_complementarity_violation_(max_complementarity_violation),
    barrier_(barrier),
    fraction_to_boundary_rate_(fraction_to_boundary_rate),
    dinequality1_(Eigen::VectorXd::Zero(dimc)),
    dinequality2_(Eigen::VectorXd::Zero(dimc)),
    dcomplementarity_(Eigen::VectorXd::Zero(dimc)) {
  try {
    if (dimc < 0) {
      throw std::out_of_range("invalid argment: dimc must not be negative");
    }
    if (max_complementarity_violation_ < 0) {
      throw std::out_of_range(
          "invalid argment: max_complementarity_violation must be positive");
    }
    if (barrier < 0) {
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


inline ComplementarityConstraint::ComplementarityConstraint()
  : dimc_(0), 
    max_complementarity_violation_(0),
    barrier_(0),
    fraction_to_boundary_rate_(0),
    dinequality1_(),
    dinequality2_(),
    dcomplementarity_() {
}


inline ComplementarityConstraint::~ComplementarityConstraint() {
}


inline bool ComplementarityConstraint::isFeasible(
    const ConstraintComponentData& data_inequality1, 
    const ConstraintComponentData& data_inequality2) const {
  assert(data_inequality1.dimc() == dimc_);
  assert(data_inequality2.dimc() == dimc_);
  assert(data_inequality1.slack.minCoeff() > 0);
  assert(data_inequality2.slack.minCoeff() > 0);
  const double max_complementarity 
      = (data_inequality1.slack.array()*data_inequality2.slack.array()).maxCoeff();
  if (max_complementarity < max_complementarity_violation_) {
    return true;
  }
  else {
    return false;
  }
}


inline void ComplementarityConstraint::setSlackAndDual(
    ConstraintComponentData& data_inequality1, 
    ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity) const {
  assert(data_inequality1.dimc() == dimc_);
  assert(data_inequality2.dimc() == dimc_);
  assert(data_complementarity.dimc() == dimc_);
  for (int i=0; i<dimc_; ++i) {
    while (data_inequality1.slack.coeff(i) < barrier_) {
      data_inequality1.slack.coeffRef(i) += barrier_;
    }
    while (data_inequality2.slack.coeff(i) < barrier_) {
      data_inequality2.slack.coeffRef(i) += barrier_;
    }
  }
  data_complementarity.slack.array() 
      = max_complementarity_violation_
          - data_inequality1.slack.array() * data_inequality2.slack.array();
  pdipmfunc::SetSlackAndDualPositive(barrier_, data_complementarity.slack, 
                                     data_complementarity.dual);
  data_inequality1.dual.array() 
      = barrier_ / data_inequality1.slack.array() 
          - data_inequality2.slack.array() * data_complementarity.dual.array();
  data_inequality2.dual.array() 
      = barrier_ / data_inequality2.slack.array() 
          - data_inequality1.slack.array() * data_complementarity.dual.array();
  for (int i=0; i<dimc_; ++i) {
    while (data_inequality1.dual.coeff(i) < barrier_) {
      data_inequality1.dual.coeffRef(i) += barrier_;
    }
    while (data_inequality2.dual.coeff(i) < barrier_) {
      data_inequality2.dual.coeffRef(i) += barrier_;
    }
  }
}


inline void ComplementarityConstraint::computeComplementarityResidual(
    const ConstraintComponentData& data_inequality1, 
    const ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity) const {
  assert(data_inequality1.dimc() == dimc_);
  assert(data_inequality2.dimc() == dimc_);
  assert(data_complementarity.dimc() == dimc_);
  assert(data_inequality1.slack.minCoeff() > 0);
  assert(data_inequality2.slack.minCoeff() > 0);
  data_complementarity.residual.array()
      = data_complementarity.slack.array() 
          + data_inequality1.slack.array() * data_inequality2.slack.array() 
          - max_complementarity_violation_;
}


inline void ComplementarityConstraint::computeDualities(
    ConstraintComponentData& data_inequality1, 
    ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity) const {
  assert(data_inequality1.dimc() == dimc_);
  assert(data_inequality2.dimc() == dimc_);
  assert(data_complementarity.dimc() == dimc_);
  assert(data_inequality1.slack.minCoeff() > 0);
  assert(data_inequality2.slack.minCoeff() > 0);
  assert(data_complementarity.slack.minCoeff() > 0);
  data_inequality1.duality.array()
      = data_inequality1.slack.array() * data_inequality1.dual.array()
          + data_inequality1.slack.array() * data_inequality2.slack.array()
                                           * data_complementarity.dual.array()
          - barrier_; 
  data_inequality2.duality.array()
      = data_inequality2.slack.array() * data_inequality2.dual.array()
          + data_inequality1.slack.array() * data_inequality2.slack.array()
                                           * data_complementarity.dual.array()
          - barrier_; 
  pdipmfunc::ComputeDuality(barrier_, data_complementarity.slack, 
                            data_complementarity.dual, 
                            data_complementarity.duality);
}


inline void ComplementarityConstraint::condenseSlackAndDual(
    const ConstraintComponentData& data_inequality1, 
    const ConstraintComponentData& data_inequality2, 
    const ConstraintComponentData& data_complementarity,
    Eigen::VectorXd& condensed_hessian_diagonal11, 
    Eigen::VectorXd& condensed_hessian_diagonal12, 
    Eigen::VectorXd& condensed_hessian_diagonal22,
    Eigen::VectorXd& condensed_dual1, 
    Eigen::VectorXd& condensed_dual2) {
  assert(data_inequality1.dimc() == dimc_);
  assert(data_inequality2.dimc() == dimc_);
  assert(data_complementarity.dimc() == dimc_);
  assert(condensed_hessian_diagonal11.size() == dimc_);
  assert(condensed_hessian_diagonal12.size() == dimc_);
  assert(condensed_hessian_diagonal22.size() == dimc_);
  assert(condensed_dual1.size() == dimc_);
  assert(condensed_dual2.size() == dimc_);
  dinequality1_.array() 
      = (data_inequality1.dual.array() 
          + data_inequality2.slack.array() * data_complementarity.dual.array())
        / data_inequality1.slack.array();
  dinequality2_.array() 
      = (data_inequality2.dual.array() 
          + data_inequality1.slack.array() * data_complementarity.dual.array())
        / data_inequality2.slack.array();
  dcomplementarity_.array() = data_complementarity.dual.array() 
                                / data_complementarity.slack.array();
  condensed_hessian_diagonal11.array() 
      = data_inequality2.slack.array() * dcomplementarity_.array() 
                                       * data_inequality2.slack.array();
  condensed_hessian_diagonal12.array() 
      = data_inequality1.slack.array() * dcomplementarity_.array() 
                                       * data_inequality2.slack.array();
  condensed_hessian_diagonal22.array() 
      = data_inequality1.slack.array() * dcomplementarity_.array() 
                                       * data_inequality1.slack.array();
  condensed_hessian_diagonal11.noalias() += dinequality1_;
  condensed_hessian_diagonal12.noalias() += data_complementarity.dual;
  condensed_hessian_diagonal22.noalias() += dinequality2_;
  condensed_dual1.array() 
      = condensed_hessian_diagonal11.array() * data_inequality1.residual.array()
          + condensed_hessian_diagonal12.array() * data_inequality2.residual.array()
          - data_inequality2.slack.array() * dcomplementarity_.array() 
                                           * data_complementarity.residual.array()
          + data_inequality2.slack.array() * data_complementarity.duality.array()
                                           / data_complementarity.slack.array()
          - data_inequality1.duality.array() / data_inequality1.slack.array();
  condensed_dual2.array() 
      = condensed_hessian_diagonal12.array() * data_inequality1.residual.array()
          + condensed_hessian_diagonal22.array() * data_inequality2.residual.array()
          - data_inequality1.slack.array() * dcomplementarity_.array() 
                                           * data_complementarity.residual.array()
          + data_inequality1.slack.array() * data_complementarity.duality.array()
                                           / data_complementarity.slack.array()
          - data_inequality2.duality.array() / data_inequality2.slack.array();
}


inline void ComplementarityConstraint::computeDirections(
    ConstraintComponentData& data_inequality1, 
    ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity) const {
  assert(data_inequality1.dimc() == dimc_);
  assert(data_inequality2.dimc() == dimc_);
  assert(data_complementarity.dimc() == dimc_);
  data_complementarity.dslack.array()
      = - data_inequality1.slack.array() * data_inequality2.dslack.array()
        - data_inequality2.slack.array() * data_inequality1.dslack.array()
        - data_complementarity.residual.array();
  data_complementarity.ddual.array()
      = - data_complementarity.dual.array() 
            * data_complementarity.dslack.array()
            / data_complementarity.slack.array()
        - data_complementarity.duality.array()
            / data_complementarity.slack.array();
  data_inequality1.ddual.array()
      = - dinequality1_.array() * data_inequality1.dslack.array()
        - data_complementarity.dual.array() * data_inequality2.dslack.array()
        - data_inequality2.slack.array() * data_complementarity.ddual.array()
        - data_inequality1.duality.array() / data_inequality1.slack.array();
  data_inequality2.ddual.array()
      = - data_complementarity.dual.array() * data_inequality1.dslack.array()
        - dinequality2_.array() * data_inequality2.dslack.array()
        - data_inequality1.slack.array() * data_complementarity.ddual.array()
        - data_inequality2.duality.array() / data_inequality2.slack.array();
}


inline void ComplementarityConstraint::set_max_complementarity_violation(
    const double max_complementarity_violation) {
  try {
    if (max_complementarity_violation_ < 0) {
      throw std::out_of_range(
          "invalid argment: max_complementarity_violation must be positive");
    }
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    std::exit(EXIT_FAILURE);
  }
  max_complementarity_violation_ = max_complementarity_violation;
}


inline void ComplementarityConstraint::set_barrier(const double barrier) {
  try {
    if (barrier < 0) {
      throw std::out_of_range("invalid argment: barrirer must be positive");
    }
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    std::exit(EXIT_FAILURE);
  }
  barrier_ = barrier;
}


inline void ComplementarityConstraint::set_fraction_to_boundary_rate(
    const double fraction_to_boundary_rate) {
  try {
    if (fraction_to_boundary_rate < 0) {
      throw std::out_of_range(
          "invalid argment: fraction_to_boundary_rate must be positive");
    }
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    std::exit(EXIT_FAILURE);
  }
  fraction_to_boundary_rate_ = fraction_to_boundary_rate;
}

} // namespace idocp


#endif // IDOCP_COMPLEMENTARITY_CONSTRAINT_HXX_ 