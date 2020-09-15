#ifndef IDOCP_CONTACT_COMPLEMENTARITY_CONSTRAINT_HPP_
#define IDOCP_CONTACT_COMPLEMENTARITY_CONSTRAINT_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/ocp/kkt_matrix.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/complementarity/contact_force_constraint.hpp"
#include "idocp/complementarity/baumgarte_constraint.hpp"
#include "idocp/complementarity/complementarity_constraint.hpp"


namespace idocp {
class ContactComplementarityConstraint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ContactComplementarityConstraint(
      const Robot& robot, const double mu,  
      const double max_complementarity_violation=1.0e-04, 
      const double barrier=1.0e-08, 
      const double fraction_to_boundary_rate=0.995);

  ContactComplementarityConstraint();

  ~ContactComplementarityConstraint();

  ContactComplementarityConstraint(const ContactComplementarityConstraint&) = default;

  ContactComplementarityConstraint& operator=(const ContactComplementarityConstraint&) = default;
 
  ContactComplementarityConstraint(ContactComplementarityConstraint&&) noexcept = default;

  ContactComplementarityConstraint& operator=(ContactComplementarityConstraint&&) noexcept = default;

  bool isFeasible(Robot& robot, const SplitSolution& s);

  void setSlackAndDual(Robot& robot, const double dtau, const SplitSolution& s);

  void computeResidual(Robot& robot, const double dtau, const SplitSolution& s);

  void augmentDualResidual(Robot& robot, const double dtau, 
                           const SplitSolution& s, KKTResidual& kkt_residual);

  void condenseSlackAndDual(Robot& robot, const double dtau, 
                            const SplitSolution& s, KKTMatrix& kkt_matrix,
                            KKTResidual& kkt_residual);

  void computeSlackAndDualDirection(const Robot& robot, const double dtau, 
                                    const SplitSolution& s, 
                                    const SplitDirection& d); 
 
  double maxSlackStepSize() const;

  double maxDualStepSize() const;

  void updateSlack(const double step_size);

  void updateDual(const double step_size);

  double costSlackBarrier() const;

  double costSlackBarrier(const double step_size) const;

  double residualL1Nrom() const;

  double computeResidualL1Nrom(Robot& robot, const double dtau, 
                               const SplitSolution& s);

  double squaredKKTErrorNorm() const;

  double computeSquaredKKTErrorNorm(Robot& robot, const double dtau, 
                                    const SplitSolution& s);

  void set_mu(const double mu);

  void set_max_complementarity_violation(
      const double max_complementarity_violation);

  void set_barrier(const double barrier);

  void set_fraction_to_boundary_rate(const double fraction_to_boundary_rate);

private:
  int dimc_; 
  double max_complementarity_violation_, barrier_, fraction_to_boundary_rate_;
  ContactForceConstraint contact_force_constraint_;
  BaumgarteConstraint baumgarte_constraint_;
  ComplementarityConstraint complementarity_constraint_;
  ConstraintComponentData contact_force_data_, baumgarte_data_, 
                          complementarity_data_;
  Eigen::VectorXd condensed_hessian_diagonal_contact_force_, 
                  condensed_hessian_diagonal_baumgarte_,
                  condensed_hessian_diagonal_contact_force_baumgarte_,
                  condensed_dual_contact_force_, condensed_dual_baumgarte_;

};

} // namespace idocp 

#include "idocp/complementarity/contact_complementarity_constraint.hxx"

#endif // IDOCP_CONTACT_COMPLEMENTARITY_CONSTRAINT_HPP_