#ifndef IDOCP_COMPLEMENTARITY_CONSTRAINT_HPP_
#define IDOCP_COMPLEMENTARITY_CONSTRAINT_HPP_ 

#include "Eigen/Core"

#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {

class ComplementarityConstraint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ComplementarityConstraint(const int dimc, 
                            const double max_complementarity_violation, 
                            const double barrier);

  ComplementarityConstraint();

  ~ComplementarityConstraint();

  ComplementarityConstraint(const ComplementarityConstraint&) = default;

  ComplementarityConstraint& operator=(const ComplementarityConstraint&) = default;
 
  ComplementarityConstraint(ComplementarityConstraint&&) noexcept = default;

  ComplementarityConstraint& operator=(ComplementarityConstraint&&) noexcept = default;

  bool isFeasible(const ConstraintComponentData& data_constraint1, 
                  const ConstraintComponentData& data_constraint2) const;

  void setSlackAndDual(ConstraintComponentData& data_constraint1, 
                       ConstraintComponentData& data_constraint2,
                       ConstraintComponentData& data_complementarity) const;

  void computeComplementarityResidual(
      const ConstraintComponentData& data_constraint1, 
      const ConstraintComponentData& data_constraint2, 
      ConstraintComponentData& data_complementarity) const;

  void computeDualities(ConstraintComponentData& data_constraint1, 
                        ConstraintComponentData& data_constraint2,
                        ConstraintComponentData& data_complementarity) const;

  void condenseSlackAndDual(const ConstraintComponentData& data_constraint1, 
                            const ConstraintComponentData& data_constraint2, 
                            const ConstraintComponentData& data_complementarity,
                            Eigen::VectorXd& condensed_hessian_diagonal11, 
                            Eigen::VectorXd& condensed_hessian_diagonal12, 
                            Eigen::VectorXd& condensed_hessian_diagonal22,
                            Eigen::VectorXd& condensed_dual1, 
                            Eigen::VectorXd& condensed_dual2);

  void computeDirections(ConstraintComponentData& data_constraint1, 
                         ConstraintComponentData& data_constraint2, 
                         ConstraintComponentData& data_complementarity) const;

  void set_max_complementarity_violation(
      const double max_complementarity_violation);

  void set_barrier(const double barrier);

private:
  int dimc_;
  double max_complementarity_violation_, barrier_;
  Eigen::VectorXd dconstraint1_, dconstraint2_, dcomplementarity_;

};
  
} // namespace idocp

#include "idocp/complementarity/complementarity_constraint.hxx"

#endif // IDOCP_COMPLEMENTARITY_CONSTRAINT_HPP_ 