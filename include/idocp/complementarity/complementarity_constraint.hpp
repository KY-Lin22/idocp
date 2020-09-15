#ifndef IDOCP_COMPLEMENTARITY_CONSTRAINT_HPP_
#define IDOCP_COMPLEMENTARITY_CONSTRAINT_HPP_ 

#include "Eigen/Core"

#include "idocp/constraints/constraint_component_data.hpp"


namespace idocp {

class ComplementarityConstraint {
public:
  ComplementarityConstraint(const int dimc, 
                            const double max_complementarity_violation=1.0e-04, 
                            const double barrier=1.0e-08,
                            const double fraction_to_boundary_rate=0.995);

  ComplementarityConstraint();

  ~ComplementarityConstraint();

  ComplementarityConstraint(const ComplementarityConstraint&) = default;

  ComplementarityConstraint& operator=(const ComplementarityConstraint&) = default;
 
  ComplementarityConstraint(ComplementarityConstraint&&) noexcept = default;

  ComplementarityConstraint& operator=(ComplementarityConstraint&&) noexcept = default;

  bool isFeasible(const ConstraintComponentData& data_inequality1, 
                  const ConstraintComponentData& data_inequality2) const;

  void setSlackAndDual(ConstraintComponentData& data_inequality1, 
                       ConstraintComponentData& data_inequality2,
                       ConstraintComponentData& data_complementarity) const;

  void computeComplementarityResidual(
      const ConstraintComponentData& data_inequality1, 
      const ConstraintComponentData& data_inequality2, 
      ConstraintComponentData& data_complementarity) const;

  void computeDualities(ConstraintComponentData& data_inequality1, 
                        ConstraintComponentData& data_inequality2,
                        ConstraintComponentData& data_complementarity) const;

  void condenseSlackAndDual(const ConstraintComponentData& data_inequality1, 
                            const ConstraintComponentData& data_inequality2, 
                            const ConstraintComponentData& data_complementarity,
                            Eigen::VectorXd& condensed_hessian_diagonal11, 
                            Eigen::VectorXd& condensed_hessian_diagonal12, 
                            Eigen::VectorXd& condensed_hessian_diagonal22,
                            Eigen::VectorXd& condensed_dual1, 
                            Eigen::VectorXd& condensed_dual2);

  void computeDirections(ConstraintComponentData& data_inequality1, 
                         ConstraintComponentData& data_inequality2, 
                         ConstraintComponentData& data_complementarity) const;

  void set_max_complementarity_violation(
      const double max_complementarity_violation);

  void set_barrier(const double barrier);

  void set_fraction_to_boundary_rate(const double fraction_to_boundary_rate);

private:
  int dimc_;
  double max_complementarity_violation_, barrier_, fraction_to_boundary_rate_;
  Eigen::VectorXd dinequality1_, dinequality2_, dcomplementarity_;

};
  
} // namespace idocp

#include "idocp/complementarity/complementarity_constraint.hxx"

#endif // IDOCP_COMPLEMENTARITY_CONSTRAINT_HPP_ 