#ifndef IDOCP_FORCE_INEQUALITY_HPP_
#define IDOCP_FORCE_INEQUALITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"


namespace idocp {
class ForceInequality {
public:

  ForceInequality(const Robot& robot, const double mu, 
                  const double barrier=1.0e-04,
                  const double fraction_to_boundary_rate=0.995);

  ForceInequality();

  ~ForceInequality();

  ForceInequality(const ForceInequality&) = default;

  ForceInequality& operator=(const ForceInequality&) = default;
 
  ForceInequality(ForceInequality&&) noexcept = default;

  ForceInequality& operator=(ForceInequality&&) noexcept = default;

  bool isFeasible(const Robot& robot, const SplitSolution& s);

  void setSlackAndDual(const Robot& robot, const double dtau, 
                       const SplitSolution& s, ConstraintComponentData& data);

  void computePrimalResidual(const Robot& robot, const double dtau,  
                             const SplitSolution& s, 
                             ConstraintComponentData& data);

  void augmentDualResidual(const Robot& robot, const double dtau, 
                           const SplitSolution& s, 
                           const ConstraintComponentData& data,
                           KKTResidual& kkt_residual);

private:
  int num_point_contacts_, dimc_; 
  double mu_, barrier_, fraction_to_boundary_rate_;

};

} // namespace idocp 

#include "idocp/complementarity/force_inequality.hxx"

#endif // IDOCP_FORCE_INEQUALITY_HPP_ 