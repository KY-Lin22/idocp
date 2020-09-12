#ifndef IDOCP_FORCE_INEQUALITY_HPP_
#define IDOCP_FORCE_INEQUALITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"


namespace idocp {
class ForceInequality {
private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ForceInequality(const Robot& robot, const double mu, 
                  const double barrier=1.0e-04,
                  const double fraction_to_boundary_rate=0.995);

  ForceInequality();

  ~ForceInequality();

  ForceInequality(const ForceInequality&) = default;

  ForceInequality& operator=(const ForceInequality&) = default;
 
  ForceInequality(ForceInequality&&) noexcept = default;

  ForceInequality& operator=(ForceInequality&&) noexcept = default;

  bool isFeasible(Robot& robot, const SplitSolution& s);

  void setSlackAndDual(Robot& robot, const double dtau, const SplitSolution& s);

  void augmentDualResidual(Robot& robot, const double dtau, 
                           KKTResidual& kkt_residual);

  void condenseSlackAndDual(Robot& robot, const double dtau, 
                            const SplitSolution& s, KKTMatrix& kkt_matrix,
                            KKTResidual& kkt_residual);

  void computeSlackAndDualDirection(Robot& robot, const double dtau, 
                                    const SplitDirection& d); 

  double residualL1Nrom(const Robot& robot, const double dtau, 
                        const SplitSolution& s) const;

  double squaredKKTErrorNorm(const Robot& robot, const double dtau, 
                             const SplitSolution& s) const;

public:
  int num_point_contacts_, dimc_; 
  double mu_, barrier_, fraction_to_boundary_rate_;
  ConstraintComponentData data_;
  Eigen::MatrixXd contact_derivatives_;

};

} // namespace idocp 

#include "idocp/complementarity/force_inequality.hxx"

#endif // IDOCP_FORCE_INEQUALITY_HPP_ 