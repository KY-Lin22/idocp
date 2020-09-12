#ifndef IDOCP_BAUMGARTE_INEQUALITY_HPP_
#define IDOCP_BAUMGARTE_INEQUALITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {
class BaumgarteInequality {
private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BaumgarteInequality(const Robot& robot, const double barrier=1.0e-04,
                      const double fraction_to_boundary_rate=0.995);

  BaumgarteInequality();

  ~BaumgarteInequality();

  BaumgarteInequality(const BaumgarteInequality&) = default;

  BaumgarteInequality& operator=(const BaumgarteInequality&) = default;
 
  BaumgarteInequality(BaumgarteInequality&&) noexcept = default;

  BaumgarteInequality& operator=(BaumgarteInequality&&) noexcept = default;

  bool isFeasible(Robot& robot, const SplitSolution& s);

  void setSlackAndDual(Robot& robot, const double dtau, 
                       const SplitSolution& s, ConstraintComponentData& data);

  void computePrimalResidual(Robot& robot, const double dtau,  
                             const SplitSolution& s, 
                             ConstraintComponentData& data);

  void augmentDualResidual(Robot& robot, const double dtau, 
                           const SplitSolution& s, 
                           const ConstraintComponentData& data,
                           KKTResidual& kkt_residual);

public:
  int num_point_contacts_, dimc_; 
  double barrier_, fraction_to_boundary_rate_;
  Eigen::MatrixXd dbaum_da_, dbaum_dv_, dbaum_da_;

};

} // namespace idocp 

#include "idocp/complementarity/baumgarte_inequality.hxx"

#endif // IDOCP_BAUMGARTE_INEQUALITY_HPP_ 