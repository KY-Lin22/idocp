#ifndef IDOCP_BAUMGARTE_INEQUALITY_HPP_
#define IDOCP_BAUMGARTE_INEQUALITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"


namespace idocp {
class BaumgarteInequality {
public:
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

private:
  int num_point_contacts_, dimc_; 
  double barrier_, fraction_to_boundary_rate_;
  Eigen::VectorXd Baumgarte_residual_;
  Eigen::MatrixXd dBaumgarte_dq_, dBaumgarte_dv_, dBaumgarte_da_;

};

} // namespace idocp 

#include "idocp/complementarity/baumgarte_inequality.hxx"

#endif // IDOCP_BAUMGARTE_INEQUALITY_HPP_ 