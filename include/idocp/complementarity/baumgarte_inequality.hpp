#ifndef IDOCP_BAUMGARTE_INEQUALITY_HPP_
#define IDOCP_BAUMGARTE_INEQUALITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"


namespace idocp {
class BaumgarteInequality {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BaumgarteInequality(const Robot& robot, const double barrier,
                      const double fraction_to_boundary_rate);

  BaumgarteInequality();

  ~BaumgarteInequality();

  BaumgarteInequality(const BaumgarteInequality&) = default;

  BaumgarteInequality& operator=(const BaumgarteInequality&) = default;
 
  BaumgarteInequality(BaumgarteInequality&&) noexcept = default;

  BaumgarteInequality& operator=(BaumgarteInequality&&) noexcept = default;

  bool isFeasible(Robot& robot, const SplitSolution& s);

  void setSlack(Robot& robot, const double dtau, const SplitSolution& s, 
                ConstraintComponentData& data);

  void computePrimalResidual(Robot& robot, const double dtau,  
                             const SplitSolution& s, 
                             ConstraintComponentData& data);

  void augmentDualResidual(Robot& robot, const double dtau, 
                           const SplitSolution& s, 
                           const ConstraintComponentData& data,
                           KKTResidual& kkt_residual);

  void computeSlackDirection(const Robot& robot, const double dtau, 
                             const SplitSolution& s, const SplitDirection& d,
                             ConstraintComponentData& data) const; 

private:
  static constexpr int kDimb = 3;
  static constexpr int kDimf = 5;
  static constexpr int kDimc = 6;
  static constexpr int kDimf_verbose = 7;
  int num_point_contacts_, dimc_; 
  double barrier_, fraction_to_boundary_rate_;
  Eigen::VectorXd Baumgarte_residual_;
  Eigen::MatrixXd dBaumgarte_dq_, dBaumgarte_dv_, dBaumgarte_da_;

};

} // namespace idocp 

#include "idocp/complementarity/baumgarte_inequality.hxx"

#endif // IDOCP_BAUMGARTE_INEQUALITY_HPP_ 