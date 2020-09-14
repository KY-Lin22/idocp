#ifndef IDOCP_BAUMGARTE_INEQUALITY_HPP_
#define IDOCP_BAUMGARTE_INEQUALITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"
#include "idocp/complementarity/contact_force_inequality.hpp"


namespace idocp {
class BaumgarteInequality {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BaumgarteInequality(const Robot& robot);

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

  void augmentCondensedHessian(Robot& robot, const double dtau, 
                               const SplitSolution& s,
                               const Eigen::VectorXd& diagonal,
                               KKTMatrix& kkt_matrix); 

  void augmentComplementarityCondensedHessian(
      Robot& robot, const double dtau, const SplitSolution& s, 
      const ContactForceInequality& contact_force_inequality, 
      const Eigen::VectorXd& diagonal, KKTMatrix& kkt_matrix); 

  void augmentCondensedResidual(Robot& robot, const double dtau, 
                                const SplitSolution& s, 
                                const Eigen::VectorXd& condensed_residual, 
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
  Eigen::VectorXd Baumgarte_residual_;
  Eigen::MatrixXd dBaumgarte_dq_, dBaumgarte_dv_, dBaumgarte_da_,
                  dBaumgarte_verbose_dq_, dBaumgarte_verbose_dv_, 
                  dBaumgarte_verbose_da_;
  Eigen::Matrix<double, 2, kDimf> Qff_rsc_;
  Eigen::Matrix<double, kDimf, 1> f_rsc_;

};

} // namespace idocp 

#include "idocp/complementarity/baumgarte_inequality.hxx"

#endif // IDOCP_BAUMGARTE_INEQUALITY_HPP_ 