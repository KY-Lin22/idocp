#ifndef IDOCP_BAUMGARTE_CONSTRAINT_HPP_
#define IDOCP_BAUMGARTE_CONSTRAINT_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"
#include "idocp/complementarity/contact_force_constraint.hpp"


namespace idocp {
class BaumgarteConstraint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BaumgarteConstraint(const Robot& robot);

  BaumgarteConstraint();

  ~BaumgarteConstraint();

  BaumgarteConstraint(const BaumgarteConstraint&) = default;

  BaumgarteConstraint& operator=(const BaumgarteConstraint&) = default;
 
  BaumgarteConstraint(BaumgarteConstraint&&) noexcept = default;

  BaumgarteConstraint& operator=(BaumgarteConstraint&&) noexcept = default;

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

  template <typename VectorType>
  void augmentCondensedHessian(Robot& robot, const double dtau, 
                               const SplitSolution& s,
                               const Eigen::MatrixBase<VectorType>& diagonal,
                               KKTMatrix& kkt_matrix); 

  template <typename VectorType>
  void augmentComplementarityCondensedHessian(
      Robot& robot, const double dtau, const SplitSolution& s, 
      const ContactForceConstraint& contact_force_constraint, 
      const Eigen::MatrixBase<VectorType>& diagonal, KKTMatrix& kkt_matrix); 

  template <typename VectorType>
  void augmentCondensedResidual(Robot& robot, const double dtau, 
                                const SplitSolution& s, 
                                const Eigen::MatrixBase<VectorType>& residual,
                                KKTResidual& kkt_residual);

  void computeSlackDirection(const Robot& robot, const double dtau, 
                             const SplitSolution& s, const SplitDirection& d,
                             ConstraintComponentData& data) const; 

private:
  static constexpr int kDimf = 5;
  static constexpr int kDimc = 6;
  int num_point_contacts_, dimc_; 
  Eigen::VectorXd Baumgarte_residual_;
  Eigen::MatrixXd dBaumgarte_dq_, dBaumgarte_dv_, dBaumgarte_da_,
                  dBaumgarte_verbose_dq_, dBaumgarte_verbose_dv_, 
                  dBaumgarte_verbose_da_;
  Eigen::Matrix<double, kDimf, 2> Qfr_rsc_;
  Eigen::Matrix<double, kDimf, 1> f_rsc_;

};

} // namespace idocp 

#include "idocp/complementarity/baumgarte_constraint.hxx"

#endif // IDOCP_BAUMGARTE_CONSTRAINT_HPP_ 