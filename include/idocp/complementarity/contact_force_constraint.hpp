#ifndef IDOCP_CONTACT_FORCE_CONSTRAINT_HPP_
#define IDOCP_CONTACT_FORCE_CONSTRAINT_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"


namespace idocp {
class ContactForceConstraint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 
  ContactForceConstraint(const Robot& robot, const double mu);

  ContactForceConstraint();

  ~ContactForceConstraint();

  ContactForceConstraint(const ContactForceConstraint&) = default;

  ContactForceConstraint& operator=(const ContactForceConstraint&) = default;
 
  ContactForceConstraint(ContactForceConstraint&&) noexcept = default;

  ContactForceConstraint& operator=(ContactForceConstraint&&) noexcept = default;

  bool isFeasible(const Robot& robot, const SplitSolution& s);

  void setSlack(const Robot& robot, const double dtau, const SplitSolution& s, 
                ConstraintComponentData& data);

  void computePrimalResidual(const Robot& robot, const double dtau,  
                             const SplitSolution& s, 
                             ConstraintComponentData& data);

  void augmentDualResidual(const Robot& robot, const double dtau, 
                           const SplitSolution& s, 
                           const ConstraintComponentData& data,
                           KKTResidual& kkt_residual);

  template <typename VectorType>
  void augmentCondensedHessian(const Robot& robot, const double dtau, 
                               const SplitSolution& s,
                               const Eigen::MatrixBase<VectorType>& diagonal,
                               KKTMatrix& kkt_matrix); 

  template <typename VectorType>
  void augmentCondensedResidual(const Robot& robot, const double dtau, 
                                const SplitSolution& s, 
                                const Eigen::MatrixBase<VectorType>& residual,
                                KKTResidual& kkt_residual);

  void computeSlackDirection(const Robot& robot, const double dtau, 
                             const SplitSolution& s, const SplitDirection& d,
                             ConstraintComponentData& data) const; 

  void set_mu(const double mu);

  double mu() const;

private:
  static constexpr int kDimf = 5;
  static constexpr int kDimc = 6;
  int num_point_contacts_, dimc_; 
  double mu_;
  Eigen::Matrix<double, kDimf, 1> f_rsc_;

};

} // namespace idocp 

#include "idocp/complementarity/contact_force_constraint.hxx"

#endif // IDOCP_CONTACT_FORCE_CONSTRAINT_HPP_ 