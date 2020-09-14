#ifndef IDOCP_CONTACT_FORCE_INEQUALITY_HPP_
#define IDOCP_CONTACT_FORCE_INEQUALITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"


namespace idocp {
class ContactForceInequality {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 
  ContactForceInequality(const Robot& robot, const double mu);

  ContactForceInequality();

  ~ContactForceInequality();

  ContactForceInequality(const ContactForceInequality&) = default;

  ContactForceInequality& operator=(const ContactForceInequality&) = default;
 
  ContactForceInequality(ContactForceInequality&&) noexcept = default;

  ContactForceInequality& operator=(ContactForceInequality&&) noexcept = default;

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

  void augmentCondensedHessian(const Robot& robot, const double dtau, 
                               const SplitSolution& s,
                               const Eigen::VectorXd& diagonal,
                               KKTMatrix& kkt_matrix); 

  void augmentCondensedResidual(const Robot& robot, const double dtau, 
                                const SplitSolution& s, 
                                const Eigen::VectorXd& condensed_residual, 
                                KKTResidual& kkt_residual);

  void computeSlackDirection(const Robot& robot, const double dtau, 
                             const SplitSolution& s, const SplitDirection& d,
                             ConstraintComponentData& data) const; 

  double mu() const;

private:
  static constexpr int kDimb = 3;
  static constexpr int kDimf = 5;
  static constexpr int kDimc = 6;
  static constexpr int kDimf_verbose = 7;
  int num_point_contacts_, dimc_; 
  double mu_;
  Eigen::Matrix<double, kDimf, 1> f_rsc_;

};

} // namespace idocp 

#include "idocp/complementarity/contact_force_inequality.hxx"

#endif // IDOCP_CONTACT_FORCE_INEQUALITY_HPP_ 