#ifndef IDOCP_CONTACT_DYNAMICS_HPP_
#define IDOCP_CONTACT_DYNAMICS_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {
class ContactDynamics {
private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ContactDynamics(const Robot& robot, const double mu, const double eps=1.0e-02, 
                  const double barrier=1.0e-08, 
                  const double fraction_to_boundary_rate=0.995);

  ContactDynamics();

  ~ContactDynamics();

  ContactDynamics(const ContactDynamics&) = default;

  ContactDynamics& operator=(const ContactDynamics&) = default;
 
  ContactDynamics(ContactDynamics&&) noexcept = default;

  ContactDynamics& operator=(ContactDynamics&&) noexcept = default;

  bool useKinematics() const override;

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
  int num_point_contacts_, dimf_; 
  double mu_, eps_, barrier_, fraction_to_boundary_rate_;
  Eigen::VectorXd contact_residual_, slack_, dual_, residual_, duality_;
  Eigen::MatrixXd contact_derivatives_;

};

} // namespace idocp 

#include "idocp/ocp/contact_dynamics.hxx"

#endif // IDOCP_CONTACT_DYNAMICS_HPP_ 