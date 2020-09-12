#ifndef IDOCP_CONTACT_INEQUALITY_HPP_
#define IDOCP_CONTACT_INEQUALITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {
class ContactInequality {
private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ContactInequality(const Robot& robot, const double barrier=1.0e-04,
                    const double fraction_to_boundary_rate=0.995);

  ContactInequality();

  ~ContactInequality();

  ContactInequality(const ContactInequality&) = default;

  ContactInequality& operator=(const ContactInequality&) = default;
 
  ContactInequality(ContactInequality&&) noexcept = default;

  ContactInequality& operator=(ContactInequality&&) noexcept = default;

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
  double barrier_, fraction_to_boundary_rate_;
  Eigen::MatrixXd contact_derivatives_;

};

} // namespace idocp 

#include "idocp/complementarity/contact_inequality.hxx"

#endif // IDOCP_CONTACT_INEQUALITY_HPP_ 