#ifndef IDOCP_CONTACT_COMPLEMENTARITY_HPP_
#define IDOCP_CONTACT_COMPLEMENTARITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/constraint_component_data.hpp"


namespace idocp {
class ContactComplementarity {
private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ContactComplementarity(const Robot& robot, const double mu,  
                         const double max_complementarity=1.0e-02, 
                         const double barrier=1.0e-04,
                         const double fraction_to_boundary_rate=0.995);

  ContactComplementarity();

  ~ContactComplementarity();

  ContactComplementarity(const ContactComplementarity&) = default;

  ContactComplementarity& operator=(const ContactComplementarity&) = default;
 
  ContactComplementarity(ContactComplementarity&&) noexcept = default;

  ContactComplementarity& operator=(ContactComplementarity&&) noexcept = default;

  bool useKinematics() const override;

  bool isFeasible(Robot& robot, const SplitSolution& s);

  void setSlackAndDual(Robot& robot, const double dtau, const SplitSolution& s);

  void augmentDualResidual(Robot& robot, const double dtau, 
                           KKTResidual& kkt_residual);

  void condenseSlackAndDual(Robot& robot, const double dtau, 
                            const SplitSolution& s, KKTMatrix& kkt_matrix,
                            KKTResidual& kkt_residual);

  void computeSlackAndDualDirection(const Robot& robot, const double dtau, 
                                    const SplitDirection& d); 

  double residualL1Nrom(const Robot& robot, const double dtau, 
                        const SplitSolution& s) const;

  double squaredKKTErrorNorm(Robot& robot, const double dtau, 
                             const SplitSolution& s) const;

public:
  int num_point_contacts_, dimf_, dimc_; 
  double mu_, max_complementarity_, barrier_, fraction_to_boundary_rate_;
  ConstraintComponentData f_data_, h_data_, c_data_;
  Eigen::VectorXd contact_residual_;
  Eigen::MatrixXd contact_derivatives_;

};

} // namespace idocp 

#include "idocp/ocp/contact_complementarity.hxx"

#endif // IDOCP_CONTACT_COMPLEMENTARITY_HPP_ 