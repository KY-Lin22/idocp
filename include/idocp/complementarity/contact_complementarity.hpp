#ifndef IDOCP_CONTACT_COMPLEMENTARITY_HPP_
#define IDOCP_CONTACT_COMPLEMENTARITY_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/ocp/kkt_matrix.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/complementarity/contact_force_inequality.hpp"
#include "idocp/complementarity/baumgarte_inequality.hpp"


namespace idocp {
class ContactComplementarity {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ContactComplementarity(const Robot& robot, const double mu,  
                         const double max_complementarity_violation=1.0e-04, 
                         const double barrier=1.0e-04,
                         const double fraction_to_boundary_rate=0.995);

  ContactComplementarity();

  ~ContactComplementarity();

  ContactComplementarity(const ContactComplementarity&) = default;

  ContactComplementarity& operator=(const ContactComplementarity&) = default;
 
  ContactComplementarity(ContactComplementarity&&) noexcept = default;

  ContactComplementarity& operator=(ContactComplementarity&&) noexcept = default;

  bool isFeasible(Robot& robot, const SplitSolution& s);

  void setSlackAndDual(Robot& robot, const double dtau, const SplitSolution& s);

  void computeResidual(Robot& robot, const double dtau, const SplitSolution& s);

  void augmentDualResidual(Robot& robot, const double dtau, 
                           const SplitSolution& s, KKTResidual& kkt_residual);

  void condenseSlackAndDual(Robot& robot, const double dtau, 
                            const SplitSolution& s, KKTMatrix& kkt_matrix,
                            KKTResidual& kkt_residual);

  void computeSlackAndDualDirection(const Robot& robot, const double dtau, 
                                    const SplitSolution& s, 
                                    const SplitDirection& d); 

  double residualL1Nrom(const Robot& robot, const double dtau, 
                        const SplitSolution& s) const;

  double squaredKKTErrorNorm(Robot& robot, const double dtau, 
                             const SplitSolution& s) const;

private:
  int dimc_; 
  double max_complementarity_violation_, barrier_, fraction_to_boundary_rate_;
  ContactForceInequality contact_force_inequality_;
  BaumgarteInequality baumgarte_inequality_;
  ConstraintComponentData force_data_, baumgarte_data_, complementarity_data_;
  Eigen::VectorXd s_g_, s_h_, g_w_, g_ss_, g_st_, g_tt_, 
                  condensed_force_residual_, condensed_baumgarte_residual_;

};

} // namespace idocp 

#include "idocp/complementarity/contact_complementarity.hxx"

#endif // IDOCP_CONTACT_COMPLEMENTARITY_HPP_ 