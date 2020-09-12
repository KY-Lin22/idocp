#ifndef IDOCP_COMPLEMENTARITY_CONSTRAINTS_BASE_HPP_
#define IDOCP_COMPLEMENTARITY_CONSTRAINTS_BASE_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

template <typename Derived>
class ComplementarityConstraintsBase {
private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ComplementarityConstraintsBase(const Robot& robot, 
                                 const double barrier, 
                                 const double fraction_to_boundary_rate);

  ComplementarityConstraintsBase();

  ~ComplementarityConstraintsBase();

  ComplementarityConstraintsBase(const ComplementarityConstraintsBase&) = default;

  ComplementarityConstraintsBase& operator=(const ComplementarityConstraintsBase&) = default;
 
  ComplementarityConstraintsBase(ComplementarityConstraintsBase&&) noexcept = default;

  ComplementarityConstraintsBase& operator=(ComplementarityConstraintsBase&&) noexcept = default;

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
  int dimc_; 
  double max_complementarity_, barrier_, fraction_to_boundary_rate_;
  Eigen::VectorXd slack_, dual_, residual_, duality_;

};

} // namespace idocp 

#include "idocp/ocp/complementarity_constraints_base.hxx"

#endif // IDOCP_COMPLEMENTARITY_CONSTRAINTS_BASE_HPP_ 