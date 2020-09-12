#ifndef IDOCP_COMPLEMENTARITY_CONSTRAINTS_BASE_HXX_
#define IDOCP_COMPLEMENTARITY_CONSTRAINTS_BASE_HXX_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

inline ComplementarityConstraintsBase::ComplementarityConstraintsBase(
    const Robot& robot, const double max_complementarity, const double barrier, 
    const double fraction_to_boundary_rate) 
  : max_complementarity_(max_complementarity),
    barrier_(barrier),
    fraction_to_boundary_rate_(fraction_to_boundary_rate) {

}

  ComplementarityConstraintsBase(const Robot& robot, 
                                 const double relax=1.0e-02, 
                                 const double barrier=1.0e-08,
                                 const double fraction_to_boundary_rate=0.995);

  ComplementarityConstraintsBase();

  ~ComplementarityConstraintsBase();

  ComplementarityConstraintsBase(const ComplementarityConstraintsBase&) = default;

  ComplementarityConstraintsBase& operator=(const ComplementarityConstraintsBase&) = default;
 
  ComplementarityConstraintsBase(ComplementarityConstraintsBase&&) noexcept = default;

  ComplementarityConstraintsBase& operator=(ComplementarityConstraintsBase&&) noexcept = default;

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


} // namespace idocp 

#endif // IDOCP_COMPLEMENTARITY_CONSTRAINTS_BASE_HXX_ 