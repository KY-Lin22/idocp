#ifndef IDOCP_MPCC_FUNC_HPP_ 
#define IDOCP_MPCC_FUNC_HPP_

#include "Eigen/Core"
#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {
namespace mpccfunc {

void SetSlackAndDualPositive(const double barrier, 
                             const double max_complementarity_violation,
                             ConstraintComponentData& data_inequality1,
                             ConstraintComponentData& data_inequality2,
                             ConstraintComponentData& data_complementarity);

void ComputeComplementarityResidual(
    const double max_complementarity_violation, 
    const ConstraintComponentData& data_inequality1,
    const ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity);

void ComputeDuality(const double barrier, 
                    ConstraintComponentData& data_inequality1,
                    ConstraintComponentData& data_inequality2, 
                    ConstraintComponentData& data_complementarity);

void ComputeComplementaritySlackAndDualDirection(
    const double barrier, const ConstraintComponentData& data_inequality1,
    const ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity);

void ComputeInequalityDualDirection(
    const double barrier, const ConstraintComponentData& data_inequality1,
    const ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity);


} // namespace mpccfunc
} // namespace idocp

#include "idocp/complementarity/mpcc_func.hxx"

#endif // IDOCP_MPCC_FUNC_HPP_