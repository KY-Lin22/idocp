#ifndef IDOCP_MPCC_FUNC_HXX_
#define IDOCP_MPCC_FUNC_HXX_

#include "idocp/complementarity/mpcc_func.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <assert.h>

namespace idocp {
namespace mpccfunc {

inline void SetSlackAndDualPositive(
    const double barrier, const double max_complementarity_violation,
    ConstraintComponentData& data_inequality1, 
    ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity) {
  assert(barrier > 0);
  assert(max_complementarity_violation > 0);
  for (int i=0; i<data_inequality1.slack.size(); ++i) {
    while (data_inequality1.slack.coeff(i) < barrier) {
      data_inequality1.slack.coeffRef(i) += barrier;
    }
    while (data_inequality2.slack.coeff(i) < barrier_) {
      data_inequality2.slack.coeffRef(i) += barrier_;
    }
  }
  data_complementarity.slack.array() 
      = max_complementarity_violation
          - data_inequality1.slack.array() * data_inequality2.slack.array();

  pdipmfunc::SetSlackAndDualPositive(barrier, data_complementarity.slack, 
                                     data_complementarity.dual);
  data_inequality1.dual.array() 
      = barrier / data_inequality1.slack.array() 
          - data_inequality2.slack.array() * data_complementarity.dual.array();
  data_inequality2.dual.array() 
      = barrier / data_inequality2.slack.array() 
          - data_inequality1.slack.array() * data_complementarity.dual.array();
  for (int i=0; i<data_inequality1.slack.size(); ++i) {
    while (data_inequality1.dual.coeff(i) < barrier) {
      data_inequality1.dual.coeffRef(i) += barrier;
    }
    while (data_inequality2.dual.coeff(i) < barrier) {
      data_inequality2.dual.coeffRef(i) += barrier;
    }
  }
}


inline void ComputeComplementarityResidual(
    const double max_complementarity_violation, 
    const ConstraintComponentData& data_inequality1,
    const ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity) {
  assert(max_complementarity_violation > 0);
  data_complementarity.residual.array()
      = data_complementarity.slack.array() 
          + data_inequality1.slack.array() * data_inequality2.slack.array() 
          - max_complementarity_violation;
}


inline void ComputeDuality(const double barrier, 
                           ConstraintComponentData& data_inequality1,
                           ConstraintComponentData& data_inequality2, 
                           ConstraintComponentData& data_complementarity) {
  assert(barrier > 0);
  data_inequality1.duality.array()
      = data_inequality1.slack.array() * data_inequality1.dual.array()
          + data_inequality1.slack.array() * data_inequality2.slack.array()
                                           * data_complementarity.dual.array()
          - barrier; 
  data_inequality2.duality.array()
      = data_inequality2.slack.array() * data_inequality2.dual.array()
          + data_inequality1.slack.array() * data_inequality2.slack.array()
                                           * data_complementarity.dual.array()
          - barrier; 
  pdipmfunc::ComputeDuality(data_complementarity.slack, 
                            data_complementarity.dual, barrier);
}


inline void ComputeComplementaritySlackAndDualDirection(
    const double barrier, const ConstraintComponentData& data_inequality1,
    const ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity) {
  assert(barrier);
  assert(max_complementarity_violation > 0);
  data_complementarity.dslack.array() 
      = - data_inequality1.slack.array() * data_inequality2.dslack.array() 
        - data_inequality2.slack.array() * data_inequality1.dslack.array()
        - data_complementarity.residual.array();
  pdipmfunc::ComputeDualDirection(complementarity_data_.slack, 
                                  complementarity_data_.dual, 
                                  complementarity_data_.dslack, 
                                  complementarity_data_.duality, 
                                  complementarity_data_.ddual);
}


inline void ComputeInequalityDualDirection(
    const double barrier, const ConstraintComponentData& data_inequality1,
    const ConstraintComponentData& data_inequality2, 
    ConstraintComponentData& data_complementarity);


} // namespace mpccfunc
} // namespace idocp


#endif // IDOCP_MPCC_FUNC_HXX_ 