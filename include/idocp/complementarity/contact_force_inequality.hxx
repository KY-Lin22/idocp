#include "idocp/complementarity/contact_force_inequality.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <assert.h>

namespace idocp {

inline ContactForceInequality::ContactForceInequality(
    const Robot& robot, const double mu, const double barrier, 
    const double fraction_to_boundary_rate)
  : num_point_contacts_(robot.num_point_contacts()),
    dimc_(6*robot.num_point_contacts()),
    mu_(mu), 
    barrier_(barrier), 
    fraction_to_boundary_rate_(fraction_to_boundary_rate) {
  f_rsc_.setZero();
}


inline ContactForceInequality::ContactForceInequality() 
  : num_point_contacts_(0),
    dimc_(0),
    mu_(0), 
    barrier_(0), 
    fraction_to_boundary_rate_(0) {
  f_rsc_.setZero();
}


inline ContactForceInequality::~ContactForceInequality() {
}


inline bool ContactForceInequality::isFeasible(const Robot& robot, 
                                               const SplitSolution& s) {
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    if (s.f_verbose.segment<kDimf>(kDimf_verbose*i).minCoeff() < 0) {
      return false;
    }
    const double fx = s.f.coeff(3*i  );
    const double fy = s.f.coeff(3*i+1);
    const double fz = s.f.coeff(3*i+2);
    assert(fx == s.f_verbose(kDimf_verbose*i  )-s.f_verbose(kDimf_verbose*i+1));
    assert(fy == s.f_verbose(kDimf_verbose*i+2)-s.f_verbose(kDimf_verbose*i+3));
    assert(fz == s.f_verbose(kDimf_verbose*i+4));
    const double friction_res = mu_*mu_*fz*fz - fx*fx - fy*fy;
    if (friction_res < 0) {
      return false;
    }
  }
  return true;
}


inline void ContactForceInequality::setSlack(const Robot& robot, 
                                             const double dtau, 
                                             const SplitSolution& s,
                                             ConstraintComponentData& data) {
  assert(dtau > 0);
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.slack.segment<kDimf>(kDimc*i) = dtau * s.f_verbose.segment<kDimf>(kDimf_verbose*i);
    const double fx = s.f.coeff(kDimb*i  );
    const double fy = s.f.coeff(kDimb*i+1);
    const double fz = s.f.coeff(kDimb*i+2);
    assert(fx == s.f_verbose(kDimf_verbose*i  )-s.f_verbose(kDimf_verbose*i+1));
    assert(fy == s.f_verbose(kDimf_verbose*i+2)-s.f_verbose(kDimf_verbose*i+3));
    assert(fz == s.f_verbose(kDimf_verbose*i+4));
    data.slack.coeffRef(kDimc*i+kDimf) = dtau * (mu_*mu_*fz*fz-fx*fx-fy*fy);
  }
  pdipmfunc::SetSlackAndDualPositive(barrier_, data.slack, data.dual);
}


inline void ContactForceInequality::computePrimalResidual(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    ConstraintComponentData& data) {
  assert(dtau > 0);
  data.residual = data.slack;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.residual.segment<kDimf>(kDimc*i).noalias() -= dtau * s.f_verbose.segment<kDimf>(kDimf_verbose*i);
    const double fx = s.f.coeff(kDimb*i  );
    const double fy = s.f.coeff(kDimb*i+1);
    const double fz = s.f.coeff(kDimb*i+2);
    assert(fx == s.f_verbose(kDimf_verbose*i  )-s.f_verbose(kDimf_verbose*i+1));
    assert(fy == s.f_verbose(kDimf_verbose*i+2)-s.f_verbose(kDimf_verbose*i+3));
    assert(fz == s.f_verbose(kDimf_verbose*i+4));
    const double friction_cone_residual = (mu_*mu_*fz*fz-fx*fx-fy*fy);
    data.residual.coeffRef(kDimc*i+kDimf) -= dtau * friction_cone_residual;
  }
}


inline void ContactForceInequality::augmentDualResidual(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const ConstraintComponentData& data, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_residual.lf().segment<kDimf>(kDimf_verbose*i).noalias() -= dtau * data.dual.segment<kDimf>(i*kDimc);
    const double fx = s.f.coeff(kDimb*i  );
    const double fy = s.f.coeff(kDimb*i+1);
    const double fz = s.f.coeff(kDimb*i+2);
    assert(fx == s.f_verbose(kDimf_verbose*i  )-s.f_verbose(kDimf_verbose*i+1));
    assert(fy == s.f_verbose(kDimf_verbose*i+2)-s.f_verbose(kDimf_verbose*i+3));
    assert(fz == s.f_verbose(kDimf_verbose*i+4));
    const double friction_cone_dual = data.dual.coeff(kDimc*i+kDimf);
    kkt_residual.lf().coeffRef(kDimf_verbose*i  ) += 2 * dtau * fx * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf_verbose*i+1) -= 2 * dtau * fx * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf_verbose*i+2) += 2 * dtau * fy * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf_verbose*i+3) -= 2 * dtau * fy * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf_verbose*i+4) -= 2 * dtau * mu_ * mu_ * fz * friction_cone_dual;
  }
}


inline void ContactForceInequality::augmentCondensedHessian(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const Eigen::VectorXd& diagonal, KKTMatrix& kkt_matrix) {
  assert(dtau > 0);
  assert(diagonal.size() == kDimc*robot.num_point_contacts());
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    f_rsc_.coeffRef(0) = - s.f.coeff(kDimb*i  );
    f_rsc_.coeffRef(1) = s.f.coeff(kDimb*i  );
    f_rsc_.coeffRef(2) = - s.f.coeff(kDimb*i+1);
    f_rsc_.coeffRef(3) = s.f.coeff(kDimb*i+1);
    f_rsc_.coeffRef(4) = mu_ * mu_ * s.f.coeff(kDimb*i+2);
    kkt_matrix.Qff().template block<kDimf, kDimf>(kDimf_verbose*i, 
                                                  kDimf_verbose*i).noalias()
        += (4*dtau*dtau*diagonal.coeff(kDimc*i+5)) * f_rsc_ * f_rsc_.transpose();
    kkt_matrix.Qff().template block<kDimf, kDimf>(kDimf_verbose*i, 
                                                  kDimf_verbose*i).diagonal().noalias()
        += (dtau*dtau) * diagonal.segment<kDimf>(kDimc*i);
  }
}


inline void ContactForceInequality::augmentCondensedResidual(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const Eigen::VectorXd& condensed_residual, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  assert(condensed_residual.size() == kDimc*robot.num_point_contacts());
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    f_rsc_.coeffRef(0) = - s.f.coeff(kDimb*i  );
    f_rsc_.coeffRef(1) = s.f.coeff(kDimb*i  );
    f_rsc_.coeffRef(2) = - s.f.coeff(kDimb*i+1);
    f_rsc_.coeffRef(3) = s.f.coeff(kDimb*i+1);
    f_rsc_.coeffRef(4) = mu_ * mu_ * s.f.coeff(kDimb*i+2);
    kkt_residual.lf().segment<kDimf>(kDimf_verbose*i).noalias() 
        += dtau * condensed_residual.segment<kDimf>(kDimc*i);
    kkt_residual.lf().segment<kDimf>(kDimf_verbose*i).noalias()
        += 2 * dtau * condensed_residual.coeff(kDimc*i+kDimf) * f_rsc_;
  }
}


inline void ContactForceInequality::computeSlackDirection(
    const Robot& robot, const double dtau, const SplitSolution& s,
    const SplitDirection& d, ConstraintComponentData& data) const {
  assert(dtau > 0);
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.dslack.segment<kDimf>(kDimc*i) 
        = dtau * d.df().segment<kDimf>(kDimf_verbose*i);
    data.dslack.coeffRef(kDimc*i+kDimf) 
        = 2 * dtau * 
            (s.f.coeff(kDimb*i  ) * (-d.df().coeff(kDimf_verbose*i  ) + d.df().coeff(kDimf_verbose*i+1))
             + s.f.coeff(kDimb*i+1) * (-d.df().coeff(kDimf_verbose*i+2) + d.df().coeff(kDimf_verbose*i+3))
             +  mu_ * mu_ * s.f.coeff(kDimb*i+2) * d.df().coeff(kDimf_verbose*i+4));
  }
  data.dslack.noalias() -= data.residual;
}

} // namespace idocp