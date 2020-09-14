#include "idocp/complementarity/contact_force_inequality.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <assert.h>

namespace idocp {

inline ContactForceInequality::ContactForceInequality(const Robot& robot, 
                                                      const double mu)
  : num_point_contacts_(robot.num_point_contacts()),
    dimc_(kDimc*robot.num_point_contacts()),
    mu_(mu) {
  f_rsc_.setZero();
}


inline ContactForceInequality::ContactForceInequality() 
  : num_point_contacts_(0),
    dimc_(0),
    mu_(0) {
  f_rsc_.setZero();
}


inline ContactForceInequality::~ContactForceInequality() {
}


inline bool ContactForceInequality::isFeasible(const Robot& robot, 
                                               const SplitSolution& s) {
  if (s.f.minCoeff() < 0) {
    return false;
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    assert(s.f_3D.coeff(3*i  ) == s.f.coeff(5*i  )-s.f.coeff(5*i+1));
    assert(s.f_3D.coeff(3*i+1) == s.f.coeff(5*i+2)-s.f.coeff(5*i+3));
    assert(s.f_3D.coeff(3*i+2) == s.f.coeff(5*i+4));
    const double fx = s.f_3D.coeff(3*i  );
    const double fy = s.f_3D.coeff(3*i+1);
    const double fz = s.f_3D.coeff(3*i+2);
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
    assert(s.f_3D.coeff(3*i  ) == s.f.coeff(kDimf*i  )-s.f.coeff(kDimf*i+1));
    assert(s.f_3D.coeff(3*i+1) == s.f.coeff(kDimf*i+2)-s.f.coeff(kDimf*i+3));
    assert(s.f_3D.coeff(3*i+2) == s.f.coeff(kDimf*i+4));
    data.slack.template segment<kDimf>(kDimc*i) 
        = dtau * s.f.template segment<kDimf>(kDimf*i);
    const double fx = s.f_3D.coeff(3*i  );
    const double fy = s.f_3D.coeff(3*i+1);
    const double fz = s.f_3D.coeff(3*i+2);
    data.slack.coeffRef(kDimc*i+kDimf) = dtau * (mu_*mu_*fz*fz-fx*fx-fy*fy);
  }
}


inline void ContactForceInequality::computePrimalResidual(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    ConstraintComponentData& data) {
  assert(dtau > 0);
  data.residual = data.slack;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    assert(s.f_3D.coeff(3*i  ) == s.f.coeff(kDimf*i  )-s.f.coeff(kDimf*i+1));
    assert(s.f_3D.coeff(3*i+1) == s.f.coeff(kDimf*i+2)-s.f.coeff(kDimf*i+3));
    assert(s.f_3D.coeff(3*i+2) == s.f.coeff(kDimf*i+4));
    data.residual.template segment<kDimf>(kDimc*i).noalias() 
        -= dtau * s.f.template segment<kDimf>(kDimf*i);
    const double fx = s.f_3D.coeff(3*i  );
    const double fy = s.f_3D.coeff(3*i+1);
    const double fz = s.f_3D.coeff(3*i+2);
    const double friction_cone_residual = (mu_*mu_*fz*fz-fx*fx-fy*fy);
    data.residual.coeffRef(kDimc*i+kDimf) -= dtau * friction_cone_residual;
  }
}


inline void ContactForceInequality::augmentDualResidual(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const ConstraintComponentData& data, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    assert(s.f_3D.coeff(3*i  ) == s.f.coeff(kDimf*i  )-s.f.coeff(kDimf*i+1));
    assert(s.f_3D.coeff(3*i+1) == s.f.coeff(kDimf*i+2)-s.f.coeff(kDimf*i+3));
    assert(s.f_3D.coeff(3*i+2) == s.f.coeff(kDimf*i+4));
    kkt_residual.lf().template segment<kDimf>(kDimf*i).noalias() 
        -= dtau * data.dual.template segment<kDimf>(kDimc*i);
    const double fx = s.f_3D.coeff(3*i  );
    const double fy = s.f_3D.coeff(3*i+1);
    const double fz = s.f_3D.coeff(3*i+2);
    const double friction_cone_dual = data.dual.coeff(kDimc*i+kDimf);
    kkt_residual.lf().coeffRef(kDimf*i  ) += 2 * dtau * fx * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf*i+1) -= 2 * dtau * fx * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf*i+2) += 2 * dtau * fy * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf*i+3) -= 2 * dtau * fy * friction_cone_dual;
    kkt_residual.lf().coeffRef(kDimf*i+4) 
        -= 2 * dtau * mu_ * mu_ * fz * friction_cone_dual;
  }
}


template <typename VectorType>
inline void ContactForceInequality::augmentCondensedHessian(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const Eigen::MatrixBase<VectorType>& diagonal, KKTMatrix& kkt_matrix) {
  assert(dtau > 0);
  assert(diagonal.size() == kDimc*robot.num_point_contacts());
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    assert(s.f_3D.coeff(3*i  ) == s.f.coeff(kDimf*i  )-s.f.coeff(kDimf*i+1));
    assert(s.f_3D.coeff(3*i+1) == s.f.coeff(kDimf*i+2)-s.f.coeff(kDimf*i+3));
    assert(s.f_3D.coeff(3*i+2) == s.f.coeff(kDimf*i+4));
    f_rsc_.coeffRef(0) = - s.f_3D.coeff(3*i  );
    f_rsc_.coeffRef(1) =   s.f_3D.coeff(3*i  );
    f_rsc_.coeffRef(2) = - s.f_3D.coeff(3*i+1);
    f_rsc_.coeffRef(3) =   s.f_3D.coeff(3*i+1);
    f_rsc_.coeffRef(4) = mu_ * mu_ * s.f_3D.coeff(3*i+2);
    kkt_matrix.Qff().template block<kDimf, kDimf>(kDimf*i, kDimf*i).noalias()
        += (4*dtau*dtau*diagonal.coeff(kDimc*i+5)) * f_rsc_ * f_rsc_.transpose();
    kkt_matrix.Qff().template block<kDimf, kDimf>(kDimf*i, kDimf*i).diagonal().noalias()
        += (dtau*dtau) * diagonal.template segment<kDimf>(kDimc*i);
  }
}


template <typename VectorType>
inline void ContactForceInequality::augmentCondensedResidual(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const Eigen::MatrixBase<VectorType>& residual, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  assert(residual.size() == kDimc*robot.num_point_contacts());
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    assert(s.f_3D.coeff(3*i  ) == s.f.coeff(kDimf*i  )-s.f.coeff(kDimf*i+1));
    assert(s.f_3D.coeff(3*i+1) == s.f.coeff(kDimf*i+2)-s.f.coeff(kDimf*i+3));
    assert(s.f_3D.coeff(3*i+2) == s.f.coeff(kDimf*i+4));
    f_rsc_.coeffRef(0) = - s.f_3D.coeff(3*i  );
    f_rsc_.coeffRef(1) =   s.f_3D.coeff(3*i  );
    f_rsc_.coeffRef(2) = - s.f_3D.coeff(3*i+1);
    f_rsc_.coeffRef(3) =   s.f_3D.coeff(3*i+1);
    f_rsc_.coeffRef(4) = mu_ * mu_ * s.f_3D.coeff(3*i+2);
    kkt_residual.lf().template segment<kDimf>(kDimf*i).noalias() 
        += dtau * residual.template segment<kDimf>(kDimc*i);
    kkt_residual.lf().template segment<kDimf>(kDimf*i).noalias() 
        += 2 * dtau * residual.coeff(kDimc*i+kDimf) * f_rsc_;
  }
}


inline void ContactForceInequality::computeSlackDirection(
    const Robot& robot, const double dtau, const SplitSolution& s,
    const SplitDirection& d, ConstraintComponentData& data) const {
  assert(dtau > 0);
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    assert(s.f_3D.coeff(3*i  ) == s.f.coeff(kDimf*i  )-s.f.coeff(kDimf*i+1));
    assert(s.f_3D.coeff(3*i+1) == s.f.coeff(kDimf*i+2)-s.f.coeff(kDimf*i+3));
    assert(s.f_3D.coeff(3*i+2) == s.f.coeff(kDimf*i+4));
    data.dslack.template segment<kDimf>(kDimc*i) 
        = dtau * d.df().template segment<kDimf>(kDimf*i);
    data.dslack.coeffRef(kDimc*i+kDimf) 
        = 2 * dtau * 
            (s.f_3D.coeff(3*i  ) * (-d.df().coeff(kDimf*i  ) + d.df().coeff(kDimf*i+1))
             + s.f_3D.coeff(3*i+1) * (-d.df().coeff(kDimf*i+2) + d.df().coeff(kDimf*i+3))
             +  mu_ * mu_ * s.f_3D.coeff(3*i+2) * d.df().coeff(kDimf*i+4));
  }
  data.dslack.noalias() -= data.residual;
}


inline void ContactForceInequality::set_mu(const double mu) {
  mu_ = mu;
}


inline double ContactForceInequality::mu() const {
  return mu_;
}

} // namespace idocp