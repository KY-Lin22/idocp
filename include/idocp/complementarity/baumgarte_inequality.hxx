#include "idocp/complementarity/baumgarte_inequality.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <assert.h>

namespace idocp {

inline BaumgarteInequality::BaumgarteInequality(const Robot& robot)
  : num_point_contacts_(robot.num_point_contacts()),
    dimc_(kDimc*robot.num_point_contacts()),
    Baumgarte_residual_(Eigen::VectorXd::Zero(3*robot.num_point_contacts())),
    dBaumgarte_dq_(Eigen::MatrixXd::Zero(3*robot.num_point_contacts(), 
                                         robot.dimv())),
    dBaumgarte_dv_(Eigen::MatrixXd::Zero(3*robot.num_point_contacts(), 
                                         robot.dimv())),
    dBaumgarte_da_(Eigen::MatrixXd::Zero(3*robot.num_point_contacts(), 
                                         robot.dimv())),
    dBaumgarte_verbose_da_(Eigen::MatrixXd::Zero(
        kDimc*robot.num_point_contacts(), robot.dimv())),
    dBaumgarte_verbose_dq_(Eigen::MatrixXd::Zero(
        kDimc*robot.num_point_contacts(), robot.dimv())),
    dBaumgarte_verbose_dv_(Eigen::MatrixXd::Zero(
        kDimc*robot.num_point_contacts(), robot.dimv())) {
  Qfr_rsc_.setZero();
  f_rsc_.setZero();
}


inline BaumgarteInequality::BaumgarteInequality() 
  : num_point_contacts_(0),
    dimc_(0),
    dBaumgarte_dq_(),
    dBaumgarte_dv_(),
    dBaumgarte_da_(),
    dBaumgarte_verbose_dq_(),
    dBaumgarte_verbose_dv_(),
    dBaumgarte_verbose_da_() {
  Qfr_rsc_.setZero();
  f_rsc_.setZero();
}


inline BaumgarteInequality::~BaumgarteInequality() {
}


inline bool BaumgarteInequality::isFeasible(Robot& robot, 
                                            const SplitSolution& s) {
  robot.computeBaumgarteResidual(Baumgarte_residual_); 
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    if (s.r.coeff(2*i  )+Baumgarte_residual_.coeff(3*i) < 0) {
      return false;
    }
    if (s.r.coeff(2*i  )-Baumgarte_residual_.coeff(3*i) < 0) {
      return false;
    }
    if (s.r.coeff(2*i+1)+Baumgarte_residual_.coeff(3*i+1) < 0) {
      return false;
    }
    if (s.r.coeff(2*i+1)-Baumgarte_residual_.coeff(3*i+1) < 0) {
      return false;
    }
    if (Baumgarte_residual_.coeff(3*i+2) < 0) {
      return false;
    }
    if (s.r.coeff(2*i)*s.r.coeff(2*i) + s.r.coeff(2*i+1)*s.r.coeff(2*i+1) < 0) {
      return false;
    }
  }
  return true;
}


inline void BaumgarteInequality::setSlack(Robot& robot, const double dtau, 
                                          const SplitSolution& s,
                                          ConstraintComponentData& data) {
  assert(dtau > 0);
  robot.computeBaumgarteResidual(dtau, Baumgarte_residual_); 
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.slack.coeffRef(kDimc*i  ) 
        = dtau * (s.r.coeff(2*i  )+Baumgarte_residual_.coeff(3*i  ));
    data.slack.coeffRef(kDimc*i+1) 
        = dtau * (s.r.coeff(2*i  )-Baumgarte_residual_.coeff(3*i  ));
    data.slack.coeffRef(kDimc*i+2) 
        = dtau * (s.r.coeff(2*i+1)+Baumgarte_residual_.coeff(3*i+1));
    data.slack.coeffRef(kDimc*i+3) 
        = dtau * (s.r.coeff(2*i+1)-Baumgarte_residual_.coeff(3*i+1));
    data.slack.coeffRef(kDimc*i+4) = dtau * Baumgarte_residual_.coeff(3*i+2);
    data.slack.coeffRef(kDimc*i+5) 
        = dtau * (s.r.coeff(2*i)*s.r.coeff(2*i) 
                    + s.r.coeff(2*i+1)*s.r.coeff(2*i+1));
  }
}


inline void BaumgarteInequality::computePrimalResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    ConstraintComponentData& data) {
  assert(dtau > 0);
  robot.computeBaumgarteResidual(Baumgarte_residual_); 
  data.residual = data.slack;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.residual.coeffRef(kDimc*i  ) 
        -= dtau * (s.r.coeff(2*i  )+Baumgarte_residual_.coeff(3*i  ));
    data.residual.coeffRef(kDimc*i+1) 
        -= dtau * (s.r.coeff(2*i  )-Baumgarte_residual_.coeff(3*i  ));
    data.residual.coeffRef(kDimc*i+2) 
        -= dtau * (s.r.coeff(2*i+1)+Baumgarte_residual_.coeff(3*i+1));
    data.residual.coeffRef(kDimc*i+3) 
        -= dtau * (s.r.coeff(2*i+1)-Baumgarte_residual_.coeff(3*i+1));
    data.residual.coeffRef(kDimc*i+4) 
        -= dtau * Baumgarte_residual_.coeff(3*i+2); 
    data.residual.coeffRef(kDimc*i+5) 
        -= dtau * (s.r.coeff(2*i)*s.r.coeff(2*i)
                    + s.r.coeff(2*i+1)*s.r.coeff(2*i+1));
  }
}


inline void BaumgarteInequality::augmentDualResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const ConstraintComponentData& data, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  robot.computeBaumgarteDerivatives(dBaumgarte_dq_, dBaumgarte_dv_, 
                                    dBaumgarte_da_); 
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    dBaumgarte_verbose_dq_.row(kDimc*i  ) =   dBaumgarte_dq_.row(3*i  );
    dBaumgarte_verbose_dq_.row(kDimc*i+1) = - dBaumgarte_dq_.row(3*i  );
    dBaumgarte_verbose_dq_.row(kDimc*i+2) =   dBaumgarte_dq_.row(3*i+1);
    dBaumgarte_verbose_dq_.row(kDimc*i+3) = - dBaumgarte_dq_.row(3*i+1);
    dBaumgarte_verbose_dq_.row(kDimc*i+4) =   dBaumgarte_dq_.row(3*i+2);
    dBaumgarte_verbose_dv_.row(kDimc*i  ) =   dBaumgarte_dv_.row(3*i  );
    dBaumgarte_verbose_dv_.row(kDimc*i+1) = - dBaumgarte_dv_.row(3*i  );
    dBaumgarte_verbose_dv_.row(kDimc*i+2) =   dBaumgarte_dv_.row(3*i+1);
    dBaumgarte_verbose_dv_.row(kDimc*i+3) = - dBaumgarte_dv_.row(3*i+1);
    dBaumgarte_verbose_dv_.row(kDimc*i+4) =   dBaumgarte_dv_.row(3*i+2);
    dBaumgarte_verbose_da_.row(kDimc*i  ) =   dBaumgarte_da_.row(3*i  );
    dBaumgarte_verbose_da_.row(kDimc*i+1) = - dBaumgarte_da_.row(3*i  );
    dBaumgarte_verbose_da_.row(kDimc*i+2) =   dBaumgarte_da_.row(3*i+1);
    dBaumgarte_verbose_da_.row(kDimc*i+3) = - dBaumgarte_da_.row(3*i+1);
    dBaumgarte_verbose_da_.row(kDimc*i+4) =   dBaumgarte_da_.row(3*i+2);
    kkt_residual.lr().coeffRef(2*i  ) 
        -= dtau * (data.dual.coeff(kDimc*i  ) + data.dual.coeff(kDimc*i+1) 
                    + 2*s.r.coeff(2*i  )*data.dual.coeff(kDimc*i+5));
    kkt_residual.lr().coeffRef(2*i+1) 
        -= dtau * (data.dual.coeff(kDimc*i+2) + data.dual.coeff(kDimc*i+3) 
                    + 2*s.r.coeff(2*i+1)*data.dual.coeff(kDimc*i+5));
  }
  kkt_residual.la().noalias() 
      -= dtau * dBaumgarte_verbose_da_.transpose() * data.dual;
  kkt_residual.lq().noalias() 
      -= dtau * dBaumgarte_verbose_dq_.transpose() * data.dual;
  kkt_residual.lv().noalias() 
      -= dtau * dBaumgarte_verbose_dv_.transpose() * data.dual;
}


template <typename VectorType>
inline void BaumgarteInequality::augmentCondensedHessian(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const Eigen::MatrixBase<VectorType>& diagonal, KKTMatrix& kkt_matrix) {
  assert(dtau > 0);
  assert(diagonal.size() == kDimc*robot.num_point_contacts());
  kkt_matrix.Qaa().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_da_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_da_;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qar().col(2*i  ).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i  )-diagonal.coeff(kDimc*i+1)) 
                       * dBaumgarte_da_.row(3*i  ).transpose();
    kkt_matrix.Qar().col(2*i+1).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i+2)-diagonal.coeff(kDimc*i+3)) 
                       * dBaumgarte_da_.row(3*i+1).transpose();
  }
  kkt_matrix.Qaq().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_da_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_dq_;
  kkt_matrix.Qav().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_da_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_dv_;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qrr().coeffRef(2*i  , 2*i  ) 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i  ) + diagonal.coeff(kDimc*i+1) 
                           + 4 * diagonal.coeff(kDimc*i+5) 
                               * s.r.coeff(2*i  ) * s.r.coeff(2*i  ));
    kkt_matrix.Qrr().coeffRef(2*i  , 2*i+1) 
        += (dtau*dtau) * (4 * diagonal.coeff(kDimc*i+5) 
                            * s.r.coeff(2*i  ) * s.r.coeff(2*i+1));
    kkt_matrix.Qrr().coeffRef(2*i+1, 2*i  ) 
        += (dtau*dtau) * (4 * diagonal.coeff(kDimc*i+5) 
                            * s.r.coeff(2*i+1) * s.r.coeff(2*i  ));
    kkt_matrix.Qrr().coeffRef(2*i+1, 2*i+1) 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i+2) + diagonal.coeff(kDimc*i+3) 
                           + 4 * diagonal.coeff(kDimc*i+5) 
                               * s.r.coeff(2*i+1) * s.r.coeff(2*i+1));
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qrq().row(2*i  ).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i  )-diagonal.coeff(kDimc*i+1)) 
                       * dBaumgarte_dq_.row(3*i  ).transpose();
    kkt_matrix.Qrq().row(2*i+1).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i+2)-diagonal.coeff(kDimc*i+3)) 
                       * dBaumgarte_dq_.row(3*i+1).transpose();
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qrv().row(2*i  ).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i  )-diagonal.coeff(kDimc*i+1)) 
                       * dBaumgarte_dv_.row(3*i  ).transpose();
    kkt_matrix.Qrv().row(2*i+1).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i+2)-diagonal.coeff(kDimc*i+3)) 
                       * dBaumgarte_dv_.row(3*i+1).transpose();
  }
  kkt_matrix.Qqq().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_dq_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_dq_;
  kkt_matrix.Qqv().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_dq_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_dv_;
  kkt_matrix.Qvv().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_dv_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_dv_;
}


template <typename VectorType>
inline void BaumgarteInequality::augmentComplementarityCondensedHessian(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const ContactForceInequality& contact_force_inequality, 
    const Eigen::MatrixBase<VectorType>& diagonal, KKTMatrix& kkt_matrix) {
  assert(diagonal.size() == kDimc*robot.num_point_contacts());
  const double mu = contact_force_inequality.mu();
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qaf().block(0, kDimf*i, robot.dimv(), kDimf).noalias() 
        += (dtau*dtau) * dBaumgarte_verbose_da_.block(kDimc*i, 0, kDimf, 
                                                      robot.dimv()).transpose()
                       * diagonal.template segment<kDimf>(kDimc*i).asDiagonal();
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    f_rsc_.coeffRef(0) = - s.f_3D.coeff(3*i  );
    f_rsc_.coeffRef(1) =   s.f_3D.coeff(3*i  );
    f_rsc_.coeffRef(2) = - s.f_3D.coeff(3*i+1);
    f_rsc_.coeffRef(3) =   s.f_3D.coeff(3*i+1);
    f_rsc_.coeffRef(4) =   mu * mu * s.f_3D.coeff(3*i+2);
    Qfr_rsc_.col(0) = (dtau*dtau) * 4 * s.r.coeff(2*i  ) 
                                  * diagonal.coeff(kDimc*i+5) * f_rsc_;
    Qfr_rsc_.col(1) = (dtau*dtau) * 4 * s.r.coeff(2*i+1) 
                                  * diagonal.coeff(kDimc*i+5) * f_rsc_;
    Qfr_rsc_.coeffRef(0, 0) += (dtau*dtau) * diagonal.coeff(kDimc*i  );
    Qfr_rsc_.coeffRef(1, 0) += (dtau*dtau) * diagonal.coeff(kDimc*i+1);
    Qfr_rsc_.coeffRef(2, 1) += (dtau*dtau) * diagonal.coeff(kDimc*i+2);
    Qfr_rsc_.coeffRef(3, 1) += (dtau*dtau) * diagonal.coeff(kDimc*i+3);
    kkt_matrix.Qfr().template block<kDimf, 2>(kDimf*i, 2*i).noalias() 
        += Qfr_rsc_;
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qfq().block(kDimf*i, 0, kDimf, robot.dimv()).noalias() 
        += (dtau*dtau) * diagonal.template segment<kDimf>(kDimc*i).asDiagonal() 
                       * dBaumgarte_verbose_dq_.block(kDimc*i, 0,  
                                                      kDimf, robot.dimv());
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qfv().block(kDimf*i, 0, kDimf, robot.dimv()).noalias() 
        += (dtau*dtau) * diagonal.template segment<kDimf>(kDimc*i).asDiagonal() 
                       * dBaumgarte_verbose_dv_.block(kDimc*i, 0, 
                                                      kDimf, robot.dimv());
  }
}


template <typename VectorType>
inline void BaumgarteInequality::augmentCondensedResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const Eigen::MatrixBase<VectorType>& residual, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  assert(residual.size() == kDimc*robot.num_point_contacts());
  kkt_residual.la().noalias() 
      -= dtau * dBaumgarte_verbose_da_.transpose() * residual;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_residual.lr().coeffRef(2*i  )
        -= dtau * (residual.coeff(kDimc*i  ) + residual.coeff(kDimc*i+1)
                    + 2 * s.r.coeff(2*i  ) * residual.coeff(kDimc*i+5));
    kkt_residual.lr().coeffRef(2*i+1)
        -= dtau * (residual.coeff(kDimc*i+2) + residual.coeff(kDimc*i+3)
                    + 2 * s.r.coeff(2*i+1) * residual.coeff(kDimc*i+5));
  }
  kkt_residual.lq().noalias() 
      -= dtau * dBaumgarte_verbose_dq_.transpose() * residual;
  kkt_residual.lv().noalias() 
      -= dtau * dBaumgarte_verbose_dv_.transpose() * residual;
}


inline void BaumgarteInequality::computeSlackDirection(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const SplitDirection& d, ConstraintComponentData& data) const {
  assert(dtau > 0);
  data.dslack = - data.residual;
  data.dslack.noalias() += dtau * dBaumgarte_verbose_da_ * d.da();
  data.dslack.noalias() += dtau * dBaumgarte_verbose_dq_ * d.dq();
  data.dslack.noalias() += dtau * dBaumgarte_verbose_dv_ * d.dv();
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.dslack.coeffRef(kDimc*i  ) += dtau * d.dr().coeff(2*i  );
    data.dslack.coeffRef(kDimc*i+1) += dtau * d.dr().coeff(2*i  );
    data.dslack.coeffRef(kDimc*i+2) += dtau * d.dr().coeff(2*i+1);
    data.dslack.coeffRef(kDimc*i+3) += dtau * d.dr().coeff(2*i+1);
    data.dslack.coeffRef(kDimc*i+5) += 2 * dtau * s.r.coeff(2*i  ) 
                                                * d.dr().coeff(2*i  );
    data.dslack.coeffRef(kDimc*i+5) += 2 * dtau * s.r.coeff(2*i+1) 
                                                * d.dr().coeff(2*i+1);
  }
}

} // namespace idocp