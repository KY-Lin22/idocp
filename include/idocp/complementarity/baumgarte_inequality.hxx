#include "idocp/complementarity/baumgarte_inequality.hpp"
#include "idocp/constraints/pdipm_func.hpp"

#include <assert.h>

namespace idocp {

inline BaumgarteInequality::BaumgarteInequality(
    const Robot& robot, const double barrier, 
    const double fraction_to_boundary_rate)
  : num_point_contacts_(robot.num_point_contacts()),
    dimc_(6*robot.num_point_contacts()),
    barrier_(barrier), 
    fraction_to_boundary_rate_(fraction_to_boundary_rate),
    Baumgarte_residual_(Eigen::VectorXd::Zero(3*robot.num_point_contacts())),
    dBaumgarte_dq_(Eigen::MatrixXd::Zero(3*robot.num_point_contacts(), 
                                         robot.dimv())),
    dBaumgarte_dv_(Eigen::MatrixXd::Zero(3*robot.num_point_contacts(), 
                                         robot.dimv())),
    dBaumgarte_da_(Eigen::MatrixXd::Zero(3*robot.num_point_contacts(), 
                                         robot.dimv())),
    dBaumgarte_verbose_da_(Eigen::MatrixXd::Zero(6*robot.num_point_contacts(), 
                                                 robot.dimv())),
    dBaumgarte_verbose_dq_(Eigen::MatrixXd::Zero(6*robot.num_point_contacts(), 
                                                 robot.dimv())),
    dBaumgarte_verbose_dv_(Eigen::MatrixXd::Zero(6*robot.num_point_contacts(), 
                                                 robot.dimv())) {
  Qff_rsc_.setZero();
  f_rsc_.setZero();
}


inline BaumgarteInequality::BaumgarteInequality() 
  : num_point_contacts_(0),
    dimc_(0),
    barrier_(0), 
    fraction_to_boundary_rate_(0),
    dBaumgarte_dq_(),
    dBaumgarte_dv_(),
    dBaumgarte_da_(),
    dBaumgarte_verbose_dq_(),
    dBaumgarte_verbose_dv_(),
    dBaumgarte_verbose_da_() {
  Qff_rsc_.setZero();
  f_rsc_.setZero();
}


inline BaumgarteInequality::~BaumgarteInequality() {
}


inline bool BaumgarteInequality::isFeasible(Robot& robot, 
                                            const SplitSolution& s) {
  robot.computeBaumgarteResidual(Baumgarte_residual_); 
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    if (s.f_verbose.coeff(7*i+5)+Baumgarte_residual_.coeff(3*i) < 0) {
      return false;
    }
    if (s.f_verbose.coeff(7*i+5)-Baumgarte_residual_.coeff(3*i) < 0) {
      return false;
    }
    if (s.f_verbose.coeff(7*i+6)+Baumgarte_residual_.coeff(3*i+1) < 0) {
      return false;
    }
    if (s.f_verbose.coeff(7*i+6)-Baumgarte_residual_.coeff(3*i+1) < 0) {
      return false;
    }
    if (Baumgarte_residual_.coeff(3*i+2) < 0) {
      return false;
    }
    if (s.f_verbose.coeff(7*i+5)*s.f_verbose.coeff(7*i+5) 
         + s.f_verbose.coeff(7*i+6)*s.f_verbose.coeff(7*i+6) < 0) {
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
        = dtau * (s.f_verbose.coeff(7*i+5)+Baumgarte_residual_.coeff(3*i  ));
    data.slack.coeffRef(kDimc*i+1) 
        = dtau * (s.f_verbose.coeff(7*i+5)-Baumgarte_residual_.coeff(3*i  ));
    data.slack.coeffRef(kDimc*i+2) 
        = dtau * (s.f_verbose.coeff(7*i+6)+Baumgarte_residual_.coeff(3*i+1));
    data.slack.coeffRef(kDimc*i+3) 
        = dtau * (s.f_verbose.coeff(7*i+6)-Baumgarte_residual_.coeff(3*i+1));
    data.slack.coeffRef(kDimc*i+4) 
        = dtau * Baumgarte_residual_.coeff(3*i+2);
    data.slack.coeffRef(kDimc*i+5) 
        = dtau * (s.f_verbose.coeff(7*i+5)*s.f_verbose.coeff(7*i+5)
                   + s.f_verbose.coeff(7*i+6)*s.f_verbose.coeff(7*i+6));
  }
  pdipmfunc::SetSlackAndDualPositive(barrier_, data.slack, data.dual);
}


inline void BaumgarteInequality::computePrimalResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    ConstraintComponentData& data) {
  assert(dtau > 0);
  robot.computeBaumgarteResidual(Baumgarte_residual_); 
  data.residual = data.slack;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.residual.coeffRef(kDimc*i  ) 
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+5)
                    + Baumgarte_residual_.coeff(kDimb*i  ));
    data.residual.coeffRef(kDimc*i+1) 
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+5)
                    - Baumgarte_residual_.coeff(kDimb*i  ));
    data.residual.coeffRef(kDimc*i+2) 
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+6)
                    + Baumgarte_residual_.coeff(kDimb*i+1));
    data.residual.coeffRef(kDimc*i+3) 
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+6)
                    - Baumgarte_residual_.coeff(kDimb*i+1));
    data.residual.coeffRef(kDimc*i+4) 
        -= dtau * Baumgarte_residual_.coeff(3*i+2);
    data.residual.coeffRef(kDimc*i+5) 
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+5)*s.f_verbose.coeff(kDimf_verbose*i+5)
                    + s.f_verbose.coeff(kDimf_verbose*i+6)*s.f_verbose.coeff(kDimf_verbose*i+6));
  }
}


inline void BaumgarteInequality::augmentDualResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const ConstraintComponentData& data, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  robot.computeBaumgarteDerivatives(dBaumgarte_dq_, dBaumgarte_dv_, dBaumgarte_da_); 
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    dBaumgarte_verbose_dq_.row(kDimc*i  ) = dBaumgarte_dq_.row(kDimb*i  );
    dBaumgarte_verbose_dq_.row(kDimc*i+1) = - dBaumgarte_dq_.row(kDimb*i  );
    dBaumgarte_verbose_dq_.row(kDimc*i+2) = dBaumgarte_dq_.row(kDimb*i+1);
    dBaumgarte_verbose_dq_.row(kDimc*i+3) = - dBaumgarte_dq_.row(kDimb*i+1);
    dBaumgarte_verbose_dq_.row(kDimc*i+4) = dBaumgarte_dq_.row(kDimb*i+2);
    dBaumgarte_verbose_dv_.row(kDimc*i  ) = dBaumgarte_dv_.row(kDimb*i  );
    dBaumgarte_verbose_dv_.row(kDimc*i+1) = - dBaumgarte_dv_.row(kDimb*i  );
    dBaumgarte_verbose_dv_.row(kDimc*i+2) = dBaumgarte_dv_.row(kDimb*i+1);
    dBaumgarte_verbose_dv_.row(kDimc*i+3) = - dBaumgarte_dv_.row(kDimb*i+1);
    dBaumgarte_verbose_dv_.row(kDimc*i+4) = dBaumgarte_dv_.row(kDimb*i+2);
    dBaumgarte_verbose_da_.row(kDimc*i  ) = dBaumgarte_da_.row(kDimb*i  );
    dBaumgarte_verbose_da_.row(kDimc*i+1) = - dBaumgarte_da_.row(kDimb*i  );
    dBaumgarte_verbose_da_.row(kDimc*i+2) = dBaumgarte_da_.row(kDimb*i+1);
    dBaumgarte_verbose_da_.row(kDimc*i+3) = - dBaumgarte_da_.row(kDimb*i+1);
    dBaumgarte_verbose_da_.row(kDimc*i+4) = dBaumgarte_da_.row(kDimb*i+2);
    kkt_residual.lf().coeffRef(i*kDimf_verbose+5) 
        -= dtau * (data.dual(i*kDimc  ) + data.dual(i*kDimc+1) 
                    + 2*data.dual(i*kDimc+5)*s.f_verbose(i*kDimf_verbose+5));
    kkt_residual.lf().coeffRef(i*kDimf_verbose+6) 
        -= dtau * (data.dual(i*kDimc+2) + data.dual(i*kDimc+3) 
                    + 2*data.dual(i*kDimc+5)*s.f_verbose(i*kDimf_verbose+6));
  }
  kkt_residual.la().noalias() -= dtau * dBaumgarte_verbose_da_.transpose() * data.dual;
  kkt_residual.lq().noalias() -= dtau * dBaumgarte_verbose_dq_.transpose() * data.dual;
  kkt_residual.lv().noalias() -= dtau * dBaumgarte_verbose_dv_.transpose() * data.dual;
}


inline void BaumgarteInequality::augmentCondensedHessian(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const Eigen::VectorXd& diagonal, KKTMatrix& kkt_matrix) {
  assert(dtau > 0);
  assert(diagonal.size() == kDimc*robot.num_point_contacts());
  kkt_matrix.Qaa().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_da_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_da_;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qaf().col(kDimf_verbose*i+5).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i  )-diagonal.coeff(kDimc*i+1)) 
                       * dBaumgarte_da_.row(kDimb*i  ).transpose();
    kkt_matrix.Qaf().col(kDimf_verbose*i+6).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i+2)-diagonal.coeff(kDimc*i+3)) 
                       * dBaumgarte_da_.row(kDimb*i+1).transpose();
  }
  kkt_matrix.Qaq().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_da_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_dq_;
  kkt_matrix.Qav().noalias() 
      += (dtau*dtau) * dBaumgarte_verbose_da_.transpose() 
                     * diagonal.asDiagonal() * dBaumgarte_verbose_dv_;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qff().coeffRef(kDimf_verbose*i+5, kDimf_verbose*i+5) 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i  ) + diagonal.coeff(kDimc*i+1) 
                           + 4 * diagonal.coeff(kDimc*i+5) 
                               * s.f_verbose.coeff(kDimf_verbose*i+5) 
                               * s.f_verbose.coeff(kDimf_verbose*i+5));
    kkt_matrix.Qff().coeffRef(kDimf_verbose*i+5, kDimf_verbose*i+6) 
        += (dtau*dtau) * (4 * diagonal.coeff(kDimc*i+5) 
                            * s.f_verbose.coeff(kDimf_verbose*i+5)
                            * s.f_verbose.coeff(kDimf_verbose*i+6));
    kkt_matrix.Qff().coeffRef(kDimf_verbose*i+6, kDimf_verbose*i+5) 
        += (dtau*dtau) * (4 * diagonal.coeff(kDimc*i+5) 
                            * s.f_verbose.coeff(kDimf_verbose*i+6)
                            * s.f_verbose.coeff(kDimf_verbose*i+5));
    kkt_matrix.Qff().coeffRef(kDimf_verbose*i+6, kDimf_verbose*i+6) 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i+2) + diagonal.coeff(kDimc*i+3) 
                           + 4 * diagonal.coeff(kDimc*i+5) 
                               * s.f_verbose.coeff(kDimf_verbose*i+6) 
                               * s.f_verbose.coeff(kDimf_verbose*i+6));
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qfq().row(kDimf_verbose*i+5).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i  )-diagonal.coeff(kDimc*i+1)) 
                       * dBaumgarte_dq_.row(kDimb*i  ).transpose();
    kkt_matrix.Qfq().row(kDimf_verbose*i+6).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i+2)-diagonal.coeff(kDimc*i+3)) 
                       * dBaumgarte_dq_.row(kDimb*i+1).transpose();
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qfv().row(kDimf_verbose*i+5).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i  )-diagonal.coeff(kDimc*i+1)) 
                       * dBaumgarte_dv_.row(kDimb*i  ).transpose();
    kkt_matrix.Qfv().row(kDimf_verbose*i+6).noalias() 
        += (dtau*dtau) * (diagonal.coeff(kDimc*i+2)-diagonal.coeff(kDimc*i+3)) 
                       * dBaumgarte_dv_.row(kDimb*i+1).transpose();
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


inline void BaumgarteInequality::augmentComplementarityCondensedHessian(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const ContactForceInequality& contact_force_inequality, 
    const Eigen::VectorXd& diagonal, KKTMatrix& kkt_matrix) {
  assert(diagonal.size() == kDimc*robot.num_point_contacts());
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qaf().block(0, kDimf_verbose*i, robot.dimv(), kDimf).noalias() 
        += (dtau*dtau) * dBaumgarte_verbose_da_.block(kDimc*i, 0, kDimf, 
                                                      robot.dimv()).transpose()
                       * diagonal.template segment<kDimf>(kDimc*i).asDiagonal();
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    f_rsc_.coeffRef(0) = - s.f.coeff(kDimb*i);
    f_rsc_.coeffRef(1) = s.f.coeff(kDimb*i);
    f_rsc_.coeffRef(2) = - s.f.coeff(kDimb*i+1);
    f_rsc_.coeffRef(3) = s.f.coeff(kDimb*i+1);
    f_rsc_.coeffRef(4) = contact_force_inequality.mu() 
                          * contact_force_inequality.mu() * s.f.coeff(kDimb*i+2);
    Qff_rsc_.row(0) = (dtau*dtau) * 4 * s.f_verbose.coeff(kDimf_verbose*i+5) 
                                      * diagonal.coeff(kDimc*i+5) * f_rsc_;
    Qff_rsc_.row(1) = (dtau*dtau) * 4 * s.f_verbose.coeff(kDimf_verbose*i+6) 
                                      * diagonal.coeff(kDimc*i+5) * f_rsc_;
    Qff_rsc_.coeffRef(0, 0) += (dtau*dtau) * diagonal.coeff(kDimc*i  );
    Qff_rsc_.coeffRef(0, 1) += (dtau*dtau) * diagonal.coeff(kDimc*i+1);
    Qff_rsc_.coeffRef(1, 2) += (dtau*dtau) * diagonal.coeff(kDimc*i+2);
    Qff_rsc_.coeffRef(1, 3) += (dtau*dtau) * diagonal.coeff(kDimc*i+3);
    kkt_matrix.Qff().block<kDimf_verbose, kDimf>(kDimf_verbose*i, kDimf_verbose*i)
        .template bottomRows<2>().noalias() += Qff_rsc_;
    kkt_matrix.Qff().transpose().block<kDimf_verbose, kDimf>(kDimf_verbose*i, 
                                                             kDimf_verbose*i)
        .template bottomRows<2>().noalias() += Qff_rsc_;
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qfq().block(kDimf_verbose*i, 0, kDimf, robot.dimv()).noalias() 
        += (dtau*dtau) * diagonal.template segment<kDimf>(kDimc*i).asDiagonal() 
                       * dBaumgarte_verbose_dq_.block(kDimc*i, 0, kDimf, robot.dimv());
  }
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_matrix.Qfv().block(kDimf_verbose*i, 0, kDimf, robot.dimv()).noalias() 
        += (dtau*dtau) * diagonal.template segment<kDimf>(kDimc*i).asDiagonal() 
                       * dBaumgarte_verbose_dv_.block(kDimc*i, 0, kDimf, robot.dimv());
  }
}


inline void BaumgarteInequality::augmentCondensedResidual(
    Robot& robot, const double dtau, const SplitSolution& s, 
    const Eigen::VectorXd& condensed_residual, KKTResidual& kkt_residual) {
  assert(dtau > 0);
  assert(condensed_residual.size() == kDimc*robot.num_point_contacts());
  kkt_residual.la().noalias() 
      += dtau * dBaumgarte_verbose_da_.transpose() * condensed_residual;
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    kkt_residual.lf().coeffRef(kDimf_verbose*i+5)
        += dtau * (condensed_residual.coeff(kDimc*i  ) 
                    + condensed_residual.coeff(kDimc*i+1)
                    + 2 * s.f_verbose.coeff(kDimf_verbose*i+5) 
                        * condensed_residual.coeff(kDimc*i+5));
    kkt_residual.lf().coeffRef(kDimf_verbose*i+6)
        += dtau * (condensed_residual.coeff(kDimc*i+2) 
                    + condensed_residual.coeff(kDimc*i+3)
                    + 2 * s.f_verbose.coeff(kDimf_verbose*i+6) 
                        * condensed_residual.coeff(kDimc*i+5));
  }
  kkt_residual.lq().noalias() 
      += dtau * dBaumgarte_verbose_dq_.transpose() * condensed_residual;
  kkt_residual.lv().noalias() 
      += dtau * dBaumgarte_verbose_dv_.transpose() * condensed_residual;
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
    data.dslack.coeffRef(kDimc*i  ) += dtau * d.df().coeff(kDimf_verbose*i+5);
    data.dslack.coeffRef(kDimc*i+1) += dtau * d.df().coeff(kDimf_verbose*i+5);
    data.dslack.coeffRef(kDimc*i+2) += dtau * d.df().coeff(kDimf_verbose*i+6);
    data.dslack.coeffRef(kDimc*i+3) += dtau * d.df().coeff(kDimf_verbose*i+6);
    data.dslack.coeffRef(kDimc*i+5) 
        += 2 * dtau * s.f_verbose(kDimf_verbose*i+5) 
                    * d.df().coeff(kDimf_verbose*i+5);
    data.dslack.coeffRef(kDimc*i+5) 
        += 2 * dtau * s.f_verbose(kDimf_verbose*i+6) 
                    * d.df().coeff(kDimf_verbose*i+6);
  }
}

} // namespace idocp