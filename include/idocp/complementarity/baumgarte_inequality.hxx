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
                                         robot.dimv())) {
}


inline BaumgarteInequality::BaumgarteInequality() 
  : num_point_contacts_(0),
    dimc_(0),
    barrier_(0), 
    fraction_to_boundary_rate_(0),
    dBaumgarte_dq_(),
    dBaumgarte_dv_(),
    dBaumgarte_da_() {
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
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+5)+Baumgarte_residual_.coeff(kDimb*i  ));
    data.residual.coeffRef(kDimc*i+1) 
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+5)-Baumgarte_residual_.coeff(kDimb*i  ));
    data.residual.coeffRef(kDimc*i+2) 
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+6)+Baumgarte_residual_.coeff(kDimb*i+1));
    data.residual.coeffRef(kDimc*i+3) 
        -= dtau * (s.f_verbose.coeff(kDimf_verbose*i+6)-Baumgarte_residual_.coeff(kDimb*i+1));
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
    kkt_residual.la().noalias() -= dtau * data.dual(kDimc*i  ) * dBaumgarte_da_.row(kDimb*i  ).transpose();
    kkt_residual.la().noalias() += dtau * data.dual(kDimc*i+1) * dBaumgarte_da_.row(kDimb*i  ).transpose();
    kkt_residual.la().noalias() -= dtau * data.dual(kDimc*i+2) * dBaumgarte_da_.row(kDimb*i+1).transpose();
    kkt_residual.la().noalias() += dtau * data.dual(kDimc*i+3) * dBaumgarte_da_.row(kDimb*i+1).transpose();
    kkt_residual.la().noalias() -= dtau * data.dual(kDimc*i+4) * dBaumgarte_da_.row(kDimb*i+2).transpose();
    kkt_residual.lf().coeffRef(i*kDimf_verbose+5) 
        -= dtau * (data.dual(i*kDimc  ) + data.dual(i*kDimc+1) + 2*data.dual(i*kDimc+5)*s.f_verbose(i*kDimf_verbose+5));
    kkt_residual.lf().coeffRef(i*kDimf_verbose+6) 
        -= dtau * (data.dual(i*kDimc+2) + data.dual(i*kDimc+3) + 2*data.dual(i*kDimc+5)*s.f_verbose(i*kDimf_verbose+6));
    kkt_residual.lq().noalias() -= dtau * data.dual(kDimc*i  ) * dBaumgarte_dq_.row(kDimb*i  ).transpose();
    kkt_residual.lq().noalias() += dtau * data.dual(kDimc*i+1) * dBaumgarte_dq_.row(kDimb*i  ).transpose();
    kkt_residual.lq().noalias() -= dtau * data.dual(kDimc*i+2) * dBaumgarte_dq_.row(kDimb*i+1).transpose();
    kkt_residual.lq().noalias() += dtau * data.dual(kDimc*i+3) * dBaumgarte_dq_.row(kDimb*i+1).transpose();
    kkt_residual.lq().noalias() -= dtau * data.dual(kDimc*i+4) * dBaumgarte_dq_.row(kDimb*i+2).transpose();
    kkt_residual.lv().noalias() -= dtau * data.dual(kDimc*i  ) * dBaumgarte_dv_.row(kDimb*i  ).transpose();
    kkt_residual.lv().noalias() += dtau * data.dual(kDimc*i+1) * dBaumgarte_dv_.row(kDimb*i  ).transpose();
    kkt_residual.lv().noalias() -= dtau * data.dual(kDimc*i+2) * dBaumgarte_dv_.row(kDimb*i+1).transpose();
    kkt_residual.lv().noalias() += dtau * data.dual(kDimc*i+3) * dBaumgarte_dv_.row(kDimb*i+1).transpose();
    kkt_residual.lv().noalias() -= dtau * data.dual(kDimc*i+4) * dBaumgarte_dv_.row(kDimb*i+2).transpose();
  }
}


inline void BaumgarteInequality::computeSlackDirection(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    const SplitDirection& d, ConstraintComponentData& data) const {
  assert(dtau > 0);
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    data.dslack.coeffRef(kDimc*i  ) = dtau * dBaumgarte_da_.row(kDimb*i  ) * d.da();
    data.dslack.coeffRef(kDimc*i+1) = - dtau * dBaumgarte_da_.row(kDimb*i  ) * d.da();
    data.dslack.coeffRef(kDimc*i+2) = dtau * dBaumgarte_da_.row(kDimb*i+1) * d.da();
    data.dslack.coeffRef(kDimc*i+3) = - dtau * dBaumgarte_da_.row(kDimb*i+1) * d.da();
    data.dslack.coeffRef(kDimc*i+4) = dtau * dBaumgarte_da_.row(kDimb*i+2) * d.da();

    data.dslack.coeffRef(kDimc*i  ) += dtau * d.df().coeff(kDimf_verbose*i+5);
    data.dslack.coeffRef(kDimc*i+1) += dtau * d.df().coeff(kDimf_verbose*i+5);
    data.dslack.coeffRef(kDimc*i+2) += dtau * d.df().coeff(kDimf_verbose*i+6);
    data.dslack.coeffRef(kDimc*i+3) += dtau * d.df().coeff(kDimf_verbose*i+6);
    data.dslack.coeffRef(kDimc*i+5) = 2 * dtau * s.f_verbose(kDimf_verbose*i+5) * d.df().coeff(kDimf_verbose*i+5);
    data.dslack.coeffRef(kDimc*i+5) += 2 * dtau * s.f_verbose(kDimf_verbose*i+6) * d.df().coeff(kDimf_verbose*i+6);

    data.dslack.coeffRef(kDimc*i  ) += dtau * dBaumgarte_dq_.row(kDimb*i  ) * d.dq();
    data.dslack.coeffRef(kDimc*i+1) += - dtau * dBaumgarte_dq_.row(kDimb*i  ) * d.dq();
    data.dslack.coeffRef(kDimc*i+2) += dtau * dBaumgarte_dq_.row(kDimb*i+1) * d.dq();
    data.dslack.coeffRef(kDimc*i+3) += - dtau * dBaumgarte_dq_.row(kDimb*i+1) * d.dq();
    data.dslack.coeffRef(kDimc*i+4) += dtau * dBaumgarte_dq_.row(kDimb*i+2) * d.dq();

    data.dslack.coeffRef(kDimc*i  ) += dtau * dBaumgarte_dv_.row(kDimb*i  ) * d.dv();
    data.dslack.coeffRef(kDimc*i+1) += - dtau * dBaumgarte_dv_.row(kDimb*i  ) * d.dv();
    data.dslack.coeffRef(kDimc*i+2) += dtau * dBaumgarte_dv_.row(kDimb*i+1) * d.dv();
    data.dslack.coeffRef(kDimc*i+3) += - dtau * dBaumgarte_dv_.row(kDimb*i+1) * d.dv();
    data.dslack.coeffRef(kDimc*i+4) += dtau * dBaumgarte_dv_.row(kDimb*i+2) * d.dv();
  }
  data.dslack.noalias() -= data.residual;
}

} // namespace idocp