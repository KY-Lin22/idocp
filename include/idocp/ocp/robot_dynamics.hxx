#ifndef IDOCP_ROBOT_DYNAMICS_HXX_
#define IDOCP_ROBOT_DYNAMICS_HXX_

#include <assert.h>

namespace idocp {

inline RobotDynamics::RobotDynamics(const Robot& robot) 
  : lu_condensed_(Eigen::VectorXd::Zero(robot.dimv())),
    du_dq_(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    du_dv_(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    du_da_(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    du_df_(Eigen::MatrixXd::Zero(robot.dimv(), 
                                 kDimf*robot.num_point_contacts())),
    du_df_3D_(Eigen::MatrixXd::Zero(robot.dimv(), 
                                    3*robot.num_point_contacts())),
    Quu_du_dq_(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    Quu_du_dv_(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    Quu_du_da_(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    Quu_du_df_(Eigen::MatrixXd::Zero(robot.dimv(), 
                                     kDimf*robot.num_point_contacts())),
    has_floating_base_(robot.has_floating_base()),
    has_contacts_(robot.has_contacts()),
    dimf_(kDimf*robot.num_point_contacts()) {
}


inline RobotDynamics::RobotDynamics() 
  : lu_condensed_(),
    du_dq_(),
    du_dv_(),
    du_da_(),
    du_df_(),
    du_df_3D_(),
    Quu_du_dq_(),
    Quu_du_dv_(),
    Quu_du_da_(),
    Quu_du_df_(),
    has_floating_base_(false),
    has_contacts_(false),
    dimf_(0) {
}


inline RobotDynamics::~RobotDynamics() {
}


inline void RobotDynamics::augmentRobotDynamics(Robot& robot, const double dtau, 
                                                const SplitSolution& s, 
                                                KKTMatrix& kkt_matrix, 
                                                KKTResidual& kkt_residual) {
  assert(dtau > 0);
  linearizeInverseDynamics(robot, s, kkt_residual);
  // augment inverse dynamics
  kkt_residual.la().noalias() += dtau * du_da_.transpose() * s.beta;
  if (robot.has_contacts()) {
    kkt_residual.lf().noalias() += dtau * du_df_.transpose() * s.beta;
  }
  kkt_residual.lq().noalias() += dtau * du_dq_.transpose() * s.beta;
  kkt_residual.lv().noalias() += dtau * du_dv_.transpose() * s.beta;
  kkt_residual.lu.noalias() -= dtau * s.beta; 
  // augment floating base constraint
  linearizeFloatingBaseConstraint(robot, dtau, s, kkt_residual);
}


inline void RobotDynamics::condenseRobotDynamics(Robot& robot, 
                                                 const double dtau, 
                                                 const SplitSolution& s, 
                                                 KKTMatrix& kkt_matrix, 
                                                 KKTResidual& kkt_residual) {
  assert(dtau > 0);
  linearizeInverseDynamics(robot, s, kkt_residual);
  lu_condensed_.noalias() 
      = kkt_residual.lu + kkt_matrix.Quu * kkt_residual.u_res;
  // condense Newton residual
  kkt_residual.la().noalias() = du_da_.transpose() * lu_condensed_;
  if (robot.has_contacts()) {
    kkt_residual.lf().noalias() = du_df_.transpose() * lu_condensed_;
  }
  kkt_residual.lq().noalias() = du_dq_.transpose() * lu_condensed_;
  kkt_residual.lv().noalias() = du_dv_.transpose() * lu_condensed_;
  kkt_residual.lu.noalias() -= dtau * s.beta;   
  // condense Hessian
  Quu_du_da_.noalias() = kkt_matrix.Quu * du_da_;
  kkt_matrix.Qaa().noalias() = du_da_.transpose() * Quu_du_da_;
  if (robot.has_contacts()) {
    Quu_du_df_.noalias() = kkt_matrix.Quu * du_df_;
    kkt_matrix.Qaf().noalias() = du_da_.transpose() * Quu_du_df_; 
  }
  Quu_du_dq_.noalias() = kkt_matrix.Quu * du_dq_;
  Quu_du_dv_.noalias() = kkt_matrix.Quu * du_dv_;
  kkt_matrix.Qaq().noalias() = du_da_.transpose() * Quu_du_dq_;
  kkt_matrix.Qav().noalias() = du_da_.transpose() * Quu_du_dv_;
  if (robot.has_contacts()) {
    kkt_matrix.Qff().noalias() = du_df_.transpose() * Quu_du_df_;
    kkt_matrix.Qfq().noalias() = du_df_.transpose() * Quu_du_dq_;
    kkt_matrix.Qfv().noalias() = du_df_.transpose() * Quu_du_dv_;
  }
  kkt_matrix.Qqq().noalias() = du_dq_.transpose() * Quu_du_dq_;
  kkt_matrix.Qqv().noalias() = du_dq_.transpose() * Quu_du_dv_;
  kkt_matrix.Qvv().noalias() = du_dv_.transpose() * Quu_du_dv_;
  // condense floating base constraint
  if (robot.has_floating_base()) {
    kkt_residual.C() 
        = dtau * (kkt_residual.u_res.template head<kDimFloatingBase>()
                  + s.u.template head<kDimFloatingBase>());
    kkt_matrix.Ca() = dtau * du_da_.template topRows<kDimFloatingBase>();
    kkt_residual.la().noalias() += kkt_matrix.Ca().transpose() * s.mu;
    if (robot.has_contacts()) {
      kkt_matrix.Cf() = dtau * du_df_.template topRows<kDimFloatingBase>();
      kkt_residual.lf().noalias() += kkt_matrix.Cf().transpose() * s.mu;
    }
    kkt_matrix.Cq() = dtau * du_dq_.template topRows<kDimFloatingBase>();
    kkt_residual.lq().noalias() += kkt_matrix.Cq().transpose() * s.mu;
    kkt_matrix.Cv() = dtau * du_dv_.template topRows<kDimFloatingBase>();
    kkt_residual.lv().noalias() += kkt_matrix.Cv().transpose() * s.mu;
    kkt_residual.lu.template head<kDimFloatingBase>().noalias() += dtau * s.mu;
  }
}


inline void RobotDynamics::computeCondensedDirection(
    const double dtau, const KKTMatrix& kkt_matrix, 
    const KKTResidual& kkt_residual, SplitDirection& d) {
  assert(dtau > 0);
  d.du = kkt_residual.u_res;
  d.du.noalias() += du_dq_ * d.dq();
  d.du.noalias() += du_dv_ * d.dv();
  d.du.noalias() += du_da_ * d.da();
  if (has_contacts_) {
    d.du.noalias() += du_df_ * d.df();
  }
  d.dbeta.noalias() = (kkt_residual.lu  + kkt_matrix.Quu * d.du) / dtau;
  if (has_floating_base_) {
    d.dbeta.template head<kDimFloatingBase>().noalias() += d.dmu();
  }
}


inline double RobotDynamics::violationL1Norm(const double dtau, 
                                             const SplitSolution& s,
                                             KKTResidual& kkt_residual) const {
  assert(dtau > 0);
  double violation = dtau * kkt_residual.u_res.lpNorm<1>();
  if (has_floating_base_) {
    violation += dtau * s.u.template head<kDimFloatingBase>().lpNorm<1>();
  }
  return violation;
}


inline double RobotDynamics::computeViolationL1Norm(
    Robot& robot, const double dtau, const SplitSolution& s, 
    KKTResidual& kkt_residual) const {
  if (robot.has_contacts()) {
    robot.setContactForces(s.f_3D);
  }
  robot.RNEA(s.q, s.v, s.a, kkt_residual.u_res);
  kkt_residual.u_res.noalias() -= s.u;
  double violation = dtau * kkt_residual.u_res.lpNorm<1>();
  if (robot.has_floating_base()) {
    violation += dtau * s.u.template head<kDimFloatingBase>().lpNorm<1>();
  }
  return violation;
}


template <typename MatrixType1, typename MatrixType2, typename MatrixType3, 
          typename MatrixType4, typename MatrixType5, typename MatrixType6>
inline void RobotDynamics::getControlInputTorquesSensitivitiesWithRespectToState(
    const Eigen::MatrixBase<MatrixType1>& da_dq,
    const Eigen::MatrixBase<MatrixType2>& da_dv,
    const Eigen::MatrixBase<MatrixType3>& df_dq,
    const Eigen::MatrixBase<MatrixType4>& df_dv,
    const Eigen::MatrixBase<MatrixType5>& Kuq,
    const Eigen::MatrixBase<MatrixType6>& Kuv) const {
  assert(da_dq.rows() == da_dq.cols());
  assert(da_dv.rows() == da_dv.cols());
  assert(df_dq.rows() == dimf_);
  assert(df_dv.rows() == dimf_);
  assert(Kuq.rows() == Kuq.cols());
  assert(Kuv.rows() == Kuv.cols());
  assert(da_dq.rows() == da_dv.rows());
  assert(da_dq.rows() == df_dq.cols());
  assert(da_dq.rows() == df_dv.cols());
  assert(da_dq.rows() == Kuq.rows());
  assert(da_dq.rows() == Kuv.rows());
  const_cast<Eigen::MatrixBase<MatrixType5>&>(Kuq) = du_dq_;
  const_cast<Eigen::MatrixBase<MatrixType5>&>(Kuq).noalias() += du_da_ * da_dq;
  if (has_contacts_) {
    const_cast<Eigen::MatrixBase<MatrixType5>&>(Kuq).noalias() += du_df_ * df_dq;
  }
  const_cast<Eigen::MatrixBase<MatrixType6>&>(Kuv) = du_dv_;
  const_cast<Eigen::MatrixBase<MatrixType6>&>(Kuv).noalias() += du_da_ * da_dv;
  if (has_contacts_) {
    const_cast<Eigen::MatrixBase<MatrixType6>&>(Kuv).noalias() += du_df_ * df_dv;
  }
}


inline void RobotDynamics::linearizeInverseDynamics(Robot& robot, 
                                                    const SplitSolution& s, 
                                                    KKTResidual& kkt_residual) {
  if (robot.has_contacts()) {
    robot.setContactForces(s.f_3D);
  }
  robot.RNEA(s.q, s.v, s.a, kkt_residual.u_res);
  kkt_residual.u_res.noalias() -= s.u;
  robot.RNEADerivatives(s.q, s.v, s.a, du_dq_, du_dv_, du_da_);
  if (robot.has_contacts()) {
    robot.dRNEAPartialdFext(du_df_3D_);
    for (int i=0; i<robot.num_point_contacts(); ++i) {
      assert(s.f_3D.coeff(3*i  ) == s.f.coeff(5*i  )-s.f.coeff(5*i+1));
      assert(s.f_3D.coeff(3*i+1) == s.f.coeff(5*i+2)-s.f.coeff(5*i+3));
      assert(s.f_3D.coeff(3*i+2) == s.f.coeff(5*i+4));
      du_df_.col(kDimf*i  )  =   du_df_3D_.col(3*i  );
      du_df_.col(kDimf*i+1)  = - du_df_3D_.col(3*i  );
      du_df_.col(kDimf*i+2)  =   du_df_3D_.col(3*i+1);
      du_df_.col(kDimf*i+3)  = - du_df_3D_.col(3*i+1);
      du_df_.col(kDimf*i+4)  =   du_df_3D_.col(3*i+2);
    }
  }
}


inline void RobotDynamics::linearizeFloatingBaseConstraint(
    const Robot& robot, const double dtau, const SplitSolution& s, 
    KKTResidual& kkt_residual) const {
  assert(dtau > 0);
  if (robot.has_floating_base()) {
    kkt_residual.C() = dtau * s.u.template head<kDimFloatingBase>();
    kkt_residual.lu.template head<kDimFloatingBase>().noalias() += dtau * s.mu;
  }
}

} // namespace idocp 

#endif // IDOCP_ROBOT_DYNAMICS_HXX_