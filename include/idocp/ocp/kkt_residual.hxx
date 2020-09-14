#ifndef IDOCP_KKT_RESIDUAL_HXX_
#define IDOCP_KKT_RESIDUAL_HXX_

namespace idocp {

inline KKTResidual::KKTResidual(const Robot& robot) 
  : lu(Eigen::VectorXd::Zero(robot.dimv())),
    u_res(Eigen::VectorXd::Zero(robot.dimv())),
    KKT_residual(Eigen::VectorXd::Zero(
        5*robot.dimv()+robot.dim_passive()+7*robot.num_point_contacts())),
    dimv_(robot.dimv()), 
    dimx_(2*robot.dimv()), 
    dimf_(7*robot.num_point_contacts()), 
    dimc_(robot.dim_passive()),
    dimKKT_(5*robot.dimv()+robot.dim_passive()+7*robot.num_point_contacts()) {
}


inline KKTResidual::KKTResidual() 
  : lu(),
    u_res(),
    KKT_residual(), 
    dimv_(0), 
    dimx_(0), 
    dimf_(0), 
    dimc_(0),
    dimKKT_(0) {
}


inline KKTResidual::~KKTResidual() {
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::Fq() {
  return KKT_residual.head(dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::Fv() {
  return KKT_residual.segment(dimv_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::Fx() {
  return KKT_residual.segment(0, dimx_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::C() {
  return KKT_residual.segment(dimx_, dimc_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::la() {
  return KKT_residual.segment(dimx_+dimc_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::lf() {
  return KKT_residual.segment(dimx_+dimc_+dimv_, dimf_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::lq() {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimf_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::lv() {
  return KKT_residual.segment(dimx_+dimc_+2*dimv_+dimf_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::lx() {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimf_, dimx_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::laf() {
  return KKT_residual.segment(dimx_+dimc_, dimv_+dimf_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::Fq() const {
  return KKT_residual.head(dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::Fv() const {
  return KKT_residual.segment(dimv_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::Fx() const {
  return KKT_residual.segment(0, dimx_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> KKTResidual::C() const {
  return KKT_residual.segment(dimx_, dimc_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> KKTResidual::la() const {
  return KKT_residual.segment(dimx_+dimc_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> KKTResidual::lf() const {
  return KKT_residual.segment(dimx_+dimc_+dimv_, dimf_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> KKTResidual::lq() const {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimf_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> KKTResidual::lv() const {
  return KKT_residual.segment(dimx_+dimc_+2*dimv_+dimf_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> KKTResidual::lx() const {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimf_, dimx_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> KKTResidual::laf() const {
  return KKT_residual.segment(dimx_+dimc_, dimv_+dimf_);
}


inline double KKTResidual::squaredKKTErrorNorm(const double dtau) const {
  double error = KKT_residual.head(dimKKT_).squaredNorm();
  error += lu.squaredNorm();
  error += dtau * dtau * u_res.squaredNorm();
  return error;
}


inline void KKTResidual::setZeroMinimum() {
  lu.setZero();
  KKT_residual.segment(dimx_+dimc_, 3*dimv_+dimf_).setZero();
}


inline void KKTResidual::setZero() {
  lu.setZero();
  u_res.setZero();
  KKT_residual.setZero();
}


inline int KKTResidual::dimKKT() const {
  return dimKKT_;
}


inline int KKTResidual::dimc() const {
  return dimc_;
}


inline int KKTResidual::dimf() const {
  return dimf_;
}

} // namespace idocp 

#endif // IDOCP_KKT_RESIDUAL_HXX_