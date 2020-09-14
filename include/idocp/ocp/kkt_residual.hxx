#ifndef IDOCP_KKT_RESIDUAL_HXX_
#define IDOCP_KKT_RESIDUAL_HXX_

namespace idocp {

inline KKTResidual::KKTResidual(const Robot& robot) 
  : lu(Eigen::VectorXd::Zero(robot.dimv())),
    u_res(Eigen::VectorXd::Zero(robot.dimv())),
    KKT_residual(Eigen::VectorXd::Zero(
        5*robot.dimv()+robot.dim_passive()+kDimfr*robot.num_point_contacts())),
    dimv_(robot.dimv()), 
    dimx_(2*robot.dimv()), 
    dimfr_(kDimfr*robot.num_point_contacts()), 
    dimf_(kDimf*robot.num_point_contacts()), 
    dimr_(kDimr*robot.num_point_contacts()), 
    dimc_(robot.dim_passive()),
    dimKKT_(
        5*robot.dimv()+robot.dim_passive()+kDimfr*robot.num_point_contacts()) {
}


inline KKTResidual::KKTResidual() 
  : lu(),
    u_res(),
    KKT_residual(), 
    dimv_(0), 
    dimx_(0), 
    dimfr_(0), 
    dimf_(0), 
    dimr_(0),
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
  return KKT_residual.head(dimx_);
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


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::lr() {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimf_, dimr_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::lq() {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimfr_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::lv() {
  return KKT_residual.segment(dimx_+dimc_+2*dimv_+dimfr_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::lx() {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimfr_, dimx_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> KKTResidual::l_afr() {
  return KKT_residual.segment(dimx_+dimc_, dimv_+dimfr_);
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
  return KKT_residual.head(dimx_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::C() const {
  return KKT_residual.segment(dimx_, dimc_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::la() const {
  return KKT_residual.segment(dimx_+dimc_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::lf() const {
  return KKT_residual.segment(dimx_+dimc_+dimv_, dimf_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::lr() const {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimf_, dimr_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::lq() const {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimfr_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::lv() const {
  return KKT_residual.segment(dimx_+dimc_+2*dimv_+dimfr_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::lx() const {
  return KKT_residual.segment(dimx_+dimc_+dimv_+dimfr_, dimx_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
KKTResidual::l_afr() const {
  return KKT_residual.segment(dimx_+dimc_, dimv_+dimfr_);
}


inline double KKTResidual::squaredKKTErrorNorm(const double dtau) const {
  double error = KKT_residual.squaredNorm();
  error += lu.squaredNorm();
  error += dtau * dtau * u_res.squaredNorm();
  return error;
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


inline int KKTResidual::dimr() const {
  return dimr_;
}

} // namespace idocp 

#endif // IDOCP_KKT_RESIDUAL_HXX_