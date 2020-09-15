#ifndef IDOCP_RICCATI_GAIN_HXX_
#define IDOCP_RICCATI_GAIN_HXX_

#include <assert.h>

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

inline RiccatiGain::RiccatiGain(const Robot& robot) 
  : K_(Eigen::MatrixXd::Zero(
          robot.dimv()+kDimfr*robot.num_point_contacts()+robot.dim_passive(), 
          2*robot.dimv())),
    k_(Eigen::VectorXd::Zero(
          robot.dimv()+kDimfr*robot.num_point_contacts()+robot.dim_passive())),
    dimv_(robot.dimv()),
    dimf_(kDimf*robot.num_point_contacts()),
    dimr_(kDimr*robot.num_point_contacts()),
    dimfr_(kDimfr*robot.num_point_contacts()),
    dimc_(robot.dim_passive()) {
}


inline RiccatiGain::RiccatiGain() 
  : K_(),
    k_(),
    dimv_(0),
    dimf_(0),
    dimr_(0),
    dimfr_(0),
    dimc_(0) {
}


inline RiccatiGain::~RiccatiGain() {
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Kaq() const {
  return K_.block(0, 0, dimv_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Kav() const {
  return K_.block(0, dimv_, dimv_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Kfrq() const {
  return K_.block(dimv_, 0, dimfr_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Kfrv() const {
  return K_.block(dimv_, dimv_, dimfr_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Kfq() const {
  return K_.block(dimv_, 0, dimf_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Kfv() const {
  return K_.block(dimv_, dimv_, dimf_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Krq() const {
  return K_.block(dimv_+dimf_, 0, dimr_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Krv() const {
  return K_.block(dimv_+dimf_, dimv_, dimr_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Kmuq() const {
  return K_.block(dimv_+dimfr_, 0, dimc_, dimv_);
}


inline const Eigen::Block<const Eigen::MatrixXd> RiccatiGain::Kmuv() const {
  return K_.block(dimv_+dimfr_, dimv_, dimc_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
RiccatiGain::ka() const {
  return k_.head(dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
RiccatiGain::kfr() const {
  return k_.segment(dimv_, dimfr_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
RiccatiGain::kf() const {
  return k_.segment(dimv_, dimf_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
RiccatiGain::kr() const {
  return k_.segment(dimv_+dimf_, dimr_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
RiccatiGain::kmu() const {
  return k_.segment(dimv_+dimfr_, dimc_);
}


template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
inline void RiccatiGain::computeFeedbackGain(
    const Eigen::MatrixBase<MatrixType1>& Ginv, 
    const Eigen::MatrixBase<MatrixType2>& Q_afr_qv, 
    const Eigen::MatrixBase<MatrixType3>& C_qv) {
  const int dimafr = dimv_ + dimfr_;
  assert(Ginv.rows() == dimafr+dimc_);
  assert(Ginv.cols() == dimafr+dimc_);
  assert(Q_afr_qv.rows() == dimafr);
  assert(Q_afr_qv.cols() == 2*dimv_);
  assert(C_qv.rows() == dimc_);
  assert(C_qv.cols() == 2*dimv_);
  K_.topRows(dimafr+dimc_).noalias() = - Ginv.leftCols(dimafr) * Q_afr_qv;
  K_.topRows(dimafr+dimc_).noalias() -= Ginv.rightCols(dimc_) * C_qv;
}


template <typename MatrixType, typename VectorType1, typename VectorType2>
inline void RiccatiGain::computeFeedforward(
    const Eigen::MatrixBase<MatrixType>& Ginv, 
    const Eigen::MatrixBase<VectorType1>& l_afr, 
    const Eigen::MatrixBase<VectorType2>& C) {
  const int dimafr = dimv_ + dimfr_;
  assert(Ginv.rows() == dimafr+dimc_);
  assert(Ginv.cols() == dimafr+dimc_);
  assert(l_afr.size() == dimafr);
  assert(C.size() == dimc_);
  k_.head(dimafr+dimc_).noalias() = - Ginv.leftCols(dimafr) * l_afr;
  k_.head(dimafr+dimc_).noalias() -= Ginv.rightCols(dimc_) * C;
}

} // namespace idocp 

#endif // IDOCP_RICCATI_GAIN_HXX_