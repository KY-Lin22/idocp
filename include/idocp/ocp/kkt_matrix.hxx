#ifndef IDOCP_KKT_MATRIX_HXX_
#define IDOCP_KKT_MATRIX_HXX_

#include <assert.h>

#include "Eigen/LU"

namespace idocp {

inline KKTMatrix::KKTMatrix(const Robot& robot) 
  : Quu(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    Fqq(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    Fqq_prev(Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv())),
    C_(Eigen::MatrixXd::Zero(robot.dim_passive(), 
                             3*robot.dimv()+kDimfr*robot.num_point_contacts())),
    Q_(Eigen::MatrixXd::Zero(3*robot.dimv()+kDimfr*robot.num_point_contacts(), 
                             3*robot.dimv()+kDimfr*robot.num_point_contacts())),
    Sc_(Eigen::MatrixXd::Zero(robot.dim_passive(), robot.dim_passive())),
    Sx_(Eigen::MatrixXd::Zero(2*robot.dimv(), 2*robot.dimv())),
    FMinv_(Eigen::MatrixXd::Zero(
        2*robot.dimv(), 
        3*robot.dimv()+robot.dim_passive()+kDimfr*robot.num_point_contacts())),
    C_H_inv_(Eigen::MatrixXd::Zero(
        robot.dim_passive(), 3*robot.dimv()+kDimfr*robot.num_point_contacts())),
    has_floating_base_(robot.has_floating_base()),
    dimv_(robot.dimv()), 
    dimx_(2*robot.dimv()), 
    dimf_(kDimf*robot.num_point_contacts()), 
    dimr_(kDimr*robot.num_point_contacts()), 
    dimc_(robot.dim_passive()),
    dimQ_(3*robot.dimv()+kDimfr*robot.num_point_contacts()),
    dimKKT_(5*robot.dimv()+robot.dim_passive()+kDimfr*robot.num_point_contacts()),
    a_begin_(0),
    f_begin_(robot.dimv()),
    r_begin_(robot.dimv()+kDimf*robot.num_point_contacts()),
    q_begin_(robot.dimv()+kDimfr*robot.num_point_contacts()),
    v_begin_(2*robot.dimv()+kDimfr*robot.num_point_contacts()) {
}


inline KKTMatrix::KKTMatrix() 
  : Quu(),
    Fqq(),
    Fqq_prev(),
    C_(), 
    Q_(), 
    Sc_(), 
    Sx_(), 
    FMinv_(),
    C_H_inv_(),
    has_floating_base_(false),
    dimv_(0), 
    dimx_(0), 
    dimf_(0), 
    dimr_(0), 
    dimc_(0),
    dimQ_(0),
    dimKKT_(0),
    a_begin_(0),
    f_begin_(0),
    r_begin_(0),
    q_begin_(0),
    v_begin_(0) {
}


inline KKTMatrix::~KKTMatrix() {
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Ca() {
  return C_.block(0, a_begin_, dimc_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Cf() {
  return C_.block(0, f_begin_, dimc_, dimf_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Cr() {
  return C_.block(0, r_begin_, dimc_, dimr_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Cq() {
  return C_.block(0, q_begin_, dimc_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Cv() {
  return C_.block(0, v_begin_, dimc_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::C_afr() {
  return C_.block(0, a_begin_, dimc_, dimv_+dimf_+dimr_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::C_qv() {
  return C_.block(0, q_begin_, dimc_, dimx_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qaa() {
  return Q_.block(a_begin_, a_begin_, dimv_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qaf() {
  return Q_.block(a_begin_, f_begin_, dimv_, dimf_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qar() {
  return Q_.block(a_begin_, r_begin_, dimv_, dimr_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qaq() {
  return Q_.block(a_begin_, q_begin_, dimv_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qav() {
  return Q_.block(a_begin_, v_begin_, dimv_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qfa() {
  return Q_.block(f_begin_, a_begin_, dimf_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qff() {
  return Q_.block(f_begin_, f_begin_, dimf_, dimf_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qfr() {
  return Q_.block(f_begin_, r_begin_, dimf_, dimr_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qfq() {
  return Q_.block(f_begin_, q_begin_, dimf_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qfv() {
  return Q_.block(f_begin_, v_begin_, dimf_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qra() {
  return Q_.block(r_begin_, a_begin_, dimr_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qrf() {
  return Q_.block(r_begin_, f_begin_, dimr_, dimf_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qrr() {
  return Q_.block(r_begin_, r_begin_, dimr_, dimr_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qrq() {
  return Q_.block(r_begin_, q_begin_, dimr_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qrv() {
  return Q_.block(r_begin_, v_begin_, dimr_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qqa() {
  return Q_.block(q_begin_, a_begin_, dimv_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qqf() {
  return Q_.block(q_begin_, f_begin_, dimv_, dimf_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qqr() {
  return Q_.block(q_begin_, r_begin_, dimv_, dimr_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qqq() {
  return Q_.block(q_begin_, q_begin_, dimv_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qqv() {
  return Q_.block(q_begin_, v_begin_, dimv_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qva() {
  return Q_.block(v_begin_, a_begin_, dimv_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qvf() {
  return Q_.block(v_begin_, f_begin_, dimv_, dimf_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qvr() {
  return Q_.block(v_begin_, r_begin_, dimv_, dimr_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qvq() {
  return Q_.block(v_begin_, q_begin_, dimv_, dimv_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qvv() {
  return Q_.block(v_begin_, v_begin_, dimv_, dimv_);
}



inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Qxx() {
  return Q_.block(q_begin_, q_begin_, dimx_, dimx_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Q_afr_afr() {
  return Q_.block(a_begin_, a_begin_, dimv_+dimf_+dimr_, dimv_+dimf_+dimr_);
}


inline Eigen::Block<Eigen::MatrixXd> KKTMatrix::Q_afr_qv() {
  return Q_.block(a_begin_, q_begin_, dimv_+dimf_+dimr_, 2*dimv_);
}


inline Eigen::MatrixXd& KKTMatrix::costHessian() {
  return Q_;
}


inline Eigen::MatrixXd& KKTMatrix::constraintsJacobian() {
  return C_;
}


inline void KKTMatrix::symmetrize() {
  Q_.triangularView<Eigen::StrictlyLower>() 
      = Q_.transpose().triangularView<Eigen::StrictlyLower>();
}


template <typename MatrixType>
inline void KKTMatrix::invert(
    const double dtau, 
    const Eigen::MatrixBase<MatrixType>& kkt_matrix_inverse) {
  assert(kkt_matrix_inverse.rows() == (dimx_+dimc_+dimQ_));
  assert(kkt_matrix_inverse.cols() == (dimx_+dimc_+dimQ_));
  // Forms the Schur complement matrix
  const int dimcQ = dimc_ + dimQ_;
  invertConstrainedHessian(
      const_cast<Eigen::MatrixBase<MatrixType>&>(kkt_matrix_inverse)
          .bottomRightCorner(dimcQ, dimcQ));
  if (has_floating_base_) {
    FMinv_.topLeftCorner(dimv_, dimcQ) 
        = dtau * kkt_matrix_inverse.block(dimx_+dimc_+v_begin_, dimx_, dimv_, dimcQ);
    FMinv_.topLeftCorner(dimv_, dimcQ).template topRows<kDimFloatingBase>().noalias()
        += Fqq.template topLeftCorner<kDimFloatingBase, kDimFloatingBase>() 
            * kkt_matrix_inverse.block(dimx_+dimc_+q_begin_, dimx_, kDimFloatingBase, dimcQ);
    FMinv_.topLeftCorner(dimv_, dimcQ).bottomRows(dimv_-kDimFloatingBase).noalias()
        -= kkt_matrix_inverse.block(dimx_+dimc_+q_begin_+kDimFloatingBase, dimx_, dimv_-kDimFloatingBase, dimcQ);
  }
  else {
    FMinv_.topLeftCorner(dimv_, dimcQ).noalias() 
        = dtau * kkt_matrix_inverse.block(dimx_+dimc_+v_begin_, dimx_, dimv_, dimcQ)
          - kkt_matrix_inverse.block(dimx_+dimc_+q_begin_, dimx_, dimv_, dimcQ);
  }
  FMinv_.bottomLeftCorner(dimv_, dimcQ).noalias() 
      = dtau * kkt_matrix_inverse.block(dimx_+dimc_+a_begin_, dimx_, dimv_, dimcQ)
        - kkt_matrix_inverse.block(dimx_+dimc_+v_begin_, dimx_, dimv_, dimcQ);
  if (has_floating_base_) {
    Sx_.topLeftCorner(dimv_, dimv_) 
        = dtau * FMinv_.block(0, dimc_+v_begin_, dimv_, dimv_);
    Sx_.topLeftCorner(dimv_, dimv_).template leftCols<kDimFloatingBase>().noalias()
        += FMinv_.block(0, dimc_+q_begin_, dimv_, dimv_).template leftCols<kDimFloatingBase>() 
            * Fqq.template topLeftCorner<kDimFloatingBase, kDimFloatingBase>().transpose();
    Sx_.topLeftCorner(dimv_, dimv_).rightCols(dimv_-kDimFloatingBase).noalias()
        -= FMinv_.block(0, dimc_+q_begin_, dimv_, dimv_).rightCols(dimv_-kDimFloatingBase);
    Sx_.bottomLeftCorner(dimv_, dimv_) 
        = dtau * FMinv_.block(dimv_, dimc_+v_begin_, dimv_, dimv_);
    Sx_.bottomLeftCorner(dimv_, dimv_).template leftCols<kDimFloatingBase>().noalias() 
        += FMinv_.block(dimv_, dimc_+q_begin_, dimv_, dimv_).template leftCols<kDimFloatingBase>() 
            * Fqq.template topLeftCorner<kDimFloatingBase, kDimFloatingBase>().transpose();
    Sx_.bottomLeftCorner(dimv_, dimv_).rightCols(dimv_-kDimFloatingBase).noalias() 
        -= FMinv_.block(dimv_, dimc_+q_begin_, dimv_, dimv_).rightCols(dimv_-kDimFloatingBase);
  }
  else {
    Sx_.topLeftCorner(dimv_, dimv_).noalias() 
        = dtau * FMinv_.block(0, dimc_+v_begin_, dimv_, dimv_) 
            - FMinv_.block(0, dimc_+q_begin_, dimv_, dimv_);
    Sx_.bottomLeftCorner(dimv_, dimv_).noalias() 
        = dtau * FMinv_.block(dimv_, dimc_+v_begin_, dimv_, dimv_) 
            - FMinv_.block(dimv_, dimc_+q_begin_, dimv_, dimv_);
  }
  Sx_.topRightCorner(dimv_, dimv_).noalias() 
      = dtau * FMinv_.block(0, dimc_+a_begin_, dimv_, dimv_)
        - FMinv_.block(0, dimc_+v_begin_, dimv_, dimv_);
  Sx_.bottomRightCorner(dimv_, dimv_).noalias() 
      = dtau * FMinv_.block(dimv_, dimc_+a_begin_, dimv_, dimv_)
        - FMinv_.block(dimv_, dimc_+v_begin_, dimv_, dimv_);
  const_cast<Eigen::MatrixBase<MatrixType>&>(kkt_matrix_inverse)
      .topLeftCorner(dimx_, dimx_).noalias()
      = - Sx_.llt().solve(Eigen::MatrixXd::Identity(dimx_, dimx_));
  const_cast<Eigen::MatrixBase<MatrixType>&>(kkt_matrix_inverse)
      .topRightCorner(dimx_, dimcQ).noalias()
      = - kkt_matrix_inverse.topLeftCorner(dimx_, dimx_)
          * FMinv_.topLeftCorner(dimx_, dimcQ);
  const_cast<Eigen::MatrixBase<MatrixType>&>(kkt_matrix_inverse)
      .bottomLeftCorner(dimcQ, dimx_)
      = kkt_matrix_inverse.topRightCorner(dimx_, dimcQ).transpose();
  const_cast<Eigen::MatrixBase<MatrixType>&>(kkt_matrix_inverse)
      .bottomRightCorner(dimcQ, dimcQ).noalias()
      -= kkt_matrix_inverse.topRightCorner(dimx_, dimcQ).transpose()
              * Sx_ * kkt_matrix_inverse.topRightCorner(dimx_, dimcQ);
}


inline void KKTMatrix::setZero() {
  Quu.setZero();
  Fqq.setZero();
  C_.setZero();
  Q_.setZero();
}


inline int KKTMatrix::dimKKT() const {
  return dimKKT_;
}


inline int KKTMatrix::dimc() const {
  return dimc_;
}


inline int KKTMatrix::dimf() const {
  return dimf_;
}


inline int KKTMatrix::dimr() const {
  return dimr_;
}


template <typename MatrixType>
inline void KKTMatrix::invertConstrainedHessian( 
    const Eigen::MatrixBase<MatrixType>& H_inv) {
  assert(H_inv.rows() == dimQ_+dimc_);
  assert(H_inv.cols() == dimQ_+dimc_);
  if (dimc_ > 0) {
    const_cast<Eigen::MatrixBase<MatrixType>&>(H_inv)
        .bottomRightCorner(dimQ_, dimQ_).noalias()
        = Q_.llt().solve(Eigen::MatrixXd::Identity(dimQ_, dimQ_));
    C_H_inv_.noalias() = C_ * H_inv.bottomRightCorner(dimQ_, dimQ_);
    Sc_.noalias() = C_H_inv_ * C_.transpose();
    const_cast<Eigen::MatrixBase<MatrixType>&>(H_inv)
        .topLeftCorner(dimc_, dimc_).noalias()
        = - Sc_.llt().solve(Eigen::MatrixXd::Identity(dimc_, dimc_));
    const_cast<Eigen::MatrixBase<MatrixType>&>(H_inv)
        .topRightCorner(dimc_, dimQ_).noalias()
        = - H_inv.topLeftCorner(dimc_, dimc_) * C_H_inv_;
    const_cast<Eigen::MatrixBase<MatrixType>&>(H_inv)
        .bottomLeftCorner(dimQ_, dimc_)
        = H_inv.topRightCorner(dimc_, dimQ_).transpose();
    const_cast<Eigen::MatrixBase<MatrixType>&>(H_inv)
        .bottomRightCorner(dimQ_, dimQ_).noalias()
        -= H_inv.topRightCorner(dimc_, dimQ_).transpose()
              * Sc_ * H_inv.topRightCorner(dimc_, dimQ_);
  }
  else {
    const_cast<Eigen::MatrixBase<MatrixType>&>(H_inv).noalias()
        = Q_.llt().solve(Eigen::MatrixXd::Identity(dimQ_, dimQ_));
  }
}

} // namespace idocp 

#endif // IDOCP_KKT_MATRIX_HXX_