#ifndef IDOCP_RICCATI_MATRIX_INVERTER_HXX_
#define IDOCP_RICCATI_MATRIX_INVERTER_HXX_

#include "Eigen/LU"

#include <assert.h>

namespace idocp {

inline RiccatiMatrixInverter::RiccatiMatrixInverter(const Robot& robot) 
  : dimv_(robot.dimv()),
    dimfr_(kDimfr*robot.num_point_contacts()),
    dimc_(robot.dim_passive()),
    dimafr_(robot.dimv()+kDimfr*robot.num_point_contacts()),
    G_inv_(Eigen::MatrixXd::Zero(
        robot.dimv()+kDimfr*robot.num_point_contacts()+robot.dim_passive(), 
        robot.dimv()+kDimfr*robot.num_point_contacts()+robot.dim_passive())),
    Sc_(Eigen::MatrixXd::Zero(robot.dim_passive(), 
                              robot.dim_passive())),
    G_inv_C_afr_trans_(Eigen::MatrixXd::Zero(
        robot.dimv()+kDimfr*robot.num_point_contacts(), robot.dim_passive())) {
}


inline RiccatiMatrixInverter::RiccatiMatrixInverter() 
  : dimv_(0),
    dimfr_(0),
    dimc_(0),
    dimafr_(0),
    G_inv_(),
    Sc_(),
    G_inv_C_afr_trans_() {
}


inline RiccatiMatrixInverter::~RiccatiMatrixInverter() {
}


template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
inline void RiccatiMatrixInverter::invert(
    const Eigen::MatrixBase<MatrixType1>& G, 
    const Eigen::MatrixBase<MatrixType2>& C_afr, 
    const Eigen::MatrixBase<MatrixType3>& G_inv) {
  assert(G.rows() == dimafr_);
  assert(G.cols() == dimafr_);
  assert(C_afr.rows() == dimc_);
  assert(C_afr.cols() == dimafr_);
  assert(G_inv.rows() == dimafr_+dimc_);
  assert(G_inv.cols() == dimafr_+dimc_);
  if (dimc_ > 0) {
    const_cast<Eigen::MatrixBase<MatrixType3>&> (G_inv)
        .topLeftCorner(dimafr_, dimafr_).noalias()
        = G.llt().solve(Eigen::MatrixXd::Identity(dimafr_, dimafr_));
    G_inv_C_afr_trans_.topLeftCorner(dimafr_, dimc_).noalias() 
        = G_inv.topLeftCorner(dimafr_, dimafr_) * C_afr.transpose();
    Sc_.topLeftCorner(dimc_, dimc_).noalias()
        = C_afr * G_inv_C_afr_trans_.topLeftCorner(dimafr_, dimc_);
    const_cast<Eigen::MatrixBase<MatrixType3>&> (G_inv)
        .block(dimafr_, dimafr_, dimc_, dimc_).noalias() 
        = - Sc_.topLeftCorner(dimc_, dimc_)
               .llt().solve(Eigen::MatrixXd::Identity(dimc_, dimc_));
    const_cast<Eigen::MatrixBase<MatrixType3>&> (G_inv)
        .block(0, dimafr_, dimafr_, dimc_).noalias() 
        = - G_inv_C_afr_trans_.topLeftCorner(dimafr_, dimc_)
            * G_inv.block(dimafr_, dimafr_, dimc_, dimc_);
    const_cast<Eigen::MatrixBase<MatrixType3>&> (G_inv)
        .block(dimafr_, 0, dimc_, dimafr_).noalias() 
        = G_inv.block(0, dimafr_, dimafr_, dimc_).transpose();
    const_cast<Eigen::MatrixBase<MatrixType3>&> (G_inv)
        .topLeftCorner(dimafr_, dimafr_).noalias()
        -= G_inv.block(0, dimafr_, dimafr_, dimc_) 
            * Sc_.topLeftCorner(dimc_, dimc_)
            * G_inv.block(0, dimafr_, dimafr_, dimc_).transpose();
  }
  else {
    const_cast<Eigen::MatrixBase<MatrixType3>&> (G_inv).noalias()
        = G.llt().solve(Eigen::MatrixXd::Identity(dimafr_, dimafr_));
  }
}


template <typename MatrixType1, typename MatrixType2>
inline void RiccatiMatrixInverter::invert(
    const Eigen::MatrixBase<MatrixType1>& G, 
    const Eigen::MatrixBase<MatrixType2>& C_afr) {
  assert(G.rows() == dimafr_);
  assert(G.cols() == dimafr_);
  assert(C_afr.rows() == dimc_);
  assert(C_afr.cols() == dimafr_);
  if (dimc_ > 0) {
    G_inv_.topLeftCorner(dimafr_, dimafr_).noalias()
        = G.llt().solve(Eigen::MatrixXd::Identity(dimafr_, dimafr_));
    G_inv_C_afr_trans_.topLeftCorner(dimafr_, dimc_).noalias() 
        = G_inv_.topLeftCorner(dimafr_, dimafr_) * C_afr.transpose();
    Sc_.topLeftCorner(dimc_, dimc_).noalias() 
        = C_afr * G_inv_C_afr_trans_.topLeftCorner(dimafr_, dimc_);
    G_inv_.block(dimafr_, dimafr_, dimc_, dimc_).noalias() 
        = - Sc_.topLeftCorner(dimc_, dimc_)
               .llt().solve(Eigen::MatrixXd::Identity(dimc_, dimc_));
    G_inv_.block(0, dimafr_, dimafr_, dimc_).noalias() 
        = - G_inv_C_afr_trans_.topLeftCorner(dimafr_, dimc_)
              * G_inv_.block(dimafr_, dimafr_, dimc_, dimc_);
    G_inv_.block(dimafr_, 0, dimc_, dimafr_).noalias() 
        = G_inv_.block(0, dimafr_, dimafr_, dimc_).transpose();
    G_inv_.topLeftCorner(dimafr_, dimafr_).noalias()
        -= G_inv_.block(0, dimafr_, dimafr_, dimc_) 
            * Sc_.topLeftCorner(dimc_, dimc_)
            * G_inv_.block(0, dimafr_, dimafr_, dimc_).transpose();
  }
  else {
    G_inv_.topLeftCorner(dimafr_, dimafr_).noalias()
        = G.llt().solve(Eigen::MatrixXd::Identity(dimafr_, dimafr_));
  }
}


template <typename MatrixType>
void RiccatiMatrixInverter::getInverseMatrix(
    const Eigen::MatrixBase<MatrixType>& G_inv) {
  assert(G_inv.rows() == dimafr_+dimc_);
  assert(G_inv.cols() == dimafr_+dimc_);
  const_cast<Eigen::MatrixBase<MatrixType>&> (G_inv).noalias()
      = G_inv_.topLeftCorner(dimafr_+dimc_, dimafr_+dimc_);
}


// template <typename MatrixType1, typename MatrixType2>
// void RiccatiMatrixInverter::firstOrderCorrection(
//     const double dtau, const Eigen::MatrixBase<MatrixType1>& dPvv, 
//     const Eigen::MatrixBase<MatrixType2>& G_inv) {
//   assert(dtau > 0);
//   assert(dPvv.rows() == dimv_);
//   assert(dPvv.cols() == dimv_);
//   assert(G_inv.rows() == dimafr_+dimc_);
//   assert(G_inv.cols() == dimafr_+dimc_);
//   if (dimc_ > 0) {
//     (const_cast<Eigen::MatrixBase<MatrixType2>&>(G_inv))
//         .topLeftCorner(dimv_, dimv_).noalias()
//         += dtau * dtau * G_inv_.block(0, 0, dimv_, dimv_) * dPvv 
//                        * G_inv_.block(0, 0, dimv_, dimv_);
//     (const_cast<Eigen::MatrixBase<MatrixType2>&>(G_inv))
//         .topRightCorner(dimv_, dimfr_+dimc_).noalias()
//         += dtau * dtau * G_inv_.block(0, 0, dimv_, dimv_) * dPvv 
//                        * G_inv_.block(0, dimv_, dimv_, dimfr_+dimc_);
//     (const_cast<Eigen::MatrixBase<MatrixType2>&>(G_inv))
//         .bottomLeftCorner(dimfr_+dimc_, dimv_).noalias()
//         += dtau * dtau * G_inv_.block(dimv_, 0, dimfr_+dimc_, dimv_) * dPvv 
//                        * G_inv_.block(0, 0, dimv_, dimv_);
//     (const_cast<Eigen::MatrixBase<MatrixType2>&>(G_inv))
//         .bottomRightCorner(dimfr_+dimc_, dimfr_+dimc_).noalias()
//         += dtau * dtau * G_inv_.block(dimv_, 0, dimfr_+dimc_, dimv_) * dPvv 
//                        * G_inv_.block(0, dimv_, dimv_, dimfr_+dimc_);
//   }
//   else {
//     (const_cast<Eigen::MatrixBase<MatrixType2>&>(G_inv)).noalias()
//         += dtau * dtau * G_inv_.topLeftCorner(dimv_, dimv_) * dPvv 
//                        * G_inv_.topLeftCorner(dimv_, dimv_);
//   }
// }

} // namespace idocp


#endif // IDOCP_RICCATI_MATRIX_INVERTER_HXX_