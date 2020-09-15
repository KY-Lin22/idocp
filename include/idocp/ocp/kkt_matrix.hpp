#ifndef IDOCP_KKT_MATRIX_HPP_
#define IDOCP_KKT_MATRIX_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

///
/// @class KKTMatrix
/// @brief The KKT matrix of a time stage.
///
class KKTMatrix {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KKTMatrix(const Robot& robot);

  KKTMatrix();

  ~KKTMatrix();

  KKTMatrix(const KKTMatrix&) = default;

  KKTMatrix& operator=(const KKTMatrix&) = default;
 
  KKTMatrix(KKTMatrix&&) noexcept = default;

  KKTMatrix& operator=(KKTMatrix&&) noexcept = default;

  Eigen::Block<Eigen::MatrixXd> Ca();

  Eigen::Block<Eigen::MatrixXd> Cf();

  Eigen::Block<Eigen::MatrixXd> Cr();

  Eigen::Block<Eigen::MatrixXd> Cq();

  Eigen::Block<Eigen::MatrixXd> Cv();

  Eigen::Block<Eigen::MatrixXd> C_afr();

  Eigen::Block<Eigen::MatrixXd> C_qv();

  Eigen::Block<Eigen::MatrixXd> Qaa();

  Eigen::Block<Eigen::MatrixXd> Qaf();

  Eigen::Block<Eigen::MatrixXd> Qar();

  Eigen::Block<Eigen::MatrixXd> Qaq();

  Eigen::Block<Eigen::MatrixXd> Qav();

  Eigen::Block<Eigen::MatrixXd> Qfa();

  Eigen::Block<Eigen::MatrixXd> Qff();

  Eigen::Block<Eigen::MatrixXd> Qfr();

  Eigen::Block<Eigen::MatrixXd> Qfq();

  Eigen::Block<Eigen::MatrixXd> Qfv();

  Eigen::Block<Eigen::MatrixXd> Qra();

  Eigen::Block<Eigen::MatrixXd> Qrf();

  Eigen::Block<Eigen::MatrixXd> Qrr();

  Eigen::Block<Eigen::MatrixXd> Qrq();

  Eigen::Block<Eigen::MatrixXd> Qrv();

  Eigen::Block<Eigen::MatrixXd> Qqa();

  Eigen::Block<Eigen::MatrixXd> Qqf();

  Eigen::Block<Eigen::MatrixXd> Qqr();

  Eigen::Block<Eigen::MatrixXd> Qqq();

  Eigen::Block<Eigen::MatrixXd> Qqv();

  Eigen::Block<Eigen::MatrixXd> Qva();

  Eigen::Block<Eigen::MatrixXd> Qvf();

  Eigen::Block<Eigen::MatrixXd> Qvr();

  Eigen::Block<Eigen::MatrixXd> Qvq();

  Eigen::Block<Eigen::MatrixXd> Qvv();

  Eigen::Block<Eigen::MatrixXd> Qxx();

  Eigen::Block<Eigen::MatrixXd> Q_fr_q();

  Eigen::Block<Eigen::MatrixXd> Q_fr_v();

  Eigen::Block<Eigen::MatrixXd> Q_afr_afr();

  Eigen::Block<Eigen::MatrixXd> Q_afr_qv();

  Eigen::MatrixXd& costHessian();

  Eigen::MatrixXd& constraintsJacobian();

  void symmetrize();

  template <typename MatrixType>
  void invert(const double dtau, 
              const Eigen::MatrixBase<MatrixType>& kkt_matrix_inverse);

  void setZero();

  int dimKKT() const;

  int dimc() const;

  int dimf() const;

  int dimr() const;

  Eigen::MatrixXd Quu, Fqq, Fqq_prev;

private:
  static constexpr int kDimFloatingBase = 6;
  static constexpr int kDimf = 5;
  static constexpr int kDimr = 2;
  static constexpr int kDimfr = kDimf + kDimr;
  bool has_floating_base_;
  int dimv_, dimx_, dimfr_, dimf_, dimr_, dimc_, dimQ_, dimKKT_,
      a_begin_, f_begin_, r_begin_, q_begin_, v_begin_;
  Eigen::MatrixXd C_, Q_, Sc_, Sx_, FMinv_, C_H_inv_;

  template <typename MatrixType>
  void invertConstrainedHessian(
      const Eigen::MatrixBase<MatrixType>& hessian_inverse);

};

} // namespace idocp 

#include "idocp/ocp/kkt_matrix.hxx"

#endif // IDOCP_KKT_MATRIX_HPP_