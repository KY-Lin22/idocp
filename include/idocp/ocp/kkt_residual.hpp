#ifndef IDOCP_KKT_RESIDUAL_HPP_
#define IDOCP_KKT_RESIDUAL_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

class KKTResidual {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KKTResidual(const Robot& robot);

  KKTResidual();

  ~KKTResidual();

  KKTResidual(const KKTResidual&) = default;

  KKTResidual& operator=(const KKTResidual&) = default;
 
  KKTResidual(KKTResidual&&) noexcept = default;

  KKTResidual& operator=(KKTResidual&&) noexcept = default;

  Eigen::VectorBlock<Eigen::VectorXd> Fq();

  Eigen::VectorBlock<Eigen::VectorXd> Fv();

  Eigen::VectorBlock<Eigen::VectorXd> Fx();

  Eigen::VectorBlock<Eigen::VectorXd> C();

  Eigen::VectorBlock<Eigen::VectorXd> la();

  Eigen::VectorBlock<Eigen::VectorXd> lf();

  Eigen::VectorBlock<Eigen::VectorXd> lr();

  Eigen::VectorBlock<Eigen::VectorXd> lq();

  Eigen::VectorBlock<Eigen::VectorXd> lv();

  Eigen::VectorBlock<Eigen::VectorXd> lx();

  Eigen::VectorBlock<Eigen::VectorXd> l_afr();

  const Eigen::VectorBlock<const Eigen::VectorXd> Fq() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> Fv() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> Fx() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> C() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> la() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> lf() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> lr() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> lq() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> lv() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> lx() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> l_afr() const;

  double squaredKKTErrorNorm(const double dtau) const;

  void setZero();

  int dimKKT() const;

  int dimc() const;

  int dimf() const;

  int dimr() const;

  Eigen::VectorXd lu, u_res, KKT_residual;

private:
  static constexpr int kDimf = 5;
  static constexpr int kDimr = 2;
  static constexpr int kDimfr = kDimf + kDimr;
  int dimv_, dimx_, dimfr_, dimf_, dimr_, dimc_, dimKKT_;

};

} // namespace idocp 

#include "idocp/ocp/kkt_residual.hxx"

#endif // IDOCP_KKT_RESIDUAL_HPP_