#ifndef IDOCP_SPLIT_DIRECTION_HPP_
#define IDOCP_SPLIT_DIRECTION_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

class SplitDirection {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SplitDirection(const Robot& robot);

  SplitDirection();

  ~SplitDirection();

  SplitDirection(const SplitDirection&) = default;

  SplitDirection& operator=(const SplitDirection&) = default;
 
  SplitDirection(SplitDirection&&) noexcept = default;

  SplitDirection& operator=(SplitDirection&&) noexcept = default;

  Eigen::VectorBlock<Eigen::VectorXd> dlmd();

  Eigen::VectorBlock<Eigen::VectorXd> dgmm();

  Eigen::VectorBlock<Eigen::VectorXd> dmu();

  Eigen::VectorBlock<Eigen::VectorXd> da();

  Eigen::VectorBlock<Eigen::VectorXd> dfr();

  Eigen::VectorBlock<Eigen::VectorXd> df();

  Eigen::VectorBlock<Eigen::VectorXd> dr();

  Eigen::VectorBlock<Eigen::VectorXd> dq();

  Eigen::VectorBlock<Eigen::VectorXd> dv();

  Eigen::VectorBlock<Eigen::VectorXd> dx();

  const Eigen::VectorBlock<const Eigen::VectorXd> dlmd() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> dgmm() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> dmu() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> da() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> dfr() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> df() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> dr() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> dq() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> dv() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> dx() const;

  void setZero();

  int dimKKT() const;

  int dimc() const;

  int dimf() const;

  int dimr() const;

  // condensed directions
  Eigen::VectorXd du, dbeta, split_direction;

private:
  static constexpr int kDimf = 5;
  static constexpr int kDimr = 2;
  static constexpr int kDimfr = kDimf + kDimr;
  int dimv_, dimx_, dimfr_, dimf_, dimr_, dimc_, dimKKT_;

};

} // namespace idocp 

#include "idocp/ocp/split_direction.hxx"

#endif // IDOCP_SPLIT_OCP_DIRECTION_HPP_