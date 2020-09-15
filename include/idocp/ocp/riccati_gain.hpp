#ifndef IDOCP_RICCATI_GAIN_HPP_
#define IDOCP_RICCATI_GAIN_HPP_

#include <assert.h>

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

class RiccatiGain {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RiccatiGain(const Robot& robot);

  RiccatiGain();

  ~RiccatiGain();

  RiccatiGain(const RiccatiGain&) = default;

  RiccatiGain& operator=(const RiccatiGain&) = default;
 
  RiccatiGain(RiccatiGain&&) noexcept = default;

  RiccatiGain& operator=(RiccatiGain&&) noexcept = default;

  const Eigen::Block<const Eigen::MatrixXd> Kaq() const;

  const Eigen::Block<const Eigen::MatrixXd> Kav() const;

  const Eigen::Block<const Eigen::MatrixXd> Kfrq() const;

  const Eigen::Block<const Eigen::MatrixXd> Kfrv() const;

  const Eigen::Block<const Eigen::MatrixXd> Kmuq() const;

  const Eigen::Block<const Eigen::MatrixXd> Kmuv() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> ka() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> kfr() const;

  const Eigen::VectorBlock<const Eigen::VectorXd> kmu() const;

  template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
  void computeFeedbackGain(const Eigen::MatrixBase<MatrixType1>& Ginv, 
                           const Eigen::MatrixBase<MatrixType2>& Q_afr_qv, 
                           const Eigen::MatrixBase<MatrixType3>& C_qv);

  template <typename MatrixType, typename VectorType1, typename VectorType2>
  void computeFeedforward(const Eigen::MatrixBase<MatrixType>& Ginv, 
                          const Eigen::MatrixBase<VectorType1>& l_afr, 
                          const Eigen::MatrixBase<VectorType2>& C);

private:
  static constexpr int kDimfr = 7;
  int dimv_, dimfr_, dimc_;
  Eigen::MatrixXd K_;
  Eigen::VectorXd k_;

};

} // namespace idocp 

#include "idocp/ocp/riccati_gain.hxx"

#endif // IDOCP_RICCATI_GAIN_HPP_