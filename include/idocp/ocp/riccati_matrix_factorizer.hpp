#ifndef IDOCP_RICCATI_MATRIX_FACTORIZER_HPP_
#define IDOCP_RICCATI_MATRIX_FACTORIZER_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

class RiccatiMatrixFactorizer {
public:
  // Constructor.
  // Argments:
  //    robot: The robot model that has been already initialized.
  RiccatiMatrixFactorizer(const Robot& robot);

  // Default constructor.
  RiccatiMatrixFactorizer();

  // Destructor.
  ~RiccatiMatrixFactorizer();
 
  // Use default copy constructor.
  RiccatiMatrixFactorizer(const RiccatiMatrixFactorizer&) = default;

  // Use default copy operator.
  RiccatiMatrixFactorizer& operator=(const RiccatiMatrixFactorizer&) = default;

  // Use default move constructor.
  RiccatiMatrixFactorizer(RiccatiMatrixFactorizer&&) noexcept = default;

  // Use default move assign operator.
  RiccatiMatrixFactorizer& operator=(RiccatiMatrixFactorizer&&) noexcept = default;

  template <typename MatrixType>
  void setStateEquationDerivative(
      const Eigen::MatrixBase<MatrixType>& dsubtract_dq);

  template <typename MatrixType>
  void setStateEquationDerivativeInverse(
      const Eigen::MatrixBase<MatrixType>& dsubtract_dq_prev);

  template <typename MatrixType1, typename MatrixType2, typename MatrixType3, 
            typename MatrixType4, typename MatrixType5, typename MatrixType6, 
            typename MatrixType7, typename MatrixType8>
  void factorize_F(const double dtau, 
                   const Eigen::MatrixBase<MatrixType1>& Pqq_next,
                   const Eigen::MatrixBase<MatrixType2>& Pqv_next,
                   const Eigen::MatrixBase<MatrixType3>& Pvq_next,
                   const Eigen::MatrixBase<MatrixType4>& Pvv_next,
                   const Eigen::MatrixBase<MatrixType5>& Qqq,
                   const Eigen::MatrixBase<MatrixType6>& Qqv,
                   const Eigen::MatrixBase<MatrixType7>& Qvq,
                   const Eigen::MatrixBase<MatrixType8>& Qvv) const;

  template <typename MatrixType1, typename MatrixType2, typename MatrixType3, 
            typename MatrixType4>
  void factorize_H(const double dtau, 
                   const Eigen::MatrixBase<MatrixType1>& Pqv_next,
                   const Eigen::MatrixBase<MatrixType2>& Pvv_next,
                   const Eigen::MatrixBase<MatrixType3>& Qqa,
                   const Eigen::MatrixBase<MatrixType4>& Qva) const;

  template <typename MatrixType1, typename MatrixType2>
  void factorize_G(const double dtau, 
                   const Eigen::MatrixBase<MatrixType1>& Pvv_next,
                   const Eigen::MatrixBase<MatrixType2>& Qaa) const;

  template <typename MatrixType1, typename MatrixType2, typename VectorType1,
            typename VectorType2, typename VectorType3, typename VectorType4>
  void factorize_la(const double dtau, 
                    const Eigen::MatrixBase<MatrixType1>& Pvq_next,
                    const Eigen::MatrixBase<MatrixType2>& Pvv_next,
                    const Eigen::MatrixBase<VectorType1>& Fq,
                    const Eigen::MatrixBase<VectorType2>& Fv,
                    const Eigen::MatrixBase<VectorType3>& sv_next,
                    const Eigen::MatrixBase<VectorType4>& la) const;

  template <typename MatrixType1, typename MatrixType2>
  void correct_P(const Eigen::MatrixBase<MatrixType1>& Pqq,
                 const Eigen::MatrixBase<MatrixType2>& Pqv) const;

  template <typename VectorType>
  void correct_s(const Eigen::MatrixBase<VectorType>& sq) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  bool has_floating_base_;
  int dimv_;
  static constexpr int kDimFloatingBase = 6;
  Eigen::MatrixXd dsubtract_dq_, dsubtract_dq_prev_inv_;

};

} // namespace idocp

#include "idocp/ocp/riccati_matrix_factorizer.hxx"

#endif // IDOCP_RICCATI_MATRIX_FACTORIZER_HPP_