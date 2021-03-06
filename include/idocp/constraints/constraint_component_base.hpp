#ifndef IDOCP_CONSTRAINT_COMPONENT_BASE_HPP_
#define IDOCP_CONSTRAINT_COMPONENT_BASE_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/constraints/constraint_component_data.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"


namespace idocp {

///
/// @typedef ConstraintComponentBase
/// @brief Base class for constraint components. 
///
class ConstraintComponentBase {
public:
  ///
  /// @brief Constructor. 
  /// @param[in] barrier Barrier parameter. Must be positive. Should be small.
  /// @param[in] fraction_to_boundary_rate Must be larger than 0 and smaller 
  /// than 1. Should be between 0.9 and 0.995.
  ///
  ConstraintComponentBase(const double barrier, 
                          const double fraction_to_boundary_rate);

  ///
  /// @brief Default constructor. 
  ///
  ConstraintComponentBase();

  ///
  /// @brief Destructor. 
  ///
  virtual ~ConstraintComponentBase() {}

  ///
  /// @brief Default copy constructor. 
  ///
  ConstraintComponentBase(const ConstraintComponentBase&) = default;

  ///
  /// @brief Default copy operator. 
  ///
  ConstraintComponentBase& operator=(const ConstraintComponentBase&) = default;

  ///
  /// @brief Default move constructor. 
  ///
  ConstraintComponentBase(ConstraintComponentBase&&) noexcept = default;

  ///
  /// @brief Default move assign operator. 
  ///
  ConstraintComponentBase& operator=(ConstraintComponentBase&&) noexcept 
      = default;

  ///
  /// @brief Check if the constraints component requres kinematics of robot 
  /// model.
  /// @return true if the constraints component requres kinematics of 
  /// Robot model. false if not.
  ///
  virtual bool useKinematics() const = 0;

  ///
  /// @brief Check whether the current solution s is feasible or not. 
  /// @param[in] robot Robot model.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] s Split solution.
  /// @return true if s is feasible. false if not.
  ///
  virtual bool isFeasible(Robot& robot, ConstraintComponentData& data, 
                          const SplitSolution& s) const = 0;

  ///
  /// @brief Set the slack and dual variables of each constraint components. 
  /// @param[in] robot Robot model.
  /// @param[out] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] dtau Time step.
  /// @param[in] s Split solution.
  ///
  virtual void setSlackAndDual(Robot& robot, ConstraintComponentData& data, 
                               const double dtau, 
                               const SplitSolution& s) const = 0;

  ///
  /// @brief Augment the dual residual of the constraints to the KKT residual 
  /// with respect to the configuration, velocity, acceleration, and contact 
  /// forces.
  /// @param[in] robot Robot model.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] dtau Time step.
  /// @param[in] s Split solution.
  /// @param[out] kkt_residual KKT residual.
  ///
  virtual void augmentDualResidual(Robot& robot, ConstraintComponentData& data,
                                   const double dtau, const SplitSolution& s,
                                   KKTResidual& kkt_residual) const = 0;

  ///
  /// @brief Augment the dual residual of the constraints to the KKT residual
  /// with respect to the control input torques.
  /// @param[in] robot Robot model.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] dtau Time step.
  /// @param[in] u Control input torques. Size must be Robot::dimv().
  /// @param[out] lu KKT residual with respect to the control input torques. 
  /// Size must be Robot::dimv().
  ///
  virtual void augmentDualResidual(const Robot& robot, 
                                   ConstraintComponentData& data,
                                   const double dtau, const Eigen::VectorXd& u,
                                   Eigen::VectorXd& lu) const = 0;

  ///
  /// @brief Consense slack and dual of the constraints and factorize condensed
  /// KKT Hessian and residual with respect to the configuration, velocity, 
  /// acceleration, and contact forces. 
  /// @param[in] robot Robot model.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData(). residual and duality are also 
  /// computed.
  /// @param[in] dtau Time step.
  /// @param[in] s Split solution.
  /// @param[out] kkt_matrix The KKT matrix. The condensed Hessians are added  
  /// to this data.
  /// @param[out] kkt_residual KKT residual. The condensed residual are added 
  /// to this data.
  ///
  virtual void condenseSlackAndDual(Robot& robot, ConstraintComponentData& data,
                                    const double dtau, const SplitSolution& s, 
                                    KKTMatrix& kkt_matrix,
                                    KKTResidual& kkt_residual) const = 0;

  ///
  /// @brief Consense slack and dual of the constraints and factorize condensed
  /// KKT Hessian and residual with respect to the configuration, velocity, 
  /// acceleration, and contact forces. 
  /// @param[in] robot Robot model.
  /// @param[in, out] data Constraints data generated by 
  /// Constraints::createConstraintsData(). residual and duality are also 
  /// computed.
  /// @param[in] dtau Time step.
  /// @param[in] u Control input torques.
  /// @param[out] Quu The KKT matrix with respect to the control input torques. 
  /// The condensed Hessians are added to this data. Size must be 
  /// Robot::dimv() x Robot::dimv().
  /// @param[out] lu KKT residual with respect to the control input torques. 
  /// The condensed residual are added to this data. Size must be Robot::dimv().
  ///
  virtual void condenseSlackAndDual(const Robot& robot, 
                                    ConstraintComponentData& data,
                                    const double dtau, const Eigen::VectorXd& u,
                                    Eigen::MatrixXd& Quu, 
                                    Eigen::VectorXd& lu) const = 0;

  ///
  /// @brief Compute directions of slack and dual.
  /// @param[in] robot Robot model.
  /// @param[in, out] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] dtau Time step.
  /// @param[in] s Split solution.
  /// @param[in] d Split direction.
  ///
  virtual void computeSlackAndDualDirection(
      Robot& robot, ConstraintComponentData& data, const double dtau, 
      const SplitSolution& s, const SplitDirection& d) const = 0;

  ///
  /// @brief Computes the primal and dual residual of the constraints. 
  /// @param[in] robot Robot model.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] dtau Time step.
  /// @param[in] s Split solution.
  ///
  virtual void computePrimalAndDualResidual(
      Robot& robot, ConstraintComponentData& data, 
      const double dtau, const SplitSolution& s) const = 0;

  ///
  /// @brief Returns the size of the constraints. 
  /// @return Size of the constraints. 
  /// 
  virtual int dimc() const = 0;

  ///
  /// @brief Returns the L1-norm of the primal residual of the constraints. 
  /// Before call this function, 
  /// ConstraintComponentBase::computePrimalResidual() must be called.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @return L1 norm of the primal residual and duality of the constraints. 
  ///
  virtual double l1NormPrimalResidual(
      const ConstraintComponentData& data) const final;

  ///
  /// @brief Returns the squared norm of the primal residual and duality of the 
  /// constraints. Before call this function, 
  /// ConstraintComponentBase::computePrimalResidual() must be called.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @return Squared norm of the, dtau, s primal residual and duality of the constraints. 
  ///
  virtual double squaredNormPrimalAndDualResidual(
      const ConstraintComponentData& data) const final;

  ///
  /// @brief Compute and returns the maximum step size by applying 
  /// fraction-to-boundary-rule to the direction of slack.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @return Maximum step size of the slack.
  ///
  virtual double maxSlackStepSize(
      const ConstraintComponentData& data) const final;

  ///
  /// @brief Compute and returns the maximum step size by applying 
  /// fraction-to-boundary-rule to the direction of dual.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @return Maximum step size of the dual.
  ///
  virtual double maxDualStepSize(
      const ConstraintComponentData& data) const final;

  ///
  /// @brief Updates the slack with step_size.
  /// @param[in, out] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] step_size Step size. 
  ///
  virtual void updateSlack(ConstraintComponentData& data, 
                           const double step_size) const final;

  ///
  /// @brief Updates the dual with step_size.
  /// @param[in, out] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] step_size Step size. 
  ///
  virtual void updateDual(ConstraintComponentData& data, 
                          const double step_size) const final;

  ///
  /// @brief Computes and returns the value of the barrier function for slack 
  /// variables.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @return Value of the barrier function. 
  ///
  virtual double costSlackBarrier(
      const ConstraintComponentData& data) const final;

  ///
  /// @brief Computes and returns the value of the barrier function for slack 
  /// variables with the step_size.
  /// @param[in] data Constraints data generated by 
  /// Constraints::createConstraintsData().
  /// @param[in] step_size Step size. 
  /// @return Value of the barrier function. 
  ///
  virtual double costSlackBarrier(const ConstraintComponentData& data, 
                                  const double step_size) const final;

  ///
  /// @brief Set the barrier parameter.
  /// @param[in] barrier Barrier parameter. Must be positive. Should be small.
  ///
  virtual void setBarrier(const double barrier) final;

  ///
  /// @brief Set the fraction to boundary rate.
  /// @param[in] fraction_to_boundary_rate Must be larger than 0 and smaller 
  /// than 1. Should be between 0.9 and 0.995.
  ///
  virtual void setFractionToBoundaryRate(
      const double fraction_to_boundary_rate) final;

protected:
  ///
  /// @brief Set the slack and dual variables positive.
  ///
  virtual void setSlackAndDualPositive(
      ConstraintComponentData& data) const final;

  ///
  /// @brief Computes the duality residual between the slack and dual variables.
  ///
  virtual void computeDuality(ConstraintComponentData& data) const final;

  ///
  /// @brief Computes the direction of the dual variable from slack, residual,
  /// duality, and the direction of the slack.
  ///
  virtual void computeDualDirection(ConstraintComponentData& data) const final;

private:
  double barrier_, fraction_to_boundary_rate_;

};

} // namespace idocp

#include "idocp/constraints/constraint_component_base.hxx"

#endif // IDOCP_CONSTRAINT_COMPONENT_BASE_HPP_