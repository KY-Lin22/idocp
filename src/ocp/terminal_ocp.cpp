#include "idocp/ocp/terminal_ocp.hpp"

#include <assert.h>


namespace idocp {

TerminalOCP::TerminalOCP(
    const Robot& robot, const std::shared_ptr<CostFunctionInterface>& cost, 
    const std::shared_ptr<ConstraintsInterface>& constraints)
  : cost_(cost),
    constraints_(constraints),
    cost_data_(robot),
    joint_constraints_(robot),
    dimq_(robot.dimq()),
    dimv_(robot.dimv()),
    lq_(Eigen::VectorXd::Zero(robot.dimv())),
    lv_(Eigen::VectorXd::Zero(robot.dimv())),
    q_res_(Eigen::VectorXd::Zero(robot.dimv())),
    v_res_(Eigen::VectorXd::Zero(robot.dimv())),
    // The following variables are only needed for line search
    q_tmp_(Eigen::VectorXd::Zero(robot.dimq())), 
    v_tmp_(Eigen::VectorXd::Zero(robot.dimv())) {
}


TerminalOCP::TerminalOCP() 
  : cost_(),
    constraints_(),
    joint_constraints_(),
    dimq_(0),
    dimv_(0),
    lq_(),
    lv_(),
    q_res_(),
    v_res_(),
    q_tmp_(), 
    v_tmp_() {
}


TerminalOCP::~TerminalOCP() {
}


bool TerminalOCP::isFeasible(const Robot& robot, const Eigen::VectorXd& q, 
                             const Eigen::VectorXd& v) {
  assert(q.size() == dimq_);
  assert(v.size() == dimv_);
  // TODO: add inequality constraints at the terminal OCP.
  // return joint_constraints_.isFeasible(robot, q, v, a, u);
  return true;
}


void TerminalOCP::initConstraints(const Robot& robot, const int time_step, 
                                  const double dtau, const Eigen::VectorXd& q, 
                                  const Eigen::VectorXd& v) {
  assert(time_step >= 0);
  assert(dtau > 0);
  assert(q.size() == dimq_);
  assert(v.size() == dimv_);
  // TODO: add inequality constraints at the terminal OCP.
  // joint_constraints_.setTimeStep(time_step);
  // joint_constraints_.setSlackAndDual(robot, dtau, q, v, a, u);
}


void TerminalOCP::linearizeOCP(Robot& robot, const double t, 
                               const Eigen::VectorXd& lmd, 
                               const Eigen::VectorXd& gmm, 
                               const Eigen::VectorXd& q, 
                               const Eigen::VectorXd& v, Eigen::MatrixXd& Qqq, 
                               Eigen::MatrixXd& Qqv, Eigen::MatrixXd& Qvq, 
                               Eigen::MatrixXd& Qvv, Eigen::VectorXd& Qq, 
                               Eigen::VectorXd& Qv) {
  assert(q.size() == dimq_);
  assert(v.size() == dimv_);
  assert(Qqq.rows() == dimv_);
  assert(Qqq.cols() == dimv_);
  assert(Qqv.rows() == dimv_);
  assert(Qqv.cols() == dimv_);
  assert(Qvq.rows() == dimv_);
  assert(Qvq.cols() == dimv_);
  assert(Qvv.rows() == dimv_);
  assert(Qvv.cols() == dimv_);
  assert(Qq.size() == dimv_);
  assert(Qv.size() == dimv_);
  if (robot.has_floating_base()) {
    robot.computeConfigurationJacobian(q);
  }
  cost_->phiq(robot, cost_data_, t, q, v, lq_);
  cost_->phiv(robot, cost_data_, t, q, v, lv_);
  Qq = - lq_ + lmd;
  Qv = - lv_ + gmm;
  cost_->phiqq(robot, cost_data_, t, q, v, Qqq);
  cost_->phivv(robot, cost_data_, t, q, v, Qvv);
}


void TerminalOCP::computeCondensedDirection(Robot& robot, const double dtau, 
                                            const Eigen::VectorXd& dq, 
                                            const Eigen::VectorXd& dv) {
  // TODO: add inequality constraints at the terminal OCP.
  // joint_constraints_.computeSlackAndDualDirection(robot, dtau, dq, dv);
}

 
double TerminalOCP::maxPrimalStepSize() {
  return 1;
  // TODO: add inequality constraints at the terminal OCP.
  // return joint_constraints_.maxSlackStepSize();
}


double TerminalOCP::maxDualStepSize() {
  return 1;
  // TODO: add inequality constraints at the terminal OCP.
  // return joint_constraints_.maxDualStepSize();
}


double TerminalOCP::terminalCost(Robot& robot, const double t, 
                                 const Eigen::VectorXd& q, 
                                 const Eigen::VectorXd& v) {
  assert(q.size() == dimq_);
  assert(v.size() == dimv_);
  return cost_->phi(robot, cost_data_, t, q, v);
}


double TerminalOCP::terminalCost(Robot& robot, const double step_size, 
                                 const double t, const Eigen::VectorXd& q, 
                                 const Eigen::VectorXd& v, 
                                 const Eigen::VectorXd& dq, 
                                 const Eigen::VectorXd& dv) {
  assert(step_size > 0);
  assert(step_size <= 1);
  assert(q.size() == dimq_);
  assert(v.size() == dimv_);
  assert(dq.size() == dimv_);
  assert(dv.size() == dimv_);
  if (robot.has_floating_base()) {
    q_tmp_ = q;
    robot.integrateConfiguration(dq, step_size, q_tmp_);
  }
  else {
    q_tmp_ = q + step_size * dq;
  }
  v_tmp_ = v + step_size * dv;
  return cost_->phi(robot, cost_data_, t, q_tmp_, v_tmp_);
}


void TerminalOCP::updateDual(const double step_size) {
  assert(step_size > 0);
  assert(step_size <= 1);
  // TODO: add inequality constraints at the terminal OCP.
  // joint_constraints_.updateDual(step_size);
}


void TerminalOCP::updatePrimal(Robot& robot, const double step_size, 
                               const Eigen::MatrixXd& Pqq, 
                               const Eigen::MatrixXd& Pqv, 
                               const Eigen::MatrixXd& Pvq, 
                               const Eigen::MatrixXd& Pvv, 
                               const Eigen::VectorXd& sq, 
                               const Eigen::VectorXd& sv, 
                               const Eigen::VectorXd& dq, 
                               const Eigen::VectorXd& dv, Eigen::VectorXd& lmd, 
                               Eigen::VectorXd& gmm, Eigen::VectorXd& q, 
                               Eigen::VectorXd& v) const {
  assert(step_size > 0);
  assert(step_size <= 1);
  assert(Pqq.rows() == dimv_);
  assert(Pqq.cols() == dimv_);
  assert(Pqv.rows() == dimv_);
  assert(Pqv.cols() == dimv_);
  assert(Pvq.rows() == dimv_);
  assert(Pvq.cols() == dimv_);
  assert(Pvv.rows() == dimv_);
  assert(Pvv.cols() == dimv_);
  assert(sq.size() == dimv_);
  assert(sv.size() == dimv_);
  assert(dq.size() == dimv_);
  assert(dv.size() == dimv_);
  assert(lmd.rows() == dimv_);
  assert(gmm.rows() == dimv_);
  assert(q.rows() == dimq_);
  assert(v.rows() == dimv_);
  lmd.noalias() += step_size * (Pqq * dq + Pqv * dv - sq);
  gmm.noalias() += step_size * (Pvq * dq + Pvv * dv - sv);
  robot.integrateConfiguration(dq, step_size, q);
  v.noalias() += step_size * dv;
}


double TerminalOCP::squaredKKTErrorNorm(Robot& robot, const double t, 
                                        const Eigen::VectorXd& lmd, 
                                        const Eigen::VectorXd& gmm, 
                                        const Eigen::VectorXd& q, 
                                        const Eigen::VectorXd& v) {
  assert(lmd.size() == dimv_);
  assert(gmm.size() == dimv_);
  assert(q.size() == dimq_);
  assert(v.size() == dimv_);
  // Compute the partial derivatives of the Lagrangian with respect to the 
  // terminal configuration and velocity.
  if (robot.has_floating_base()) {
    robot.computeConfigurationJacobian(q);
  }
  cost_->phiq(robot, cost_data_, t, q, v, lq_);
  cost_->phiv(robot, cost_data_, t, q, v, lv_);
  cost_->phiq(robot, cost_data_, t, q, v, lq_);
  cost_->phiv(robot, cost_data_, t, q, v, lv_);
  lq_.noalias() -= lmd;
  lv_.noalias() -= gmm;
  double error = 0;
  error += lq_.squaredNorm();
  error += lv_.squaredNorm();
  return error;
}

} // namespace idocp