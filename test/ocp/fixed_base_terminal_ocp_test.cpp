#include <string>
#include <memory>

#include <gtest/gtest.h>
#include "Eigen/Core"
#include "Eigen/LU"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/terminal_ocp.hpp"
#include "idocp/cost/cost_function_data.hpp"
#include "idocp/manipulator/cost_function.hpp"
#include "idocp/manipulator/constraints.hpp"


namespace idocp {

class FixedBaseTerminalOCPTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    std::random_device rnd;
    urdf_ = "../urdf/iiwa14/iiwa14.urdf";
    robot_ = Robot(urdf_);
    data_ = CostFunctionData(robot_);
    t_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    q_ = Eigen::VectorXd::Random(robot_.dimq());
    v_ = Eigen::VectorXd::Random(robot_.dimv());
    lmd_ = Eigen::VectorXd::Random(robot_.dimv());
    gmm_ = Eigen::VectorXd::Random(robot_.dimv());
  }

  virtual void TearDown() {
  }

  double t_, dtau_;
  std::string urdf_;
  Robot robot_;
  CostFunctionData data_;
  Eigen::VectorXd q_, v_, lmd_, gmm_;
};


TEST_F(FixedBaseTerminalOCPTest, isFeasible) {
  std::shared_ptr<CostFunctionInterface> cost 
      = std::make_shared<manipulator::CostFunction>(robot_);
  std::shared_ptr<ConstraintsInterface> constraints 
      = std::make_shared<manipulator::Constraints>(robot_);
  TerminalOCP ocp(robot_, cost, constraints);
  EXPECT_TRUE(ocp.isFeasible(robot_, q_, v_));
}


TEST_F(FixedBaseTerminalOCPTest, linearizeOCP) {
  std::shared_ptr<CostFunctionInterface> cost 
      = std::make_shared<manipulator::CostFunction>(robot_);
  std::shared_ptr<ConstraintsInterface> constraints 
      = std::make_shared<manipulator::Constraints>(robot_);
  TerminalOCP ocp(robot_, cost, constraints);
  manipulator::CostFunction cost_ref(robot_);
  manipulator::Constraints constraintsref(robot_);
  robot_.generateFeasibleConfiguration(q_);
  std::random_device rnd;
  ocp.initConstraints(robot_, rnd()%50, dtau_, q_, v_);
  EXPECT_TRUE(ocp.isFeasible(robot_, q_, v_));
  const int dimq = robot_.dimq();
  const int dimv = robot_.dimv();
  Eigen::MatrixXd Qqq = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Qqv = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Qvq = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Qvv = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::VectorXd Qq = Eigen::VectorXd::Zero(dimv);
  Eigen::VectorXd Qv = Eigen::VectorXd::Zero(dimv);
  ocp.linearizeOCP(robot_, t_, lmd_, gmm_, q_, v_, 
                   Qqq, Qqv, Qvq, Qvv, Qq, Qv);
  Eigen::MatrixXd Qqq_ref = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Qqv_ref = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Qvq_ref = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Qvv_ref = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::VectorXd Qq_ref = Eigen::VectorXd::Zero(dimv);
  Eigen::VectorXd Qv_ref = Eigen::VectorXd::Zero(dimv);
  cost_ref.phiq(robot_, data_, t_, q_, v_, Qq_ref);
  cost_ref.phiv(robot_, data_, t_, q_, v_, Qv_ref);
  Qq_ref = - Qq_ref + lmd_;
  Qv_ref = - Qv_ref + gmm_;
  cost_ref.phiqq(robot_, data_, t_, q_, v_, Qqq_ref);
  cost_ref.phivv(robot_, data_, t_, q_, v_, Qvv_ref);
  EXPECT_TRUE(Qqq.isApprox(Qqq_ref));
  EXPECT_TRUE(Qqv.isApprox(Qqv_ref));
  EXPECT_TRUE(Qvq.isApprox(Qvq_ref));
  EXPECT_TRUE(Qvv.isApprox(Qvv_ref));
  EXPECT_TRUE(Qq.isApprox(Qq_ref));
  EXPECT_TRUE(Qv.isApprox(Qv_ref));
}


TEST_F(FixedBaseTerminalOCPTest, terminalCost) {
  std::shared_ptr<CostFunctionInterface> cost 
      = std::make_shared<manipulator::CostFunction>(robot_);
  std::shared_ptr<ConstraintsInterface> constraints 
      = std::make_shared<manipulator::Constraints>(robot_);
  TerminalOCP ocp(robot_, cost, constraints);
  manipulator::CostFunction cost_ref(robot_);
  manipulator::Constraints constraintsref(robot_);
  robot_.generateFeasibleConfiguration(q_);
  std::random_device rnd;
  ocp.initConstraints(robot_, rnd()%50, dtau_, q_, v_);
  EXPECT_TRUE(ocp.isFeasible(robot_, q_, v_));
  const int dimq = robot_.dimq();
  const int dimv = robot_.dimv();
  const Eigen::VectorXd dq = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd dv = Eigen::VectorXd::Random(dimv);
  double step_size = std::abs(Eigen::VectorXd::Random(1)[0]);
  while (step_size > 1) {
    step_size = std::abs(Eigen::VectorXd::Random(1)[0]);
  }
  ocp.initConstraints(robot_, rnd()%50, dtau_, q_, v_);
  EXPECT_TRUE(ocp.isFeasible(robot_, q_, v_));
  const double terminal_cost
      = ocp.terminalCost(robot_, t_, q_, v_);
  const double terminal_cost_with_step
      = ocp.terminalCost(robot_, step_size, t_, q_, v_, dq, dv);
  const double terminal_cost_ref = cost_ref.phi(robot_, data_, t_, q_, v_);
  const double terminal_cost_with_step_ref 
      = cost_ref.phi(robot_, data_, t_, q_+step_size*dq, v_+step_size*dv);
  EXPECT_DOUBLE_EQ(terminal_cost, terminal_cost_ref);
  EXPECT_DOUBLE_EQ(terminal_cost_with_step, terminal_cost_with_step_ref);
}


TEST_F(FixedBaseTerminalOCPTest, updatePrimalAndDual) {
  std::shared_ptr<CostFunctionInterface> cost 
      = std::make_shared<manipulator::CostFunction>(robot_);
  std::shared_ptr<ConstraintsInterface> constraints 
      = std::make_shared<manipulator::Constraints>(robot_);
  TerminalOCP ocp(robot_, cost, constraints);
  manipulator::CostFunction cost_ref(robot_);
  manipulator::Constraints constraintsref(robot_);
  robot_.generateFeasibleConfiguration(q_);
  std::random_device rnd;
  ocp.initConstraints(robot_, rnd()%50, dtau_, q_, v_);
  EXPECT_TRUE(ocp.isFeasible(robot_, q_, v_));
  const int dimq = robot_.dimq();
  const int dimv = robot_.dimv();
  Eigen::VectorXd q_ref = q_;
  Eigen::VectorXd v_ref = v_;
  Eigen::VectorXd lmd_ref = lmd_;
  Eigen::VectorXd gmm_ref = gmm_;
  const Eigen::MatrixXd Pqq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Pqv = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Pvq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Pvv = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::VectorXd sq = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd sv = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd dq = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd dv = Eigen::VectorXd::Random(dimv);
  double step_size = std::abs(Eigen::VectorXd::Random(1)[0]);
  while (step_size > 1) {
    step_size = std::abs(Eigen::VectorXd::Random(1)[0]);
  }
  ocp.updateDual(step_size);
  ocp.updatePrimal(robot_, step_size, Pqq, Pqv, Pvq, Pvv, sq, sv, dq, dv, 
                   lmd_, gmm_, q_, v_);
  lmd_ref += step_size * (Pqq * dq + Pqv * dv - sq);
  gmm_ref += step_size * (Pvq * dq + Pvv * dv - sv);
  q_ref += step_size * dq;
  v_ref += step_size * dv;
  EXPECT_TRUE(lmd_.isApprox(lmd_ref));
  EXPECT_TRUE(gmm_.isApprox(gmm_ref));
  EXPECT_TRUE(q_.isApprox(q_ref));
  EXPECT_TRUE(v_.isApprox(v_ref));
}


TEST_F(FixedBaseTerminalOCPTest, squaredKKTError) {
  std::shared_ptr<CostFunctionInterface> cost 
      = std::make_shared<manipulator::CostFunction>(robot_);
  std::shared_ptr<ConstraintsInterface> constraints 
      = std::make_shared<manipulator::Constraints>(robot_);
  TerminalOCP ocp(robot_, cost, constraints);
  manipulator::CostFunction cost_ref(robot_);
  manipulator::Constraints constraintsref(robot_);
  robot_.generateFeasibleConfiguration(q_);
  std::random_device rnd;
  ocp.initConstraints(robot_, rnd()%50, dtau_, q_, v_);
  EXPECT_TRUE(ocp.isFeasible(robot_, q_, v_));
  const int dimq = robot_.dimq();
  const int dimv = robot_.dimv();
  const double result 
      = ocp.squaredKKTErrorNorm(robot_, t_, lmd_, gmm_, q_, v_);
  Eigen::VectorXd lq_ref = Eigen::VectorXd::Zero(dimv);
  Eigen::VectorXd lv_ref = Eigen::VectorXd::Zero(dimv);
  cost_ref.phiq(robot_, data_, t_, q_, v_, lq_ref);
  cost_ref.phiv(robot_, data_, t_, q_, v_, lv_ref);
  lq_ref -= lmd_;
  lv_ref -= gmm_;
  const double result_ref = lq_ref.squaredNorm() + lv_ref.squaredNorm();
  EXPECT_DOUBLE_EQ(result, result_ref);
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}