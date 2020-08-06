#include <string>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/constraints/pdipm_func.hpp"
#include "idocp/constraints/joint_space_constraints/joint_variables_upper_limits.hpp"


namespace idocp {
namespace pdipm {

class JointPositionUpperLimitsTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    fixed_base_urdf_ = "../../urdf/iiwa14/iiwa14.urdf";
    floating_base_urdf_ = "../../urdf/anymal/anymal.urdf";
    fixed_base_robot_ = Robot(fixed_base_urdf_);
    floating_base_robot_ = Robot(floating_base_urdf_);
    barrier_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    while (barrier_ == 0) {
      barrier_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    }
  }

  virtual void TearDown() {
  }

  double barrier_, dtau_;
  Eigen::VectorXd slack_, dual_, dslack_, ddual_;
  std::string fixed_base_urdf_, floating_base_urdf_;
  Robot fixed_base_robot_, floating_base_robot_;
};


TEST_F(JointPositionUpperLimitsTest, isFeasibleFixedBase) {
  JointVariablesUpperLimits limit(fixed_base_robot_, 
                                  fixed_base_robot_.upperJointPositionLimit(), 
                                  barrier_);
  Eigen::VectorXd q(fixed_base_robot_.dimq());
  fixed_base_robot_.generateFeasibleConfiguration(q);
  EXPECT_TRUE(limit.isFeasible(q));
  q = 2*fixed_base_robot_.upperJointPositionLimit();
  EXPECT_FALSE(limit.isFeasible(q));
}


TEST_F(JointPositionUpperLimitsTest, isFeasibleFloatingBase) {
  JointVariablesUpperLimits limit(floating_base_robot_, 
                                  floating_base_robot_.upperJointPositionLimit(), 
                                  barrier_);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(floating_base_robot_.dimq());
  floating_base_robot_.generateFeasibleConfiguration(q);
  EXPECT_TRUE(limit.isFeasible(q));
  const int dimc = floating_base_robot_.upperJointPositionLimit().size();
  q.tail(dimc) = 2*floating_base_robot_.upperJointPositionLimit();
  ASSERT_EQ(q.size(), floating_base_robot_.dimq());
  EXPECT_FALSE(limit.isFeasible(q));
}


TEST_F(JointPositionUpperLimitsTest, setSlackAndDualFixedBase) {
  JointVariablesUpperLimits limit(fixed_base_robot_, 
                                  fixed_base_robot_.upperJointPositionLimit(), 
                                  barrier_);
  const int dimq = fixed_base_robot_.dimq();
  Eigen::VectorXd q(dimq);
  Eigen::VectorXd qmax = fixed_base_robot_.upperJointPositionLimit();
  ASSERT_EQ(dimq, fixed_base_robot_.upperJointPositionLimit().size());
  fixed_base_robot_.generateFeasibleConfiguration(q);
  limit.setSlackAndDual(dtau_, q);
  Eigen::VectorXd Cq = Eigen::VectorXd::Zero(dimq);
  limit.augmentDualResidual(dtau_, Cq);
  Eigen::VectorXd slack_ref = Eigen::VectorXd::Zero(dimq);
  Eigen::VectorXd dual_ref = Eigen::VectorXd::Zero(dimq);
  slack_ref = dtau_ * (qmax-q);
  Eigen::VectorXd Cq_ref = Eigen::VectorXd::Zero(dimq);
  pdipmfunc::SetSlackAndDualPositive(dimq, barrier_, slack_ref, dual_ref);
  Cq_ref = dtau_ * dual_ref;
  EXPECT_TRUE(Cq.isApprox(Cq_ref));
  const double cost_slack_barrier = limit.costSlackBarrier();
  const double cost_slack_barrier_ref 
      = pdipmfunc::SlackBarrierCost(dimq, barrier_, slack_ref);
  EXPECT_DOUBLE_EQ(cost_slack_barrier, cost_slack_barrier_ref);
  const double l1residual = limit.residualL1Nrom(dtau_, q);
  const double l1residual_ref = (dtau_*(q-qmax)+slack_ref).lpNorm<1>();
  EXPECT_DOUBLE_EQ(l1residual, l1residual_ref);
  const double l2residual = limit.residualSquaredNrom(dtau_, q);
  Eigen::VectorXd duality_ref = Eigen::VectorXd::Zero(dimq);
  pdipmfunc::ComputeDualityResidual(barrier_, slack_ref, dual_ref, duality_ref);
  const double l2residual_ref 
      = (dtau_*(q-qmax)+slack_ref).squaredNorm() + duality_ref.squaredNorm();
  EXPECT_DOUBLE_EQ(l2residual, l2residual_ref);
}


TEST_F(JointPositionUpperLimitsTest, setSlackAndDualFloatingBase) {
  JointVariablesUpperLimits limit(floating_base_robot_, 
                                  floating_base_robot_.upperJointPositionLimit(), 
                                  barrier_);
  const int dimq = floating_base_robot_.dimq();
  const int dimv = floating_base_robot_.dimv();
  Eigen::VectorXd q(dimq);
  Eigen::VectorXd qmax = floating_base_robot_.upperJointPositionLimit();
  const int dimc = floating_base_robot_.upperJointPositionLimit().size();
  ASSERT_EQ(dimc+6, dimv);
  floating_base_robot_.generateFeasibleConfiguration(q);
  limit.setSlackAndDual(dtau_, q);
  Eigen::VectorXd Cq = Eigen::VectorXd::Zero(dimv);
  limit.augmentDualResidual(dtau_, Cq);
  Eigen::VectorXd slack_ref = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd dual_ref = Eigen::VectorXd::Zero(dimc);
  slack_ref = dtau_ * (qmax-q.tail(dimc));
  Eigen::VectorXd Cq_ref = Eigen::VectorXd::Zero(dimv);
  pdipmfunc::SetSlackAndDualPositive(dimc, barrier_, slack_ref, dual_ref);
  Cq_ref.tail(dimc) = dtau_ * dual_ref;
  EXPECT_TRUE(Cq.isApprox(Cq_ref));
  const double cost_slack_barrier = limit.costSlackBarrier();
  const double cost_slack_barrier_ref 
      = pdipmfunc::SlackBarrierCost(dimc, barrier_, slack_ref);
  EXPECT_DOUBLE_EQ(cost_slack_barrier, cost_slack_barrier_ref);
  const double l1residual = limit.residualL1Nrom(dtau_, q);
  const double l1residual_ref = (dtau_*(q.tail(dimc)-qmax)+slack_ref).lpNorm<1>();
  EXPECT_DOUBLE_EQ(l1residual, l1residual_ref);
  const double l2residual = limit.residualSquaredNrom(dtau_, q);
  Eigen::VectorXd duality_ref = Eigen::VectorXd::Zero(dimc);
  pdipmfunc::ComputeDualityResidual(barrier_, slack_ref, dual_ref, duality_ref);
  const double l2residual_ref 
      = (dtau_*(q.tail(dimc)-qmax)+slack_ref).squaredNorm() + duality_ref.squaredNorm();
  EXPECT_DOUBLE_EQ(l2residual, l2residual_ref);
}


TEST_F(JointPositionUpperLimitsTest, condenseSlackAndDualFixedBase) {
  JointVariablesUpperLimits limit(fixed_base_robot_, 
                                  fixed_base_robot_.upperJointPositionLimit(), 
                                  barrier_);
  const int dimq = fixed_base_robot_.dimq();
  Eigen::VectorXd q(dimq);
  Eigen::VectorXd qmax = fixed_base_robot_.upperJointPositionLimit();
  ASSERT_EQ(dimq, fixed_base_robot_.upperJointPositionLimit().size());
  fixed_base_robot_.generateFeasibleConfiguration(q);
  limit.setSlackAndDual(dtau_, q);
  Eigen::VectorXd Cq = Eigen::VectorXd::Zero(dimq);
  limit.augmentDualResidual(dtau_, Cq);
  Eigen::VectorXd slack_ref = Eigen::VectorXd::Zero(dimq);
  Eigen::VectorXd dual_ref = Eigen::VectorXd::Zero(dimq);
  slack_ref = dtau_ * (qmax-q);
  Eigen::VectorXd Cq_ref = Eigen::VectorXd::Zero(dimq);
  pdipmfunc::SetSlackAndDualPositive(dimq, barrier_, slack_ref, dual_ref);
  Cq_ref = dtau_ * dual_ref;
  EXPECT_TRUE(Cq.isApprox(Cq_ref));
  double cost_slack_barrier = limit.costSlackBarrier();
  double cost_slack_barrier_ref 
      = pdipmfunc::SlackBarrierCost(dimq, barrier_, slack_ref);
  EXPECT_DOUBLE_EQ(cost_slack_barrier, cost_slack_barrier_ref);
  Cq = Eigen::VectorXd::Ones(dimq);
  Eigen::MatrixXd Cqq = Eigen::MatrixXd::Ones(dimq, dimq);
  limit.condenseSlackAndDual(dtau_, q, Cqq, Cq);
  Cq_ref = Eigen::VectorXd::Ones(dimq);
  Eigen::MatrixXd Cqq_ref = Eigen::MatrixXd::Ones(dimq, dimq);
  for (int i=0; i<dimq; ++i) {
    Cqq_ref(i, i) += dtau_ * dtau_ * dual_ref.coeff(i) / slack_ref.coeff(i);
  }
  Eigen::VectorXd residual_ref = dtau_ * (q-qmax) + slack_ref;
  Eigen::VectorXd duality_ref = Eigen::VectorXd::Zero(dimq);
  pdipmfunc::ComputeDualityResidual(barrier_, slack_ref, dual_ref, duality_ref);
  Cq_ref.array() 
      += dtau_ * (dual_ref.array()*residual_ref.array()-duality_ref.array()) 
               / slack_ref.array();
  EXPECT_TRUE(Cq.isApprox(Cq_ref));
  EXPECT_TRUE(Cqq.isApprox(Cqq_ref));
  std::cout << "Cqq " << std::endl;
  std::cout << Cqq << "\n" << std::endl;
  std::cout << "Cqq_ref " << std::endl;
  std::cout << Cqq_ref << "\n" << std::endl;
  const Eigen::VectorXd dq = Eigen::VectorXd::Random(dimq);
  limit.computeSlackAndDualDirection(dtau_, dq);
  const Eigen::VectorXd dslack_ref = - dtau_ * dq - residual_ref;
  Eigen::VectorXd ddual_ref = Eigen::VectorXd::Zero(dimq);
  pdipmfunc::ComputeDualDirection(dual_ref, slack_ref, dslack_ref, duality_ref, 
                                  ddual_ref);
  const double margin_rate = 0.995;
  const double slack_step_size = limit.maxSlackStepSize(margin_rate);
  const double dual_step_size = limit.maxDualStepSize(margin_rate);
  const double slack_step_size_ref 
      = pdipmfunc::FractionToBoundary(dimq, margin_rate, slack_ref, dslack_ref);
  const double dual_step_size_ref 
      = pdipmfunc::FractionToBoundary(dimq, margin_rate, dual_ref, ddual_ref);
  EXPECT_DOUBLE_EQ(slack_step_size, slack_step_size_ref);
  EXPECT_DOUBLE_EQ(dual_step_size, dual_step_size_ref);
  const double step_size = std::min(slack_step_size, dual_step_size); 
  const double berrier = limit.costSlackBarrier(step_size);
  const double berrier_ref 
      = pdipmfunc::SlackBarrierCost(dimq, barrier_, 
                                    slack_ref+step_size*dslack_ref);
  EXPECT_DOUBLE_EQ(berrier, berrier_ref);
  limit.updateSlack(step_size);
  limit.updateDual(step_size);
  cost_slack_barrier = limit.costSlackBarrier();
  slack_ref += step_size * dslack_ref;
  dual_ref += step_size * ddual_ref;
  cost_slack_barrier_ref 
      = pdipmfunc::SlackBarrierCost(dimq, barrier_, slack_ref);
  EXPECT_DOUBLE_EQ(cost_slack_barrier, cost_slack_barrier_ref);
  limit.augmentDualResidual(dtau_, Cq);
  Cq_ref += dtau_ * dual_ref;
  EXPECT_TRUE(Cq.isApprox(Cq_ref));
}


TEST_F(JointPositionUpperLimitsTest, condenseSlackAndDualFloatingBase) {
  JointVariablesUpperLimits limit(floating_base_robot_, 
                                  floating_base_robot_.upperJointPositionLimit(), 
                                  barrier_);
  const int dimq = floating_base_robot_.dimq();
  const int dimv = floating_base_robot_.dimv();
  const int dim_passive = floating_base_robot_.dim_passive();
  Eigen::VectorXd q(dimq);
  Eigen::VectorXd qmax = floating_base_robot_.upperJointPositionLimit();
  const int dimc = floating_base_robot_.upperJointPositionLimit().size();
  ASSERT_EQ(dimc+6, dimv);
  floating_base_robot_.generateFeasibleConfiguration(q);
  limit.setSlackAndDual(dtau_, q);
  Eigen::VectorXd Cq = Eigen::VectorXd::Zero(dimv);
  limit.augmentDualResidual(dtau_, Cq);
  Eigen::VectorXd slack_ref = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd dual_ref = Eigen::VectorXd::Zero(dimc);
  slack_ref = dtau_ * (qmax-q.tail(dimc));
  Eigen::VectorXd Cq_ref = Eigen::VectorXd::Zero(dimv);
  pdipmfunc::SetSlackAndDualPositive(dimc, barrier_, slack_ref, dual_ref);
  Cq_ref.tail(dimc) = dtau_ * dual_ref;
  EXPECT_TRUE(Cq.isApprox(Cq_ref));
  double cost_slack_barrier = limit.costSlackBarrier();
  double cost_slack_barrier_ref 
      = pdipmfunc::SlackBarrierCost(dimc, barrier_, slack_ref);
  EXPECT_DOUBLE_EQ(cost_slack_barrier, cost_slack_barrier_ref);
  Cq = Eigen::VectorXd::Ones(dimv);
  Eigen::MatrixXd Cqq = Eigen::MatrixXd::Ones(dimv, dimv);
  limit.condenseSlackAndDual(dtau_, q, Cqq, Cq);
  Cq_ref = Eigen::VectorXd::Ones(dimv);
  Eigen::MatrixXd Cqq_ref = Eigen::MatrixXd::Ones(dimv, dimv);
  for (int i=0; i<dimc; ++i) {
    Cqq_ref(dim_passive+i, dim_passive+i) 
        += dtau_ * dtau_ * dual_ref.coeff(i) / slack_ref.coeff(i);
  }
  Eigen::VectorXd residual_ref = dtau_ * (q.tail(dimc)-qmax) + slack_ref;
  Eigen::VectorXd duality_ref = Eigen::VectorXd::Zero(dimc);
  pdipmfunc::ComputeDualityResidual(barrier_, slack_ref, dual_ref, duality_ref);
  Cq_ref.tail(dimc).array() 
      += dtau_ * (dual_ref.array()*residual_ref.array()-duality_ref.array()) 
               / slack_ref.array();
  EXPECT_TRUE(Cq.isApprox(Cq_ref));
  EXPECT_TRUE(Cqq.isApprox(Cqq_ref));
  std::cout << "Cqq " << std::endl;
  std::cout << Cqq << "\n" << std::endl;
  std::cout << "Cqq_ref " << std::endl;
  std::cout << Cqq_ref << "\n" << std::endl;
  const Eigen::VectorXd dq = Eigen::VectorXd::Random(dimv);
  limit.computeSlackAndDualDirection(dtau_, dq);
  const Eigen::VectorXd dslack_ref = - dtau_ * dq.tail(dimc) - residual_ref;
  Eigen::VectorXd ddual_ref = Eigen::VectorXd::Zero(dimc);
  pdipmfunc::ComputeDualDirection(dual_ref, slack_ref, dslack_ref, duality_ref, 
                                  ddual_ref);
  const double margin_rate = 0.995;
  const double slack_step_size = limit.maxSlackStepSize(margin_rate);
  const double dual_step_size = limit.maxDualStepSize(margin_rate);
  const double slack_step_size_ref 
      = pdipmfunc::FractionToBoundary(dimc, margin_rate, slack_ref, dslack_ref);
  const double dual_step_size_ref 
      = pdipmfunc::FractionToBoundary(dimc, margin_rate, dual_ref, ddual_ref);
  EXPECT_DOUBLE_EQ(slack_step_size, slack_step_size_ref);
  EXPECT_DOUBLE_EQ(dual_step_size, dual_step_size_ref);
  const double step_size = std::min(slack_step_size, dual_step_size); 
  const double berrier = limit.costSlackBarrier(step_size);
  const double berrier_ref 
      = pdipmfunc::SlackBarrierCost(dimc, barrier_, 
                                    slack_ref+step_size*dslack_ref);
  EXPECT_DOUBLE_EQ(berrier, berrier_ref);
  limit.updateSlack(step_size);
  limit.updateDual(step_size);
  cost_slack_barrier = limit.costSlackBarrier();
  slack_ref += step_size * dslack_ref;
  dual_ref += step_size * ddual_ref;
  cost_slack_barrier_ref 
      = pdipmfunc::SlackBarrierCost(dimc, barrier_, slack_ref);
  EXPECT_DOUBLE_EQ(cost_slack_barrier, cost_slack_barrier_ref);
  limit.augmentDualResidual(dtau_, Cq);
  Cq_ref.tail(dimc) += dtau_ * dual_ref;
  EXPECT_TRUE(Cq.isApprox(Cq_ref));
}

} // namespace pdipm
} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}