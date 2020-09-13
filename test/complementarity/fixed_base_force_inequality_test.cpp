#include <string>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/split_direction.hpp"
#include "idocp/ocp/kkt_matrix.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/complementarity/force_inequality.hpp"
#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {

class FixedBaseForceInequalityTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    mu_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    barrier_ = 1.0e-04;
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    fraction_to_boundary_rate_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    const std::vector<int> contact_frames = {18};
    const std::string urdf = "../urdf/iiwa14/iiwa14.urdf";
    robot_ = Robot(urdf, contact_frames, 0, 0);
    force_inequality_ = ForceInequality(robot_, mu_, barrier_, fraction_to_boundary_rate_);
  }

  virtual void TearDown() {
  }

  double mu_, barrier_, dtau_, fraction_to_boundary_rate_;
  Eigen::VectorXd slack_, dual_, dslack_, ddual_;
  Robot robot_;
  ForceInequality force_inequality_;
};


TEST_F(FixedBaseForceInequalityTest, isFeasible) {
  SplitSolution s(robot_);
  s.f_verbose = - Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  assert(s.f.size() == 3);
  assert(s.f_verbose.size() == 7);
  assert(robot_.num_point_contacts() == 1);
  s.set_f();
  EXPECT_FALSE(force_inequality_.isFeasible(robot_, s));
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 0.9 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  EXPECT_FALSE(force_inequality_.isFeasible(robot_, s));
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  EXPECT_TRUE(force_inequality_.isFeasible(robot_, s));
}


TEST_F(FixedBaseForceInequalityTest, computePrimalResidual) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  ASSERT_TRUE(force_inequality_.isFeasible(robot_, s));
  ConstraintComponentData data(6);
  data.slack = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  force_inequality_.computePrimalResidual(robot_, dtau_, s, data);
  Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(6));
  residual_ref.head(5) = - dtau_ * s.f_verbose.head(5);
  residual_ref(5) = - dtau_ * (mu_*mu_*s.f(2)*s.f(2) - s.f(0)*s.f(0) - s.f(1)*s.f(1));
  residual_ref += data.slack;
  EXPECT_TRUE(data.residual.isApprox(residual_ref));
  force_inequality_.setSlackAndDual(robot_, dtau_, s, data);
  force_inequality_.computePrimalResidual(robot_, dtau_, s, data);
  EXPECT_TRUE(data.residual.isZero());
}


TEST_F(FixedBaseForceInequalityTest, augmentDualResidual) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  ASSERT_TRUE(force_inequality_.isFeasible(robot_, s));
  ConstraintComponentData data(6);
  data.dual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  KKTResidual kkt_residual(robot_);
  force_inequality_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6, 7));
  g_f(0, 0) = dtau_;
  g_f(1, 1) = dtau_;
  g_f(2, 2) = dtau_;
  g_f(3, 3) = dtau_;
  g_f(4, 4) = dtau_;
  g_f(5, 0) = - 2 * dtau_ * s.f(0);
  g_f(5, 1) = 2 * dtau_ * s.f(0);
  g_f(5, 2) = - 2 * dtau_ * s.f(1);
  g_f(5, 3) = 2 * dtau_ * s.f(1);
  g_f(5, 4) = 2 * mu_ * mu_ * dtau_ * s.f(2);
  std::cout << g_f << std::endl;
  Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(7));
  residual_ref = - g_f.transpose() * data.dual;
  EXPECT_TRUE(kkt_residual.lf().isApprox(residual_ref));
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}