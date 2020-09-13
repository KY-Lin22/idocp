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
#include "idocp/complementarity/contact_force_inequality.hpp"
#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {

class FixedBaseContactForceInequalityTest : public ::testing::Test {
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
    contact_force_inequality_ = ContactForceInequality(robot_, mu_, barrier_, fraction_to_boundary_rate_);
  }

  virtual void TearDown() {
  }

  double mu_, barrier_, dtau_, fraction_to_boundary_rate_;
  Eigen::VectorXd slack_, dual_, dslack_, ddual_;
  Robot robot_;
  ContactForceInequality contact_force_inequality_;
};


TEST_F(FixedBaseContactForceInequalityTest, isFeasible) {
  SplitSolution s(robot_);
  s.f_verbose = - Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  assert(s.f.size() == 3);
  assert(s.f_verbose.size() == 7);
  assert(robot_.num_point_contacts() == 1);
  s.set_f();
  EXPECT_FALSE(contact_force_inequality_.isFeasible(robot_, s));
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 0.9 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  EXPECT_FALSE(contact_force_inequality_.isFeasible(robot_, s));
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  EXPECT_TRUE(contact_force_inequality_.isFeasible(robot_, s));
}


TEST_F(FixedBaseContactForceInequalityTest, computePrimalResidual) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  ASSERT_TRUE(contact_force_inequality_.isFeasible(robot_, s));
  ConstraintComponentData data(6);
  data.slack = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  contact_force_inequality_.computePrimalResidual(robot_, dtau_, s, data);
  Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(6));
  residual_ref.head(5) = - dtau_ * s.f_verbose.head(5);
  residual_ref(5) = - dtau_ * (mu_*mu_*s.f(2)*s.f(2) - s.f(0)*s.f(0) - s.f(1)*s.f(1));
  residual_ref += data.slack;
  EXPECT_TRUE(data.residual.isApprox(residual_ref));
  contact_force_inequality_.setSlack(robot_, dtau_, s, data);
  contact_force_inequality_.computePrimalResidual(robot_, dtau_, s, data);
  EXPECT_TRUE(data.residual.isZero());
}


TEST_F(FixedBaseContactForceInequalityTest, augmentDualResidual) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  ASSERT_TRUE(contact_force_inequality_.isFeasible(robot_, s));
  ConstraintComponentData data(6);
  data.dual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  KKTResidual kkt_residual(robot_);
  contact_force_inequality_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
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


TEST_F(FixedBaseContactForceInequalityTest, augmentCondensedHessian) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  ASSERT_TRUE(contact_force_inequality_.isFeasible(robot_, s));
  ConstraintComponentData data(6);
  data.dual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  KKTMatrix kkt_matrix(robot_);
  const Eigen::VectorXd diagonal = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  contact_force_inequality_.augmentCondensedHessian(robot_, dtau_, s, diagonal, kkt_matrix);
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
  Eigen::MatrixXd Qff_ref(Eigen::MatrixXd::Zero(7, 7));
  Qff_ref = g_f.transpose() * diagonal.asDiagonal() * g_f;
  EXPECT_TRUE(kkt_matrix.Qff().isApprox(Qff_ref));
}


TEST_F(FixedBaseContactForceInequalityTest, augmentCondensedResidual) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  ASSERT_TRUE(contact_force_inequality_.isFeasible(robot_, s));
  ConstraintComponentData data(6);
  data.dual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  KKTResidual kkt_residual(robot_);
  const Eigen::VectorXd condensed_residual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  contact_force_inequality_.augmentCondensedResidual(robot_, dtau_, s, condensed_residual, kkt_residual);
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
  Eigen::MatrixXd lf_ref(Eigen::VectorXd::Zero(7));
  lf_ref = g_f.transpose() * condensed_residual;
  EXPECT_TRUE(kkt_residual.lf().isApprox(lf_ref));
}


TEST_F(FixedBaseContactForceInequalityTest, computeSlackDirection) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
  s.set_f();
  ASSERT_TRUE(contact_force_inequality_.isFeasible(robot_, s));
  ConstraintComponentData data(6);
  data.residual = Eigen::VectorXd::Random(6);
  SplitDirection d(robot_);
  d.df() = Eigen::VectorXd::Random(7*robot_.num_point_contacts());
  contact_force_inequality_.computeSlackDirection(robot_, dtau_, s, d, data);
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
  Eigen::VectorXd dslack_ref(Eigen::VectorXd::Zero(7));
  dslack_ref = g_f * d.df() - data.residual;
  EXPECT_TRUE(data.dslack.isApprox(dslack_ref));
  std::cout << data.dslack.transpose() << std::endl;
  std::cout << dslack_ref.transpose() << std::endl;
}


} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}