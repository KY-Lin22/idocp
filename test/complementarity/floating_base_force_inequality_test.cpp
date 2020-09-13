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

class FloatingBaseForceInequalityTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    mu_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    barrier_ = 1.0e-04;
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    fraction_to_boundary_rate_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    const std::vector<int> contact_frames = {14, 24, 34, 44};
    const std::string urdf = "../urdf/anymal/anymal.urdf";
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


TEST_F(FloatingBaseForceInequalityTest, isFeasible) {
  SplitSolution s(robot_);
  s.f_verbose = - Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  assert(s.f.size() == 3*robot_.num_point_contacts());
  assert(s.f_verbose.size() == 7*robot_.num_point_contacts());
  assert(robot_.num_point_contacts() == 4);
  s.set_f();
  EXPECT_FALSE(force_inequality_.isFeasible(robot_, s));
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f_verbose(7*i+4) = 0.9 * std::sqrt((s.f(3*i)*s.f(3*i)+s.f(3*i+1)*s.f(3*i+1))/(mu_*mu_));
  }
  s.set_f();
  EXPECT_FALSE(force_inequality_.isFeasible(robot_, s));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f_verbose(7*i+4) = 1.1 * std::sqrt((s.f(3*i)*s.f(3*i)+s.f(3*i+1)*s.f(3*i+1))/(mu_*mu_));
  }
  s.set_f();
  EXPECT_TRUE(force_inequality_.isFeasible(robot_, s));
}


TEST_F(FloatingBaseForceInequalityTest, computePrimalResidual) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f_verbose(7*i+4) = 1.1 * std::sqrt((s.f(3*i)*s.f(3*i)+s.f(3*i+1)*s.f(3*i+1))/(mu_*mu_));
  }
  s.set_f();
  ASSERT_TRUE(force_inequality_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  data.slack = Eigen::VectorXd::Random(dimc).array().abs();
  force_inequality_.computePrimalResidual(robot_, dtau_, s, data);
  Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(dimc));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    residual_ref.segment<6>(6*i).head(5) = - dtau_ * s.f_verbose.segment<7>(7*i).head(5);
    residual_ref.segment<6>(6*i).coeffRef(5) = - dtau_ * (mu_*mu_*s.f(3*i+2)*s.f(3*i+2) - s.f(3*i+0)*s.f(3*i+0) - s.f(3*i+1)*s.f(3*i+1));
  }
  residual_ref += data.slack;
  EXPECT_TRUE(data.residual.isApprox(residual_ref));
  force_inequality_.setSlackAndDual(robot_, dtau_, s, data);
  force_inequality_.computePrimalResidual(robot_, dtau_, s, data);
  EXPECT_TRUE(data.residual.isZero());
}


TEST_F(FloatingBaseForceInequalityTest, augmentDualResidual) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f_verbose(7*i+4) = 1.1 * std::sqrt((s.f(3*i)*s.f(3*i)+s.f(3*i+1)*s.f(3*i+1))/(mu_*mu_));
  }
  s.set_f();
  ASSERT_TRUE(force_inequality_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  data.dual = Eigen::VectorXd::Random(dimc).array().abs();
  KKTResidual kkt_residual(robot_);
  force_inequality_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6*robot_.num_point_contacts(), 
                                            7*robot_.num_point_contacts()));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    g_f(6*i+0, 7*i+0) = dtau_;
    g_f(6*i+1, 7*i+1) = dtau_;
    g_f(6*i+2, 7*i+2) = dtau_;
    g_f(6*i+3, 7*i+3) = dtau_;
    g_f(6*i+4, 7*i+4) = dtau_;
    g_f(6*i+5, 7*i+0) = - 2 * dtau_ * s.f(3*i+0);
    g_f(6*i+5, 7*i+1) = 2 * dtau_ * s.f(3*i+0);
    g_f(6*i+5, 7*i+2) = - 2 * dtau_ * s.f(3*i+1);
    g_f(6*i+5, 7*i+3) = 2 * dtau_ * s.f(3*i+1);
    g_f(6*i+5, 7*i+4) = 2 * mu_ * mu_ * dtau_ * s.f(3*i+2);
  }
  std::cout << g_f << std::endl;
  Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(7*robot_.num_point_contacts()));
  residual_ref = - g_f.transpose() * data.dual;
  EXPECT_TRUE(kkt_residual.lf().isApprox(residual_ref));
}


TEST_F(FloatingBaseForceInequalityTest, computeSlackDirection) {
  SplitSolution s(robot_);
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f_verbose(7*i+4) = 1.1 * std::sqrt((s.f(3*i)*s.f(3*i)+s.f(3*i+1)*s.f(3*i+1))/(mu_*mu_));
  }
  s.set_f();
  ASSERT_TRUE(force_inequality_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  data.residual = Eigen::VectorXd::Random(dimc);
  SplitDirection d(robot_);
  d.df() = Eigen::VectorXd::Random(7*robot_.num_point_contacts());
  force_inequality_.computeSlackDirection(robot_, dtau_, s, d, data);
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6*robot_.num_point_contacts(), 
                                            7*robot_.num_point_contacts()));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    g_f(6*i+0, 7*i+0) = dtau_;
    g_f(6*i+1, 7*i+1) = dtau_;
    g_f(6*i+2, 7*i+2) = dtau_;
    g_f(6*i+3, 7*i+3) = dtau_;
    g_f(6*i+4, 7*i+4) = dtau_;
    g_f(6*i+5, 7*i+0) = - 2 * dtau_ * s.f(3*i+0);
    g_f(6*i+5, 7*i+1) = 2 * dtau_ * s.f(3*i+0);
    g_f(6*i+5, 7*i+2) = - 2 * dtau_ * s.f(3*i+1);
    g_f(6*i+5, 7*i+3) = 2 * dtau_ * s.f(3*i+1);
    g_f(6*i+5, 7*i+4) = 2 * mu_ * mu_ * dtau_ * s.f(3*i+2);
  }
  std::cout << g_f << std::endl;
  Eigen::VectorXd dslack_ref(Eigen::VectorXd::Zero(7*robot_.num_point_contacts()));
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