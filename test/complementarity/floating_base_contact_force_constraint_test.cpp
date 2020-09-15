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
#include "idocp/complementarity/contact_force_constraint.hpp"
#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {

class FloatingBaseContactForceConstraintTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    mu_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    const std::vector<int> contact_frames = {14, 24, 34, 44};
    const std::string urdf = "../urdf/anymal/anymal.urdf";
    robot_ = Robot(urdf, contact_frames, 0, 0);
    contact_force_constraint_ = ContactForceConstraint(robot_, mu_);
  }

  virtual void TearDown() {
  }

  double mu_, dtau_;
  Eigen::VectorXd slack_, dual_, dslack_, ddual_;
  Robot robot_;
  ContactForceConstraint contact_force_constraint_;
};


TEST_F(FloatingBaseContactForceConstraintTest, isFeasible) {
  EXPECT_DOUBLE_EQ(contact_force_constraint_.mu(), mu_);
  SplitSolution s(robot_);
  s.f = - Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  ASSERT_TRUE(s.f.size() == 5*robot_.num_point_contacts());
  ASSERT_TRUE(s.f_3D.size() == 3*robot_.num_point_contacts());
  ASSERT_TRUE(robot_.num_point_contacts() == 4);
  s.set_f_3D();
  EXPECT_FALSE(contact_force_constraint_.isFeasible(robot_, s));
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f(5*i+4) = 0.9 * std::sqrt((s.f_3D(3*i)*s.f_3D(3*i)+s.f_3D(3*i+1)*s.f_3D(3*i+1))/(mu_*mu_));
  }
  s.set_f_3D();
  EXPECT_FALSE(contact_force_constraint_.isFeasible(robot_, s));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f(5*i+4) = 1.1 * std::sqrt((s.f_3D(3*i)*s.f_3D(3*i)+s.f_3D(3*i+1)*s.f_3D(3*i+1))/(mu_*mu_));
  }
  s.set_f_3D();
  EXPECT_TRUE(contact_force_constraint_.isFeasible(robot_, s));
}


TEST_F(FloatingBaseContactForceConstraintTest, computePrimalResidual) {
  SplitSolution s(robot_);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f(5*i+4) = 1.1 * std::sqrt((s.f_3D(3*i)*s.f_3D(3*i)+s.f_3D(3*i+1)*s.f_3D(3*i+1))/(mu_*mu_));
  }
  s.set_f_3D();
  ASSERT_TRUE(contact_force_constraint_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  data.slack = Eigen::VectorXd::Random(dimc).array().abs();
  contact_force_constraint_.computePrimalResidual(robot_, dtau_, s, data);
  Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(dimc));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    residual_ref.segment<6>(6*i).head(5) = data.slack.segment<6>(6*i).head(5) - dtau_ * s.f.segment<5>(5*i);
    residual_ref.segment<6>(6*i).coeffRef(5) 
        = data.slack.segment<6>(6*i).coeffRef(5) - dtau_ * (mu_*mu_*s.f_3D(3*i+2)*s.f_3D(3*i+2) - s.f_3D(3*i+0)*s.f_3D(3*i+0) - s.f_3D(3*i+1)*s.f_3D(3*i+1));
  }
  EXPECT_TRUE(data.residual.isApprox(residual_ref));
  contact_force_constraint_.setSlack(robot_, dtau_, s, data);
  contact_force_constraint_.computePrimalResidual(robot_, dtau_, s, data);
  EXPECT_TRUE(data.residual.isZero());
}


TEST_F(FloatingBaseContactForceConstraintTest, augmentDualResidual) {
  SplitSolution s(robot_);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f(5*i+4) = 1.1 * std::sqrt((s.f_3D(3*i)*s.f_3D(3*i)+s.f_3D(3*i+1)*s.f_3D(3*i+1))/(mu_*mu_));
  }
  s.set_f_3D();
  ASSERT_TRUE(contact_force_constraint_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  data.dual = Eigen::VectorXd::Random(dimc).array().abs();
  KKTResidual kkt_residual(robot_);
  contact_force_constraint_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6*robot_.num_point_contacts(), 
                                            5*robot_.num_point_contacts()));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    g_f(6*i+0, 5*i+0) = dtau_;
    g_f(6*i+1, 5*i+1) = dtau_;
    g_f(6*i+2, 5*i+2) = dtau_;
    g_f(6*i+3, 5*i+3) = dtau_;
    g_f(6*i+4, 5*i+4) = dtau_;
    g_f(6*i+5, 5*i+0) = - 2 * dtau_ * s.f_3D(3*i+0);
    g_f(6*i+5, 5*i+1) = 2 * dtau_ * s.f_3D(3*i+0);
    g_f(6*i+5, 5*i+2) = - 2 * dtau_ * s.f_3D(3*i+1);
    g_f(6*i+5, 5*i+3) = 2 * dtau_ * s.f_3D(3*i+1);
    g_f(6*i+5, 5*i+4) = 2 * mu_ * mu_ * dtau_ * s.f_3D(3*i+2);
  }
  std::cout << g_f << std::endl;
  Eigen::VectorXd lf_ref(Eigen::VectorXd::Zero(5*robot_.num_point_contacts()));
  lf_ref = - g_f.transpose() * data.dual;
  EXPECT_TRUE(kkt_residual.lf().isApprox(lf_ref));
}


TEST_F(FloatingBaseContactForceConstraintTest, augmentCondensedHessian) {
  SplitSolution s(robot_);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f(5*i+4) = 1.1 * std::sqrt((s.f_3D(3*i)*s.f_3D(3*i)+s.f_3D(3*i+1)*s.f_3D(3*i+1))/(mu_*mu_));
  }
  s.set_f_3D();
  ASSERT_TRUE(contact_force_constraint_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  data.residual = Eigen::VectorXd::Random(dimc);
  KKTMatrix kkt_matrix(robot_);
  const Eigen::VectorXd diagonal = Eigen::VectorXd::Random(dimc).array().abs();
  contact_force_constraint_.augmentCondensedHessian(robot_, dtau_, s, diagonal, kkt_matrix);
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6*robot_.num_point_contacts(), 
                                            5*robot_.num_point_contacts()));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    g_f(6*i+0, 5*i+0) = dtau_;
    g_f(6*i+1, 5*i+1) = dtau_;
    g_f(6*i+2, 5*i+2) = dtau_;
    g_f(6*i+3, 5*i+3) = dtau_;
    g_f(6*i+4, 5*i+4) = dtau_;
    g_f(6*i+5, 5*i+0) = - 2 * dtau_ * s.f_3D(3*i+0);
    g_f(6*i+5, 5*i+1) = 2 * dtau_ * s.f_3D(3*i+0);
    g_f(6*i+5, 5*i+2) = - 2 * dtau_ * s.f_3D(3*i+1);
    g_f(6*i+5, 5*i+3) = 2 * dtau_ * s.f_3D(3*i+1);
    g_f(6*i+5, 5*i+4) = 2 * mu_ * mu_ * dtau_ * s.f_3D(3*i+2);
  }
  std::cout << g_f << std::endl;
  Eigen::MatrixXd Qff_ref(Eigen::MatrixXd::Zero(5*robot_.num_point_contacts(), 5*robot_.num_point_contacts()));
  Qff_ref = g_f.transpose() * diagonal.asDiagonal() * g_f;
  EXPECT_TRUE(kkt_matrix.Qff().isApprox(Qff_ref));
  std::cout << Qff_ref << std::endl;
}


TEST_F(FloatingBaseContactForceConstraintTest, augmentCondensedResidual) {
  SplitSolution s(robot_);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f(5*i+4) = 1.1 * std::sqrt((s.f_3D(3*i)*s.f_3D(3*i)+s.f_3D(3*i+1)*s.f_3D(3*i+1))/(mu_*mu_));
  }
  s.set_f_3D();
  ASSERT_TRUE(contact_force_constraint_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  data.residual = Eigen::VectorXd::Random(dimc);
  KKTResidual kkt_residual(robot_);
  const Eigen::VectorXd condensed_residual = Eigen::VectorXd::Random(dimc).array().abs();
  contact_force_constraint_.augmentCondensedResidual(robot_, dtau_, s, condensed_residual, kkt_residual);
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6*robot_.num_point_contacts(), 
                                            5*robot_.num_point_contacts()));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    g_f(6*i+0, 5*i+0) = dtau_;
    g_f(6*i+1, 5*i+1) = dtau_;
    g_f(6*i+2, 5*i+2) = dtau_;
    g_f(6*i+3, 5*i+3) = dtau_;
    g_f(6*i+4, 5*i+4) = dtau_;
    g_f(6*i+5, 5*i+0) = - 2 * dtau_ * s.f_3D(3*i+0);
    g_f(6*i+5, 5*i+1) = 2 * dtau_ * s.f_3D(3*i+0);
    g_f(6*i+5, 5*i+2) = - 2 * dtau_ * s.f_3D(3*i+1);
    g_f(6*i+5, 5*i+3) = 2 * dtau_ * s.f_3D(3*i+1);
    g_f(6*i+5, 5*i+4) = 2 * mu_ * mu_ * dtau_ * s.f_3D(3*i+2);
  }
  std::cout << g_f << std::endl;
  Eigen::VectorXd lf_ref(Eigen::VectorXd::Zero(5*robot_.num_point_contacts()));
  lf_ref = - g_f.transpose() * condensed_residual;
  EXPECT_TRUE(kkt_residual.lf().isApprox(lf_ref));
}


TEST_F(FloatingBaseContactForceConstraintTest, computeSlackDirection) {
  SplitSolution s(robot_);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    s.f(5*i+4) = 1.1 * std::sqrt((s.f_3D(3*i)*s.f_3D(3*i)+s.f_3D(3*i+1)*s.f_3D(3*i+1))/(mu_*mu_));
  }
  s.set_f_3D();
  ASSERT_TRUE(contact_force_constraint_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  data.residual = Eigen::VectorXd::Random(dimc);
  SplitDirection d(robot_);
  d.df() = Eigen::VectorXd::Random(5*robot_.num_point_contacts());
  contact_force_constraint_.computeSlackDirection(robot_, dtau_, s, d, data);
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6*robot_.num_point_contacts(), 
                                            5*robot_.num_point_contacts()));
  for (int i=0; i<robot_.num_point_contacts(); ++i) {
    g_f(6*i+0, 5*i+0) = dtau_;
    g_f(6*i+1, 5*i+1) = dtau_;
    g_f(6*i+2, 5*i+2) = dtau_;
    g_f(6*i+3, 5*i+3) = dtau_;
    g_f(6*i+4, 5*i+4) = dtau_;
    g_f(6*i+5, 5*i+0) = - 2 * dtau_ * s.f_3D(3*i+0);
    g_f(6*i+5, 5*i+1) = 2 * dtau_ * s.f_3D(3*i+0);
    g_f(6*i+5, 5*i+2) = - 2 * dtau_ * s.f_3D(3*i+1);
    g_f(6*i+5, 5*i+3) = 2 * dtau_ * s.f_3D(3*i+1);
    g_f(6*i+5, 5*i+4) = 2 * mu_ * mu_ * dtau_ * s.f_3D(3*i+2);
  }
  std::cout << g_f << std::endl;
  Eigen::VectorXd dslack_ref(Eigen::VectorXd::Zero(6*robot_.num_point_contacts()));
  dslack_ref = g_f * d.df() - data.residual;
  EXPECT_TRUE(data.dslack.isApprox(dslack_ref));
}


TEST_F(FloatingBaseContactForceConstraintTest, withoutContacts) {
  SplitSolution s(robot_);
  const std::string urdf = "../urdf/anymal/anymal.urdf";
  robot_ = Robot(urdf);
  contact_force_constraint_ = ContactForceConstraint(robot_, mu_);
  EXPECT_TRUE(contact_force_constraint_.isFeasible(robot_, s));
  const int dimc = 6*robot_.num_point_contacts();
  ConstraintComponentData data(dimc);
  contact_force_constraint_.setSlack(robot_, dtau_, s, data);
  contact_force_constraint_.computePrimalResidual(robot_, dtau_, s, data);
  EXPECT_TRUE(data.residual.isZero());
  KKTResidual kkt_residual(robot_);
  contact_force_constraint_.augmentDualResidual(robot_, dtau_, s, data, 
                                                kkt_residual);
  EXPECT_TRUE(kkt_residual.KKT_residual.isZero());
  KKTMatrix kkt_matrix(robot_);
  Eigen::VectorXd diag = Eigen::VectorXd::Random(dimc);
  contact_force_constraint_.augmentCondensedHessian(robot_, dtau_, s, diag, 
                                                    kkt_matrix);
  contact_force_constraint_.augmentCondensedResidual(robot_, dtau_, s, diag, 
                                                     kkt_residual);
  EXPECT_TRUE(kkt_matrix.costHessian().isZero());
  EXPECT_TRUE(kkt_matrix.constraintsJacobian().isZero());
  EXPECT_TRUE(kkt_residual.KKT_residual.isZero());
  SplitDirection d(robot_);
  d.df() = Eigen::VectorXd::Random(5*robot_.num_point_contacts());
  contact_force_constraint_.computeSlackDirection(robot_, dtau_, s, d, data);
  EXPECT_TRUE(data.dslack.isZero());
}


} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}