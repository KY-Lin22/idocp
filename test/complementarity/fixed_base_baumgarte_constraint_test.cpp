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
#include "idocp/complementarity/baumgarte_constraint.hpp"
#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {

class FixedBaseBaumgarteConstraintTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    const std::vector<int> contact_frames = {18};
    const std::string urdf = "../urdf/iiwa14/iiwa14.urdf";
    robot_ = Robot(urdf, contact_frames, 0, 0);
    baumgarte_constraint_= BaumgarteConstraint(robot_);
  }

  virtual void TearDown() {
  }

  double dtau_;
  Eigen::VectorXd slack_, dual_, dslack_, ddual_;
  Robot robot_;
  BaumgarteConstraint baumgarte_constraint_;
};


TEST_F(FixedBaseBaumgarteConstraintTest, isFeasible) {
  SplitSolution s(robot_);
  ASSERT_TRUE(s.r.size() == 2);
  ASSERT_TRUE(robot_.num_point_contacts() == 1);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.r = - Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  robot_.updateKinematics(s.q, s.v, s.a);
  EXPECT_FALSE(baumgarte_constraint_.isFeasible(robot_, s));
}


TEST_F(FixedBaseBaumgarteConstraintTest, computePrimalResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  ConstraintComponentData data(6);
  data.slack = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  robot_.updateKinematics(s.q, s.v, s.a);
  baumgarte_constraint_.computePrimalResidual(robot_, dtau_, s, data);
  Eigen::VectorXd baum_residual(Eigen::VectorXd::Zero(3));
  robot_.computeBaumgarteResidual(baum_residual); 
  Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(6));
  residual_ref(0) = - dtau_ * (s.r(0) + baum_residual(0));
  residual_ref(1) = - dtau_ * (s.r(0) - baum_residual(0));
  residual_ref(2) = - dtau_ * (s.r(1) + baum_residual(1));
  residual_ref(3) = - dtau_ * (s.r(1) - baum_residual(1));
  residual_ref(4) = - dtau_ * baum_residual(2);
  residual_ref(5) = - dtau_ * (s.r(0)*s.r(0)+s.r(1)*s.r(1));
  residual_ref += data.slack;
  if (baumgarte_constraint_.isFeasible(robot_, s)) {
    EXPECT_TRUE(data.residual.isApprox(residual_ref));
  }
}


TEST_F(FixedBaseBaumgarteConstraintTest, augmentDualResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  ConstraintComponentData data(6);
  data.dual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  KKTResidual kkt_residual(robot_);
  robot_.updateKinematics(s.q, s.v, s.a);
  baumgarte_constraint_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_r(Eigen::MatrixXd::Zero(6, 2));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) =   dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) =   dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) =   dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();
  g_r(0, 0) = dtau_;
  g_r(1, 0) = dtau_;
  g_r(2, 1) = dtau_;
  g_r(3, 1) = dtau_;
  g_r(5, 0) = 2 * dtau_ * s.r(0);
  g_r(5, 1) = 2 * dtau_ * s.r(1);
  g_q.row(0) =   dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) =   dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) =   dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();
  g_v.row(0) =   dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) =   dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) =   dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  std::cout << g_r << std::endl;
  Eigen::VectorXd la_ref(Eigen::VectorXd::Zero(robot_.dimv()));
  Eigen::VectorXd lr_ref(Eigen::VectorXd::Zero(2));
  Eigen::VectorXd lq_ref(Eigen::VectorXd::Zero(robot_.dimv()));
  Eigen::VectorXd lv_ref(Eigen::VectorXd::Zero(robot_.dimv()));
  la_ref = - g_a.transpose() * data.dual;
  lr_ref = - g_r.transpose() * data.dual;
  lq_ref = - g_q.transpose() * data.dual;
  lv_ref = - g_v.transpose() * data.dual;
  EXPECT_TRUE(kkt_residual.la().isApprox(la_ref));
  EXPECT_TRUE(kkt_residual.lf().isZero());
  EXPECT_TRUE(kkt_residual.lr().isApprox(lr_ref));
  EXPECT_TRUE(kkt_residual.lq().isApprox(lq_ref));
  EXPECT_TRUE(kkt_residual.lv().isApprox(lv_ref));
}


TEST_F(FixedBaseBaumgarteConstraintTest, augmentCondensedHessian) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  KKTMatrix kkt_matrix(robot_);
  const Eigen::VectorXd diagonal = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  robot_.updateKinematics(s.q, s.v, s.a);
  KKTResidual kkt_residual(robot_);
  ConstraintComponentData data(6);
  baumgarte_constraint_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  baumgarte_constraint_.augmentCondensedHessian(robot_, dtau_, s, diagonal, kkt_matrix);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_r(Eigen::MatrixXd::Zero(6, 2));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) =   dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) =   dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) =   dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();
  g_r(0, 0) = dtau_;
  g_r(1, 0) = dtau_;
  g_r(2, 1) = dtau_;
  g_r(3, 1) = dtau_;
  g_r(5, 0) = 2 * dtau_ * s.r(0);
  g_r(5, 1) = 2 * dtau_ * s.r(1);
  g_q.row(0) =   dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) =   dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) =   dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();
  g_v.row(0) =   dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) =   dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) =   dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  Eigen::MatrixXd g_x(Eigen::MatrixXd::Zero(6, 3*robot_.dimv()+7));
  g_x.block(0, 0, 6, robot_.dimv()) = g_a;
  g_x.block(0, robot_.dimv()+5, 6, 2) = g_r;
  g_x.block(0, robot_.dimv()+7, 6, robot_.dimv()) = g_q;
  g_x.block(0, 2*robot_.dimv()+7, 6, robot_.dimv()) = g_v;
  std::cout << g_x << std::endl;

  KKTMatrix kkt_matrix_ref(robot_);
  kkt_matrix_ref.costHessian() = g_x.transpose() * diagonal.asDiagonal() * g_x;
  EXPECT_TRUE(kkt_matrix.Qaa().isApprox(kkt_matrix_ref.Qaa()));
  EXPECT_TRUE(kkt_matrix.Qaf().isApprox(kkt_matrix_ref.Qaf()));
  EXPECT_TRUE(kkt_matrix.Qar().isApprox(kkt_matrix_ref.Qar()));
  EXPECT_TRUE(kkt_matrix.Qaq().isApprox(kkt_matrix_ref.Qaq()));
  EXPECT_TRUE(kkt_matrix.Qav().isApprox(kkt_matrix_ref.Qav()));
  EXPECT_TRUE(kkt_matrix.Qff().isApprox(kkt_matrix_ref.Qff()));
  EXPECT_TRUE(kkt_matrix.Qfr().isApprox(kkt_matrix_ref.Qfr()));
  EXPECT_TRUE(kkt_matrix.Qfq().isApprox(kkt_matrix_ref.Qfq()));
  EXPECT_TRUE(kkt_matrix.Qfv().isApprox(kkt_matrix_ref.Qfv()));
  EXPECT_TRUE(kkt_matrix.Qrr().isApprox(kkt_matrix_ref.Qrr()));
  EXPECT_TRUE(kkt_matrix.Qrq().isApprox(kkt_matrix_ref.Qrq()));
  EXPECT_TRUE(kkt_matrix.Qrv().isApprox(kkt_matrix_ref.Qrv()));
  EXPECT_TRUE(kkt_matrix.Qqq().isApprox(kkt_matrix_ref.Qqq()));
  EXPECT_TRUE(kkt_matrix.Qqv().isApprox(kkt_matrix_ref.Qqv()));
  EXPECT_TRUE(kkt_matrix.Qvv().isApprox(kkt_matrix_ref.Qvv()));
}



TEST_F(FixedBaseBaumgarteConstraintTest, augmentComplementarityCondensedHessian) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  KKTMatrix kkt_matrix(robot_);
  const Eigen::VectorXd diagonal = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  robot_.updateKinematics(s.q, s.v, s.a);
  KKTResidual kkt_residual(robot_);
  ConstraintComponentData data(6);
  const double mu = std::abs(Eigen::VectorXd::Random(1)[0]);
  ContactForceConstraint force_constraint(robot_, mu);
  baumgarte_constraint_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  baumgarte_constraint_.augmentComplementarityCondensedHessian(robot_, dtau_, s, force_constraint, diagonal, kkt_matrix);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_r(Eigen::MatrixXd::Zero(6, 2));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) =   dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) =   dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) =   dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();
  g_r(0, 0) = dtau_;
  g_r(1, 0) = dtau_;
  g_r(2, 1) = dtau_;
  g_r(3, 1) = dtau_;
  g_r(5, 0) = 2 * dtau_ * s.r(0);
  g_r(5, 1) = 2 * dtau_ * s.r(1);
  g_q.row(0) =   dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) =   dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) =   dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();
  g_v.row(0) =   dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) =   dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) =   dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  Eigen::MatrixXd g_x(Eigen::MatrixXd::Zero(6, 3*robot_.dimv()+7));
  g_x.block(0, 0, 6, robot_.dimv()) = g_a;
  g_x.block(0, robot_.dimv()+5, 6, 2) = g_r;
  g_x.block(0, robot_.dimv()+7, 6, robot_.dimv()) = g_q;
  g_x.block(0, 2*robot_.dimv()+7, 6, robot_.dimv()) = g_v;
  std::cout << g_x << std::endl;

  Eigen::MatrixXd h_f(Eigen::MatrixXd::Zero(6, 5));
  h_f(0, 0) = dtau_;
  h_f(1, 1) = dtau_;
  h_f(2, 2) = dtau_;
  h_f(3, 3) = dtau_;
  h_f(4, 4) = dtau_;
  h_f(5, 0) = - 2 * dtau_ * s.f_3D(0);
  h_f(5, 1) = 2 * dtau_ * s.f_3D(0);
  h_f(5, 2) = - 2 * dtau_ * s.f_3D(1);
  h_f(5, 3) = 2 * dtau_ * s.f_3D(1);
  h_f(5, 4) = 2 * mu * mu * dtau_ * s.f_3D(2);
  Eigen::MatrixXd h_x(Eigen::MatrixXd::Zero(6, 3*robot_.dimv()+7));
  h_x.block(0, robot_.dimv(), 6, 5) = h_f;
  std::cout << h_x << std::endl;

  KKTMatrix kkt_matrix_ref(robot_);
  kkt_matrix_ref.costHessian() += h_x.transpose() * diagonal.asDiagonal() * g_x;
  kkt_matrix_ref.costHessian() += g_x.transpose() * diagonal.asDiagonal() * h_x;
  std::cout << kkt_matrix_ref.costHessian() << std::endl;
  EXPECT_TRUE(kkt_matrix.Qaa().isApprox(kkt_matrix_ref.Qaa()));
  EXPECT_TRUE(kkt_matrix.Qaf().isApprox(kkt_matrix_ref.Qaf()));
  EXPECT_TRUE(kkt_matrix.Qar().isApprox(kkt_matrix_ref.Qar()));
  EXPECT_TRUE(kkt_matrix.Qaq().isApprox(kkt_matrix_ref.Qaq()));
  EXPECT_TRUE(kkt_matrix.Qav().isApprox(kkt_matrix_ref.Qav()));
  EXPECT_TRUE(kkt_matrix.Qff().isApprox(kkt_matrix_ref.Qff()));
  EXPECT_TRUE(kkt_matrix.Qfr().isApprox(kkt_matrix_ref.Qfr()));
  EXPECT_TRUE(kkt_matrix.Qfq().isApprox(kkt_matrix_ref.Qfq()));
  EXPECT_TRUE(kkt_matrix.Qfv().isApprox(kkt_matrix_ref.Qfv()));
  EXPECT_TRUE(kkt_matrix.Qrr().isApprox(kkt_matrix_ref.Qrr()));
  EXPECT_TRUE(kkt_matrix.Qrq().isApprox(kkt_matrix_ref.Qrq()));
  EXPECT_TRUE(kkt_matrix.Qrv().isApprox(kkt_matrix_ref.Qrv()));
  EXPECT_TRUE(kkt_matrix.Qqq().isApprox(kkt_matrix_ref.Qqq()));
  EXPECT_TRUE(kkt_matrix.Qqv().isApprox(kkt_matrix_ref.Qqv()));
  EXPECT_TRUE(kkt_matrix.Qvv().isApprox(kkt_matrix_ref.Qvv()));
}


TEST_F(FixedBaseBaumgarteConstraintTest, augmentCondensedResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  const Eigen::VectorXd condensed_residual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  robot_.updateKinematics(s.q, s.v, s.a);
  KKTResidual kkt_residual(robot_);
  ConstraintComponentData data(6);
  baumgarte_constraint_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  kkt_residual.setZero();
  baumgarte_constraint_.augmentCondensedResidual(robot_, dtau_, s, condensed_residual, kkt_residual);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_r(Eigen::MatrixXd::Zero(6, 2));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) =   dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) =   dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) =   dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();
  g_r(0, 0) = dtau_;
  g_r(1, 0) = dtau_;
  g_r(2, 1) = dtau_;
  g_r(3, 1) = dtau_;
  g_r(5, 0) = 2 * dtau_ * s.r(0);
  g_r(5, 1) = 2 * dtau_ * s.r(1);
  g_q.row(0) =   dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) =   dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) =   dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();
  g_v.row(0) =   dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) =   dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) =   dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  Eigen::MatrixXd g_x(Eigen::MatrixXd::Zero(6, 3*robot_.dimv()+7));
  g_x.block(0, 0, 6, robot_.dimv()) = g_a;
  g_x.block(0, robot_.dimv()+5, 6, 2) = g_r;
  g_x.block(0, robot_.dimv()+7, 6, robot_.dimv()) = g_q;
  g_x.block(0, 2*robot_.dimv()+7, 6, robot_.dimv()) = g_v;
  std::cout << g_x << std::endl;

  KKTResidual kkt_residual_ref(robot_);
  kkt_residual_ref.KKT_residual.tail(3*robot_.dimv()+7*robot_.num_point_contacts()) 
      = - g_x.transpose() * condensed_residual;
  EXPECT_TRUE(kkt_residual.la().isApprox(kkt_residual_ref.la()));
  EXPECT_TRUE(kkt_residual.lf().isApprox(kkt_residual_ref.lf()));
  EXPECT_TRUE(kkt_residual.lr().isApprox(kkt_residual_ref.lr()));
  EXPECT_TRUE(kkt_residual.lq().isApprox(kkt_residual_ref.lq()));
  EXPECT_TRUE(kkt_residual.lv().isApprox(kkt_residual_ref.lv()));
}



TEST_F(FixedBaseBaumgarteConstraintTest, computeSlackDirection) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  SplitDirection d(robot_);
  d.da() = Eigen::VectorXd::Random(robot_.dimv());
  d.dr() = Eigen::VectorXd::Random(2);
  d.dq() = Eigen::VectorXd::Random(robot_.dimv());
  d.dv() = Eigen::VectorXd::Random(robot_.dimv());
  ConstraintComponentData data(6);
  data.residual = Eigen::VectorXd::Random(6*robot_.num_point_contacts());
  robot_.updateKinematics(s.q, s.v, s.a);
  KKTResidual kkt_residual(robot_);
  baumgarte_constraint_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  baumgarte_constraint_.computeSlackDirection(robot_, dtau_, s, d, data);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_r(Eigen::MatrixXd::Zero(6, 2));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) =   dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) =   dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) =   dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();
  g_r(0, 0) = dtau_;
  g_r(1, 0) = dtau_;
  g_r(2, 1) = dtau_;
  g_r(3, 1) = dtau_;
  g_r(5, 0) = 2 * dtau_ * s.r(0);
  g_r(5, 1) = 2 * dtau_ * s.r(1);
  g_q.row(0) =   dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) =   dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) =   dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();
  g_v.row(0) =   dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) =   dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) =   dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  Eigen::VectorXd dslack_ref(Eigen::VectorXd::Zero(6));
  dslack_ref += g_a * d.da();
  dslack_ref += g_r * d.dr();
  dslack_ref += g_q * d.dq();
  dslack_ref += g_v * d.dv();
  dslack_ref -= data.residual;
  EXPECT_TRUE(data.dslack.isApprox(dslack_ref));
  std::cout << data.dslack.transpose() << std::endl;
  std::cout << dslack_ref.transpose() << std::endl;
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}