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
#include "idocp/complementarity/baumgarte_inequality.hpp"
#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {

class FixedBaseBaumgarteInequalityTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    barrier_ = 1.0e-04;
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    fraction_to_boundary_rate_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    const std::vector<int> contact_frames = {18};
    const std::string urdf = "../urdf/iiwa14/iiwa14.urdf";
    robot_ = Robot(urdf, contact_frames, 0, 0);
    baumgarte_inequality_= BaumgarteInequality(robot_, barrier_, fraction_to_boundary_rate_);
  }

  virtual void TearDown() {
  }

  double barrier_, dtau_, fraction_to_boundary_rate_;
  Eigen::VectorXd slack_, dual_, dslack_, ddual_;
  Robot robot_;
  BaumgarteInequality baumgarte_inequality_;
};


TEST_F(FixedBaseBaumgarteInequalityTest, isFeasible) {
  SplitSolution s(robot_);
  assert(s.f.size() == 3);
  assert(s.f_verbose.size() == 7);
  assert(robot_.num_point_contacts() == 1);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.f_verbose(5) = -1;
  s.f_verbose(6) = -1;
  s.set_f();
  robot_.updateKinematics(s.q, s.v, s.a);
  EXPECT_FALSE(baumgarte_inequality_.isFeasible(robot_, s));
}


TEST_F(FixedBaseBaumgarteInequalityTest, computePrimalResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  ConstraintComponentData data(6);
  data.slack = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  robot_.updateKinematics(s.q, s.v, s.a);
  baumgarte_inequality_.computePrimalResidual(robot_, dtau_, s, data);
  Eigen::VectorXd baum_residual(Eigen::VectorXd::Zero(3));
  robot_.computeBaumgarteResidual(baum_residual); 
  Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(6));
  residual_ref(0) = - dtau_ * (s.f_verbose(5) + baum_residual(0));
  residual_ref(1) = - dtau_ * (s.f_verbose(5) - baum_residual(0));
  residual_ref(2) = - dtau_ * (s.f_verbose(6) + baum_residual(1));
  residual_ref(3) = - dtau_ * (s.f_verbose(6) - baum_residual(1));
  residual_ref(4) = - dtau_ * baum_residual(2);
  residual_ref(5) = - dtau_ * (s.f_verbose(5)*s.f_verbose(5)+s.f_verbose(6)*s.f_verbose(6));
  residual_ref += data.slack;
  if (baumgarte_inequality_.isFeasible(robot_, s)) {
    EXPECT_TRUE(data.residual.isApprox(residual_ref));
  }
}


TEST_F(FixedBaseBaumgarteInequalityTest, augmentDualResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  ConstraintComponentData data(6);
  data.dual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  KKTResidual kkt_residual(robot_);
  robot_.updateKinematics(s.q, s.v, s.a);
  baumgarte_inequality_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6, 7));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) = dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) = dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) = dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();

  g_f(0, 5) = dtau_;
  g_f(1, 5) = dtau_;
  g_f(2, 6) = dtau_;
  g_f(3, 6) = dtau_;
  g_f(5, 5) = 2 * dtau_ * s.f_verbose(5);
  g_f(5, 6) = 2 * dtau_ * s.f_verbose(6);

  g_q.row(0) = dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) = dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) = dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();

  g_v.row(0) = dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) = dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) = dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  std::cout << g_f << std::endl;
  Eigen::VectorXd la_ref(Eigen::VectorXd::Zero(robot_.dimv()));
  Eigen::VectorXd lf_ref(Eigen::VectorXd::Zero(7));
  Eigen::VectorXd lq_ref(Eigen::VectorXd::Zero(robot_.dimv()));
  Eigen::VectorXd lv_ref(Eigen::VectorXd::Zero(robot_.dimv()));
  la_ref = - g_a.transpose() * data.dual;
  lf_ref = - g_f.transpose() * data.dual;
  lq_ref = - g_q.transpose() * data.dual;
  lv_ref = - g_v.transpose() * data.dual;
  EXPECT_TRUE(kkt_residual.la().isApprox(la_ref));
  EXPECT_TRUE(kkt_residual.lf().isApprox(lf_ref));
  EXPECT_TRUE(kkt_residual.lq().isApprox(lq_ref));
  EXPECT_TRUE(kkt_residual.lv().isApprox(lv_ref));
}


TEST_F(FixedBaseBaumgarteInequalityTest, augmentCondensedHessian) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  KKTMatrix kkt_matrix(robot_);
  const Eigen::VectorXd diagonal = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  robot_.updateKinematics(s.q, s.v, s.a);
  KKTResidual kkt_residual(robot_);
  ConstraintComponentData data(6);
  baumgarte_inequality_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  baumgarte_inequality_.augmentCondensedHessian(robot_, dtau_, s, diagonal, kkt_matrix);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6, 7));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) = dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) = dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) = dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();

  g_f(0, 5) = dtau_;
  g_f(1, 5) = dtau_;
  g_f(2, 6) = dtau_;
  g_f(3, 6) = dtau_;
  g_f(5, 5) = 2 * dtau_ * s.f_verbose(5);
  g_f(5, 6) = 2 * dtau_ * s.f_verbose(6);

  g_q.row(0) = dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) = dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) = dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();

  g_v.row(0) = dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) = dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) = dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  Eigen::MatrixXd g_x(Eigen::MatrixXd::Zero(6, 3*robot_.dimv()+7));
  g_x.block(0, 0, 6, robot_.dimv()) = g_a;
  g_x.block(0, robot_.dimv(), 6, 7) = g_f;
  g_x.block(0, robot_.dimv()+7, 6, robot_.dimv()) = g_q;
  g_x.block(0, 2*robot_.dimv()+7, 6, robot_.dimv()) = g_v;
  std::cout << g_x << std::endl;

  KKTMatrix kkt_matrix_ref(robot_);
  kkt_matrix_ref.costHessian() = g_x.transpose() * diagonal.asDiagonal() * g_x;
  EXPECT_TRUE(kkt_matrix.Qaa().isApprox(kkt_matrix_ref.Qaa()));
  EXPECT_TRUE(kkt_matrix.Qaf().isApprox(kkt_matrix_ref.Qaf()));
  EXPECT_TRUE(kkt_matrix.Qaq().isApprox(kkt_matrix_ref.Qaq()));
  EXPECT_TRUE(kkt_matrix.Qav().isApprox(kkt_matrix_ref.Qav()));
  EXPECT_TRUE(kkt_matrix.Qff().isApprox(kkt_matrix_ref.Qff()));
  EXPECT_TRUE(kkt_matrix.Qfq().isApprox(kkt_matrix_ref.Qfq()));
  EXPECT_TRUE(kkt_matrix.Qfv().isApprox(kkt_matrix_ref.Qfv()));
  EXPECT_TRUE(kkt_matrix.Qqq().isApprox(kkt_matrix_ref.Qqq()));
  EXPECT_TRUE(kkt_matrix.Qqv().isApprox(kkt_matrix_ref.Qqv()));
  EXPECT_TRUE(kkt_matrix.Qvv().isApprox(kkt_matrix_ref.Qvv()));
}


TEST_F(FixedBaseBaumgarteInequalityTest, augmentCondensedResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  const Eigen::VectorXd condensed_residual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
  robot_.updateKinematics(s.q, s.v, s.a);
  KKTResidual kkt_residual(robot_);
  ConstraintComponentData data(6);
  baumgarte_inequality_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  kkt_residual.setZero();
  baumgarte_inequality_.augmentCondensedResidual(robot_, dtau_, s, condensed_residual, kkt_residual);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6, 7));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) = dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) = dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) = dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();

  g_f(0, 5) = dtau_;
  g_f(1, 5) = dtau_;
  g_f(2, 6) = dtau_;
  g_f(3, 6) = dtau_;
  g_f(5, 5) = 2 * dtau_ * s.f_verbose(5);
  g_f(5, 6) = 2 * dtau_ * s.f_verbose(6);

  g_q.row(0) = dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) = dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) = dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();

  g_v.row(0) = dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) = dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) = dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  Eigen::MatrixXd g_x(Eigen::MatrixXd::Zero(6, 3*robot_.dimv()+7));
  g_x.block(0, 0, 6, robot_.dimv()) = g_a;
  g_x.block(0, robot_.dimv(), 6, 7) = g_f;
  g_x.block(0, robot_.dimv()+7, 6, robot_.dimv()) = g_q;
  g_x.block(0, 2*robot_.dimv()+7, 6, robot_.dimv()) = g_v;
  std::cout << g_x << std::endl;

  KKTResidual kkt_residual_ref(robot_);
  kkt_residual_ref.KKT_residual().tail(3*robot_.dimv()+7*robot_.num_point_contacts()) 
      = g_x.transpose() * condensed_residual;
  EXPECT_TRUE(kkt_residual.la().isApprox(kkt_residual_ref.la()));
  EXPECT_TRUE(kkt_residual.lf().isApprox(kkt_residual_ref.lf()));
  EXPECT_TRUE(kkt_residual.lq().isApprox(kkt_residual_ref.lq()));
  EXPECT_TRUE(kkt_residual.lv().isApprox(kkt_residual_ref.lv()));
}



TEST_F(FixedBaseBaumgarteInequalityTest, computeSlackDirection) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.set_f();
  SplitDirection d(robot_);
  d.da() = Eigen::VectorXd::Random(robot_.dimv());
  d.df() = Eigen::VectorXd::Random(7);
  d.dq() = Eigen::VectorXd::Random(robot_.dimv());
  d.dv() = Eigen::VectorXd::Random(robot_.dimv());
  ConstraintComponentData data(6);
  data.residual = Eigen::VectorXd::Random(6*robot_.num_point_contacts());
  robot_.updateKinematics(s.q, s.v, s.a);
  KKTResidual kkt_residual(robot_);
  baumgarte_inequality_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
  baumgarte_inequality_.computeSlackDirection(robot_, dtau_, s, d, data);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd g_a(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_f(Eigen::MatrixXd::Zero(6, 7));
  Eigen::MatrixXd g_q(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  Eigen::MatrixXd g_v(Eigen::MatrixXd::Zero(6, robot_.dimv()));
  g_a.row(0) = dtau_ * dbaum_da.row(0);
  g_a.row(1) = - dtau_ * dbaum_da.row(0);
  g_a.row(2) = dtau_ * dbaum_da.row(1);
  g_a.row(3) = - dtau_ * dbaum_da.row(1);
  g_a.row(4) = dtau_ * dbaum_da.row(2);
  g_a.row(5).setZero();

  g_f(0, 5) = dtau_;
  g_f(1, 5) = dtau_;
  g_f(2, 6) = dtau_;
  g_f(3, 6) = dtau_;
  g_f(5, 5) = 2 * dtau_ * s.f_verbose(5);
  g_f(5, 6) = 2 * dtau_ * s.f_verbose(6);

  g_q.row(0) = dtau_ * dbaum_dq.row(0);
  g_q.row(1) = - dtau_ * dbaum_dq.row(0);
  g_q.row(2) = dtau_ * dbaum_dq.row(1);
  g_q.row(3) = - dtau_ * dbaum_dq.row(1);
  g_q.row(4) = dtau_ * dbaum_dq.row(2);
  g_q.row(5).setZero();

  g_v.row(0) = dtau_ * dbaum_dv.row(0);
  g_v.row(1) = - dtau_ * dbaum_dv.row(0);
  g_v.row(2) = dtau_ * dbaum_dv.row(1);
  g_v.row(3) = - dtau_ * dbaum_dv.row(1);
  g_v.row(4) = dtau_ * dbaum_dv.row(2);
  g_v.row(5).setZero();

  std::cout << g_f << std::endl;
  Eigen::VectorXd dslack_ref(Eigen::VectorXd::Zero(6));
  dslack_ref += g_a * d.da();
  dslack_ref += g_f * d.df();
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