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
    mu_ = std::abs(Eigen::VectorXd::Random(1)[0]);
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

  double mu_, barrier_, dtau_, fraction_to_boundary_rate_;
  Eigen::VectorXd slack_, dual_, dslack_, ddual_;
  Robot robot_;
  BaumgarteInequality baumgarte_inequality_;
};


TEST_F(FixedBaseBaumgarteInequalityTest, isFeasible) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
  s.f_verbose(5) = -1;
  s.f_verbose(6) = -1;
  assert(s.f.size() == 3);
  assert(s.f_verbose.size() == 7);
  assert(robot_.num_point_contacts() == 1);
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
  ASSERT_TRUE(baumgarte_inequality_.isFeasible(robot_, s));
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
  EXPECT_TRUE(data.residual.isApprox(residual_ref));
  std::cout << data.residual.transpose() << std::endl;
  std::cout << residual_ref.transpose() << std::endl;
  // baumgarte_inequality_.setSlackAndDual(robot_, dtau_, s, data);
  // baumgarte_inequality_.computePrimalResidual(robot_, dtau_, s, data);
  // EXPECT_TRUE(data.residual.isZero());
}


// TEST_F(FixedBaseBaumgarteInequalityTest, augmentDualResidual) {
//   SplitSolution s(robot_);
//   s.f_verbose = Eigen::VectorXd::Random(7*robot_.num_point_contacts()).array().abs();
//   s.set_f();
//   s.f_verbose(4) = 1.1 * std::sqrt((s.f(0)*s.f(0)+s.f(1)*s.f(1))/(mu_*mu_));
//   s.set_f();
//   ASSERT_TRUE(baumgarte_inequality_.isFeasible(robot_, s));
//   ConstraintComponentData data(6);
//   data.dual = Eigen::VectorXd::Random(6*robot_.num_point_contacts()).array().abs();
//   KKTResidual kkt_residual(robot_);
//   baumgarte_inequality_.augmentDualResidual(robot_, dtau_, s, data, kkt_residual);
//   Eigen::MatrixXd g_x (Eigen::MatrixXd::Zero(6, 7));
//   g_x(0, 0) = dtau_;
//   g_x(1, 1) = dtau_;
//   g_x(2, 2) = dtau_;
//   g_x(3, 3) = dtau_;
//   g_x(4, 4) = dtau_;
//   g_x(5, 0) = - 2 * dtau_ * s.f(0);
//   g_x(5, 1) = 2 * dtau_ * s.f(0);
//   g_x(5, 2) = - 2 * dtau_ * s.f(1);
//   g_x(5, 3) = 2 * dtau_ * s.f(1);
//   g_x(5, 4) = 2 * mu_ * mu_ * dtau_ * s.f(2);
//   Eigen::VectorXd residual_ref(Eigen::VectorXd::Zero(7));
//   residual_ref = - g_x.transpose() * data.dual;
//   EXPECT_TRUE(kkt_residual.lf().isApprox(residual_ref));
// }

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}