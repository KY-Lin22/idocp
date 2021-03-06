#include <string>

#include <gtest/gtest.h>
#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/cost/contact_cost.hpp"
#include "idocp/cost/cost_function_data.hpp"
#include "idocp/ocp/split_solution.hpp"
#include "idocp/ocp/kkt_residual.hpp"
#include "idocp/ocp/kkt_matrix.hpp"


namespace idocp {

class FixedBaseContactCostTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    std::random_device rnd;
    urdf_ = "../urdf/iiwa14/iiwa14.urdf";
    contact_frames_ = {18};
    robot_ = Robot(urdf_, contact_frames_);
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    t_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    data_ = CostFunctionData(robot_);
    s = SplitSolution(robot_);
    kkt_res = KKTResidual(robot_);
    kkt_mat = KKTMatrix(robot_);
  }

  virtual void TearDown() {
  }

  double dtau_, t_;
  std::vector<int> contact_frames_;
  std::string urdf_;
  Robot robot_;
  CostFunctionData data_;
  SplitSolution s;
  KKTResidual kkt_res;
  KKTMatrix kkt_mat;
};


TEST_F(FixedBaseContactCostTest, setWeights) {
  const int dimq = robot_.dimq();
  const int dimv = robot_.dimv();
  std::vector<Eigen::Vector3d> f_weight, f_ref;
  f_weight.push_back(Eigen::Vector3d::Random());
  f_ref.push_back(Eigen::Vector3d::Random());
  ContactCost cost(robot_);
  EXPECT_FALSE(cost.useKinematics());
  cost.set_f_weight(f_weight);
  cost.set_f_ref(f_ref);
  s = SplitSolution::Random(robot_);
  ASSERT_EQ(robot_.dimf(), 0);
  kkt_res.setContactStatus(robot_);
  kkt_mat.setContactStatus(robot_);
  EXPECT_DOUBLE_EQ(cost.l(robot_, data_, t_, dtau_, s), 0);
  Eigen::VectorXd lf = Eigen::VectorXd::Zero(robot_.max_dimf());
  cost.lf(robot_, data_, t_, dtau_, s, kkt_res);
  EXPECT_TRUE(kkt_res.lf().isZero());
  Eigen::MatrixXd lff = Eigen::MatrixXd::Zero(robot_.max_dimf(), robot_.max_dimf());
  cost.lff(robot_, data_, t_, dtau_, s, kkt_mat);
  EXPECT_TRUE(kkt_mat.Qff().isZero());
  std::vector<bool> active_contacts;
  active_contacts = {true};
  robot_.setContactStatus(active_contacts);
  ASSERT_EQ(robot_.dimf(), 3);
  kkt_res.setContactStatus(robot_);
  kkt_mat.setContactStatus(robot_);
  const double l_ref = 0.5 * dtau_ * (f_weight[0].array()* (s.f[0]-f_ref[0]).array()*(s.f[0]-f_ref[0]).array()).sum();
  EXPECT_DOUBLE_EQ(cost.l(robot_, data_, t_, dtau_, s), l_ref);
  cost.lf(robot_, data_, t_, dtau_, s, kkt_res);
  Eigen::VectorXd lf_ref = Eigen::VectorXd::Zero(robot_.max_dimf());
  lf_ref = dtau_ * f_weight[0].asDiagonal() * (s.f[0]-f_ref[0]);
  EXPECT_TRUE(kkt_res.lf().isApprox(lf_ref));
  cost.lff(robot_, data_, t_, dtau_, s, kkt_mat);
  Eigen::MatrixXd lff_ref = Eigen::MatrixXd::Zero(robot_.max_dimf(), robot_.max_dimf());
  lff_ref = dtau_*f_weight[0].asDiagonal();
  EXPECT_TRUE(kkt_mat.Qff().isApprox(lff_ref));
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}