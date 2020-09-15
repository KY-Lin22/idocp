#include <string>
#include <memory>

#include <gtest/gtest.h>
#include "Eigen/Core"
#include "Eigen/LU"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/riccati_gain.hpp"


namespace idocp {

class RiccatiGainTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    std::random_device rnd;
    fixed_base_urdf_ = "../urdf/iiwa14/iiwa14.urdf";
    floating_base_urdf_ = "../urdf/anymal/anymal.urdf";
    fixed_base_robot_ = Robot(fixed_base_urdf_);
    floating_base_robot_ = Robot(floating_base_urdf_);
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
  }

  virtual void TearDown() {
  }

  double dtau_;
  std::string fixed_base_urdf_, floating_base_urdf_;
  Robot fixed_base_robot_, floating_base_robot_;
};


TEST_F(RiccatiGainTest, fixed_base_without_contacts) {
  const int dimv = fixed_base_robot_.dimv();
  const int dimafr = fixed_base_robot_.dimv() + 7*fixed_base_robot_.num_point_contacts();
  const int dimfr = 7*fixed_base_robot_.num_point_contacts();
  const int dimc = fixed_base_robot_.dim_passive();
  ASSERT_EQ(dimfr, 0);
  ASSERT_EQ(dimc, 0);
  const Eigen::MatrixXd Ginv_seed = Eigen::MatrixXd::Random(dimafr+dimc, dimafr+dimc);
  const Eigen::MatrixXd Ginv = Ginv_seed * Ginv_seed.transpose();
  const Eigen::MatrixXd Q_afr_qv = Eigen::MatrixXd::Random(dimafr, 2*dimv);
  const Eigen::MatrixXd C_qv = Eigen::MatrixXd::Random(dimc, 2*dimv);
  RiccatiGain gain(fixed_base_robot_);
  gain.computeFeedbackGain(Ginv, Q_afr_qv, C_qv);
  Eigen::MatrixXd HC = Eigen::MatrixXd::Random(dimafr+dimc, 2*dimv);
  HC.topRows(dimafr) = Q_afr_qv;
  HC.bottomRows(dimc) = C_qv;
  const Eigen::MatrixXd K_ref = - Ginv * HC;
  ASSERT_EQ(K_ref.rows(), dimv);
  ASSERT_EQ(K_ref.cols(), 2*dimv);
  EXPECT_TRUE(K_ref.topLeftCorner(dimv, dimv).isApprox(gain.Kaq()));
  EXPECT_TRUE(K_ref.topRightCorner(dimv, dimv).isApprox(gain.Kav()));
  EXPECT_TRUE(K_ref.block(dimv, 0, dimfr, dimv).isApprox(gain.Kfrq()));
  EXPECT_TRUE(K_ref.block(dimv, dimv, dimfr, dimv).isApprox(gain.Kfrv()));
  EXPECT_TRUE(K_ref.bottomLeftCorner(dimc, dimv).isApprox(gain.Kmuq()));
  EXPECT_TRUE(K_ref.bottomRightCorner(dimc, dimv).isApprox(gain.Kmuv()));
  const Eigen::VectorXd l_afr = Eigen::VectorXd::Random(dimafr);
  const Eigen::VectorXd C = Eigen::VectorXd::Random(dimc);
  gain.computeFeedforward(Ginv, l_afr, C);
  Eigen::VectorXd h = Eigen::VectorXd::Random(dimafr+dimc);
  h.head(dimafr) = l_afr;
  h.tail(dimc) = C;
  const Eigen::VectorXd k_ref = - Ginv * h;
  ASSERT_EQ(k_ref.size(), dimv);
  EXPECT_TRUE(k_ref.head(dimv).isApprox(gain.ka()));
  EXPECT_TRUE(k_ref.segment(dimv, dimfr).isApprox(gain.kfr()));
  EXPECT_TRUE(k_ref.tail(dimc).isApprox(gain.kmu()));
}


TEST_F(RiccatiGainTest, fixed_base_with_contacts) {
  std::vector<int> contact_frames = {18};
  fixed_base_robot_ = Robot(fixed_base_urdf_, contact_frames, 0, 0);
  const int dimv = fixed_base_robot_.dimv();
  const int dimafr = fixed_base_robot_.dimv() + 7*fixed_base_robot_.num_point_contacts();
  const int dimfr = 7*fixed_base_robot_.num_point_contacts();
  const int dimc = fixed_base_robot_.dim_passive();
  ASSERT_EQ(dimfr, 7);
  ASSERT_EQ(dimc, 0);
  const Eigen::MatrixXd Ginv_seed = Eigen::MatrixXd::Random(dimafr+dimc, dimafr+dimc);
  const Eigen::MatrixXd Ginv = Ginv_seed * Ginv_seed.transpose();
  const Eigen::MatrixXd Q_afr_qv = Eigen::MatrixXd::Random(dimafr, 2*dimv);
  const Eigen::MatrixXd C_qv = Eigen::MatrixXd::Random(dimc, 2*dimv);
  RiccatiGain gain(fixed_base_robot_);
  gain.computeFeedbackGain(Ginv, Q_afr_qv, C_qv);
  Eigen::MatrixXd HC = Eigen::MatrixXd::Random(dimafr+dimc, 2*dimv);
  HC.topRows(dimafr) = Q_afr_qv;
  HC.bottomRows(dimc) = C_qv;
  const Eigen::MatrixXd K_ref = - Ginv * HC;
  ASSERT_EQ(K_ref.rows(), dimafr);
  ASSERT_EQ(K_ref.cols(), 2*dimv);
  EXPECT_TRUE(K_ref.topLeftCorner(dimv, dimv).isApprox(gain.Kaq()));
  EXPECT_TRUE(K_ref.topRightCorner(dimv, dimv).isApprox(gain.Kav()));
  EXPECT_TRUE(K_ref.block(dimv, 0, dimfr, dimv).isApprox(gain.Kfrq()));
  EXPECT_TRUE(K_ref.block(dimv, dimv, dimfr, dimv).isApprox(gain.Kfrv()));
  EXPECT_TRUE(K_ref.bottomLeftCorner(dimc, dimv).isApprox(gain.Kmuq()));
  EXPECT_TRUE(K_ref.bottomRightCorner(dimc, dimv).isApprox(gain.Kmuv()));
  const Eigen::VectorXd l_afr = Eigen::VectorXd::Random(dimafr);
  const Eigen::VectorXd C = Eigen::VectorXd::Random(dimc);
  gain.computeFeedforward(Ginv, l_afr, C);
  Eigen::VectorXd h = Eigen::VectorXd::Random(dimafr+dimc);
  h.head(dimafr) = l_afr;
  h.tail(dimc) = C;
  const Eigen::VectorXd k_ref = - Ginv * h;
  ASSERT_EQ(k_ref.size(), dimafr);
  EXPECT_TRUE(k_ref.head(dimv).isApprox(gain.ka()));
  EXPECT_TRUE(k_ref.segment(dimv, dimfr).isApprox(gain.kfr()));
  EXPECT_TRUE(k_ref.tail(dimc).isApprox(gain.kmu()));
}


TEST_F(RiccatiGainTest, floating_base_without_contacts) {
  const int dimv = floating_base_robot_.dimv();
  const int dimafr = floating_base_robot_.dimv() + 7*floating_base_robot_.num_point_contacts();
  const int dimfr = 7*floating_base_robot_.num_point_contacts();
  const int dimc = floating_base_robot_.dim_passive();
  ASSERT_EQ(dimfr, 0);
  ASSERT_EQ(dimc, 6);
  const Eigen::MatrixXd Ginv_seed = Eigen::MatrixXd::Random(dimafr+dimc, dimafr+dimc);
  const Eigen::MatrixXd Ginv = Ginv_seed * Ginv_seed.transpose();
  const Eigen::MatrixXd Q_afr_qv = Eigen::MatrixXd::Random(dimafr, 2*dimv);
  const Eigen::MatrixXd C_qv = Eigen::MatrixXd::Random(dimc, 2*dimv);
  RiccatiGain gain(floating_base_robot_);
  gain.computeFeedbackGain(Ginv, Q_afr_qv, C_qv);
  Eigen::MatrixXd HC = Eigen::MatrixXd::Random(dimafr+dimc, 2*dimv);
  HC.topRows(dimafr) = Q_afr_qv;
  HC.bottomRows(dimc) = C_qv;
  const Eigen::MatrixXd K_ref = - Ginv * HC;
  ASSERT_EQ(K_ref.rows(), dimv+dimc);
  ASSERT_EQ(K_ref.cols(), 2*dimv);
  EXPECT_TRUE(K_ref.topLeftCorner(dimv, dimv).isApprox(gain.Kaq()));
  EXPECT_TRUE(K_ref.topRightCorner(dimv, dimv).isApprox(gain.Kav()));
  EXPECT_TRUE(K_ref.block(dimv, 0, dimfr, dimv).isApprox(gain.Kfrq()));
  EXPECT_TRUE(K_ref.block(dimv, dimv, dimfr, dimv).isApprox(gain.Kfrv()));
  EXPECT_TRUE(K_ref.bottomLeftCorner(dimc, dimv).isApprox(gain.Kmuq()));
  EXPECT_TRUE(K_ref.bottomRightCorner(dimc, dimv).isApprox(gain.Kmuv()));
  const Eigen::VectorXd l_afr = Eigen::VectorXd::Random(dimafr);
  const Eigen::VectorXd C = Eigen::VectorXd::Random(dimc);
  gain.computeFeedforward(Ginv, l_afr, C);
  Eigen::VectorXd h = Eigen::VectorXd::Random(dimafr+dimc);
  h.head(dimafr) = l_afr;
  h.tail(dimc) = C;
  const Eigen::VectorXd k_ref = - Ginv * h;
  ASSERT_EQ(k_ref.size(), dimv+dimc);
  EXPECT_TRUE(k_ref.head(dimv).isApprox(gain.ka()));
  EXPECT_TRUE(k_ref.segment(dimv, dimfr).isApprox(gain.kfr()));
  EXPECT_TRUE(k_ref.tail(dimc).isApprox(gain.kmu()));
}


TEST_F(RiccatiGainTest, floating_base_with_contacts) {
  const std::vector<int> contact_frames = {14, 24, 34, 44};
  floating_base_robot_ = Robot(floating_base_urdf_, contact_frames, 0, 0);
  const int dimv = floating_base_robot_.dimv();
  const int dimafr = floating_base_robot_.dimv() + 7*floating_base_robot_.num_point_contacts();
  const int dimfr = 7*floating_base_robot_.num_point_contacts();
  const int dimc = floating_base_robot_.dim_passive();
  ASSERT_EQ(dimfr, 28);
  ASSERT_EQ(dimc, 6);
  const Eigen::MatrixXd Ginv_seed = Eigen::MatrixXd::Random(dimafr+dimc, dimafr+dimc);
  const Eigen::MatrixXd Ginv = Ginv_seed * Ginv_seed.transpose();
  const Eigen::MatrixXd Q_afr_qv = Eigen::MatrixXd::Random(dimafr, 2*dimv);
  const Eigen::MatrixXd C_qv = Eigen::MatrixXd::Random(dimc, 2*dimv);
  RiccatiGain gain(floating_base_robot_);
  gain.computeFeedbackGain(Ginv, Q_afr_qv, C_qv);
  Eigen::MatrixXd HC = Eigen::MatrixXd::Random(dimafr+dimc, 2*dimv);
  HC.topRows(dimafr) = Q_afr_qv;
  HC.bottomRows(dimc) = C_qv;
  const Eigen::MatrixXd K_ref = - Ginv * HC;
  ASSERT_EQ(K_ref.rows(), dimafr+dimc);
  ASSERT_EQ(K_ref.cols(), 2*dimv);
  EXPECT_TRUE(K_ref.topLeftCorner(dimv, dimv).isApprox(gain.Kaq()));
  EXPECT_TRUE(K_ref.topRightCorner(dimv, dimv).isApprox(gain.Kav()));
  EXPECT_TRUE(K_ref.block(dimv, 0, dimfr, dimv).isApprox(gain.Kfrq()));
  EXPECT_TRUE(K_ref.block(dimv, dimv, dimfr, dimv).isApprox(gain.Kfrv()));
  EXPECT_TRUE(K_ref.bottomLeftCorner(dimc, dimv).isApprox(gain.Kmuq()));
  EXPECT_TRUE(K_ref.bottomRightCorner(dimc, dimv).isApprox(gain.Kmuv()));
  const Eigen::VectorXd l_afr = Eigen::VectorXd::Random(dimafr);
  const Eigen::VectorXd C = Eigen::VectorXd::Random(dimc);
  gain.computeFeedforward(Ginv, l_afr, C);
  Eigen::VectorXd h = Eigen::VectorXd::Random(dimafr+dimc);
  h.head(dimafr) = l_afr;
  h.tail(dimc) = C;
  const Eigen::VectorXd k_ref = - Ginv * h;
  ASSERT_EQ(k_ref.size(), dimafr+dimc);
  EXPECT_TRUE(k_ref.head(dimv).isApprox(gain.ka()));
  EXPECT_TRUE(k_ref.segment(dimv, dimfr).isApprox(gain.kfr()));
  EXPECT_TRUE(k_ref.tail(dimc).isApprox(gain.kmu()));
}


} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}