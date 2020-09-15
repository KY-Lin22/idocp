#include <string>
#include <memory>

#include <gtest/gtest.h>
#include "Eigen/Core"
#include "Eigen/LU"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/riccati_matrix_inverter.hpp"
#include "idocp/ocp/riccati_gain.hpp"


namespace idocp {

class RiccatiTest : public ::testing::Test {
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


TEST_F(RiccatiTest, fixed_base_without_contacts) {
  const int dimv = fixed_base_robot_.dimv();
  const int dimafr = fixed_base_robot_.dimv() + 7*fixed_base_robot_.num_point_contacts();
  const int dimfr = 7*fixed_base_robot_.num_point_contacts();
  const int dimc = fixed_base_robot_.dim_passive();
  ASSERT_EQ(dimfr, 0);
  ASSERT_EQ(dimc, 0);
  Eigen::MatrixXd Qqa = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::MatrixXd Qva = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::MatrixXd Qaa = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::VectorXd la = Eigen::VectorXd::Random(dimv);
  Eigen::MatrixXd Kaq = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Kav = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::VectorXd ka = Eigen::VectorXd::Zero(dimv);
  Qaa.triangularView<Eigen::StrictlyLower>() 
      = Qaa.transpose().triangularView<Eigen::StrictlyLower>();
  Qaa = Qaa * Qaa.transpose(); // Makes Qaa semi positive define
  Qaa.noalias() += Eigen::MatrixXd::Identity(dimv, dimv); // Makes Qaa sufficiently positive define
  RiccatiMatrixInverter inverter(fixed_base_robot_);
  Eigen::MatrixXd G = Qaa;
  Eigen::MatrixXd C_afr = Eigen::MatrixXd::Zero(dimc, dimafr);
  Eigen::MatrixXd Ginv = Eigen::MatrixXd::Zero(dimafr+dimc, dimafr+dimc);
  inverter.invert(G, C_afr, Ginv);
  Eigen::MatrixXd Q_afr_qv = Eigen::MatrixXd::Zero(dimafr, 2*dimv);
  Q_afr_qv.leftCols(dimv) = Qqa.transpose();
  Q_afr_qv.rightCols(dimv) = Qva.transpose();
  Eigen::MatrixXd C_qv = Eigen::MatrixXd::Zero(dimc, 2*dimv);
  RiccatiGain gain(fixed_base_robot_);
  gain.computeFeedbackGain(Ginv, Q_afr_qv, C_qv);
  Eigen::VectorXd l_afr = Eigen::VectorXd::Zero(dimafr);
  l_afr.head(dimafr) = l_afr;
  Eigen::VectorXd C = Eigen::VectorXd::Zero(dimc);
  gain.computeFeedforward(Ginv, l_afr, C);
  const Eigen::MatrixXd Qaa_inv = Qaa.inverse();
  const Eigen::MatrixXd Kaq_ref = - Qaa_inv * Qqa.transpose();
  const Eigen::MatrixXd Kav_ref = - Qaa_inv * Qva.transpose();
  const Eigen::VectorXd ka_ref = - Qaa_inv * l_afr;
  EXPECT_TRUE(gain.Kaq().isApprox(Kaq_ref));
  EXPECT_TRUE(gain.Kav().isApprox(Kav_ref));
  EXPECT_TRUE(gain.ka().isApprox(ka_ref));
  std::cout << "Kaq error:" << std::endl;
  std::cout << gain.Kaq() - Kaq_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kav error:" << std::endl;
  std::cout << gain.Kav() - Kav_ref << std::endl;
  std::cout << std::endl;
  std::cout << "ka error:" << std::endl;
  std::cout << gain.ka() - ka_ref << std::endl;
  std::cout << std::endl;
}


TEST_F(RiccatiTest, fixed_base_with_contacts) {
  std::vector<int> contact_frames = {18};
  fixed_base_robot_ = Robot(fixed_base_urdf_, contact_frames, 0, 0);
  const int dimv = fixed_base_robot_.dimv();
  const int dimafr = fixed_base_robot_.dimv() + 7*fixed_base_robot_.num_point_contacts();
  const int dimfr = 7*fixed_base_robot_.num_point_contacts();
  const int dimc = fixed_base_robot_.dim_passive();
  ASSERT_EQ(dimfr, 7);
  ASSERT_EQ(dimc, 0);
  Eigen::MatrixXd gen_mat = Eigen::MatrixXd::Random(dimafr, dimafr);
  gen_mat.triangularView<Eigen::StrictlyLower>() 
      = gen_mat.transpose().triangularView<Eigen::StrictlyLower>();
  Eigen::MatrixXd pos_mat = gen_mat * gen_mat.transpose(); // Makes pos_mat semi positive define
  pos_mat.noalias() += Eigen::MatrixXd::Identity(dimafr, dimafr); // Makes pos_mat sufficiently positive define
  Eigen::MatrixXd Qqa = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::MatrixXd Qva = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::MatrixXd Qaa = pos_mat.block(0, 0, dimv, dimv);
  Eigen::MatrixXd Qqfr = Eigen::MatrixXd::Random(dimv, dimfr);
  Eigen::MatrixXd Qvfr = Eigen::MatrixXd::Random(dimv, dimfr);
  Eigen::MatrixXd Qafr = pos_mat.block(0, dimv, dimv, dimfr);
  Eigen::MatrixXd Qfrfr = pos_mat.block(dimv, dimv, dimfr, dimfr);
  Eigen::VectorXd la = Eigen::VectorXd::Random(dimv);
  Eigen::VectorXd lfr = Eigen::VectorXd::Random(dimfr);
  Eigen::MatrixXd Kaq = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Kav = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Kfrq = Eigen::MatrixXd::Zero(dimfr, dimv);
  Eigen::MatrixXd Kfrv = Eigen::MatrixXd::Zero(dimfr, dimv);
  Eigen::VectorXd ka = Eigen::VectorXd::Zero(dimv);
  Eigen::VectorXd kfr = Eigen::VectorXd::Zero(dimfr);
  Eigen::MatrixXd M = pos_mat;
  M.triangularView<Eigen::StrictlyLower>() 
      = M.transpose().triangularView<Eigen::StrictlyLower>();
  const Eigen::MatrixXd Minv = M.inverse();
  const Eigen::MatrixXd Kaq_ref = - Minv.block(0, 0, dimv, dimv) * Qqa.transpose()
                                  - Minv.block(0, dimv, dimv, dimfr) * Qqfr.transpose();
  const Eigen::MatrixXd Kav_ref = - Minv.block(0, 0, dimv, dimv) * Qva.transpose()
                                  - Minv.block(0, dimv, dimv, dimfr) * Qvfr.transpose();
  const Eigen::MatrixXd Kfrq_ref = - Minv.block(dimv, 0, dimfr, dimv) * Qqa.transpose()
                                   - Minv.block(dimv, dimv, dimfr, dimfr) * Qqfr.transpose();
  const Eigen::MatrixXd Kfrv_ref = - Minv.block(dimv, 0, dimfr, dimv) * Qva.transpose()
                                   - Minv.block(dimv, dimv, dimfr, dimfr) * Qvfr.transpose();
  const Eigen::VectorXd ka_ref = - Minv.block(0, 0, dimv, dimv) * la
                                 - Minv.block(0, dimv, dimv, dimfr) * lfr;
  const Eigen::VectorXd kfr_ref = - Minv.block(dimv, 0, dimfr, dimv) * la
                                  - Minv.block(dimv, dimv, dimfr, dimfr) * lfr;
  RiccatiMatrixInverter inverter(fixed_base_robot_);
  Eigen::MatrixXd G = M;
  Eigen::MatrixXd Ginv = Eigen::MatrixXd::Zero(dimafr, dimafr);
  Eigen::MatrixXd C_afr = Eigen::MatrixXd::Zero(dimc, dimafr);
  inverter.invert(G, C_afr, Ginv);
  Eigen::MatrixXd Q_afr_qv = Eigen::MatrixXd::Zero(dimafr, 2*dimv);
  Q_afr_qv.topLeftCorner(dimv, dimv) = Qqa.transpose();
  Q_afr_qv.topRightCorner(dimv, dimv) = Qva.transpose();
  Q_afr_qv.bottomLeftCorner(dimfr, dimv) = Qqfr.transpose();
  Q_afr_qv.bottomRightCorner(dimfr, dimv) = Qvfr.transpose();
  Eigen::MatrixXd C_qv = Eigen::MatrixXd::Zero(dimc, 2*dimv);
  RiccatiGain gain(fixed_base_robot_);
  gain.computeFeedbackGain(Ginv, Q_afr_qv, C_qv);
  Eigen::VectorXd l_afr = Eigen::VectorXd::Zero(dimafr);
  l_afr.head(dimv) = la;
  l_afr.tail(dimfr) = lfr;
  Eigen::VectorXd C = Eigen::VectorXd::Zero(dimc);
  gain.computeFeedforward(Ginv, l_afr, C);
  EXPECT_TRUE(gain.Kaq().isApprox(Kaq_ref));
  EXPECT_TRUE(gain.Kav().isApprox(Kav_ref));
  EXPECT_TRUE(gain.Kfrq().isApprox(Kfrq_ref));
  EXPECT_TRUE(gain.Kfrv().isApprox(Kfrv_ref));
  EXPECT_TRUE(gain.ka().isApprox(ka_ref));
  EXPECT_TRUE(gain.kfr().isApprox(kfr_ref));
  std::cout << "Kaq error:" << std::endl;
  std::cout << gain.Kaq() - Kaq_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kav error:" << std::endl;
  std::cout << gain.Kav() - Kav_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kfrq error:" << std::endl;
  std::cout << gain.Kfrq() - Kfrq_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kfrv error:" << std::endl;
  std::cout << gain.Kfrv() - Kfrv_ref << std::endl;
  std::cout << std::endl;
  std::cout << "ka error:" << std::endl;
  std::cout << gain.ka() - ka_ref << std::endl;
  std::cout << std::endl;
  std::cout << "kfr error:" << std::endl;
  std::cout << gain.kfr() - kfr_ref << std::endl;
  std::cout << std::endl;
}


TEST_F(RiccatiTest, floating_base_without_contacts) {
  const int dimv = floating_base_robot_.dimv();
  const int dimafr = floating_base_robot_.dimv() + 7*floating_base_robot_.num_point_contacts();
  const int dimfr = 7*floating_base_robot_.num_point_contacts();
  const int dimc = floating_base_robot_.dim_passive();
  ASSERT_EQ(dimfr, 0);
  ASSERT_EQ(dimc, 6);
  Eigen::MatrixXd gen_mat = Eigen::MatrixXd::Random(dimv, dimv);
  gen_mat.triangularView<Eigen::StrictlyLower>() 
      = gen_mat.transpose().triangularView<Eigen::StrictlyLower>();
  Eigen::MatrixXd pos_mat = gen_mat * gen_mat.transpose(); // Makes pos_mat semi positive define
  pos_mat.noalias() += Eigen::MatrixXd::Identity(dimv, dimv); // Makes pos_mat sufficiently positive define
  Eigen::MatrixXd Qqa = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::MatrixXd Qva = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::MatrixXd Qaa = pos_mat.block(0, 0, dimv, dimv);
  Eigen::MatrixXd Cq = Eigen::MatrixXd::Random(dimc, dimv);
  Eigen::MatrixXd Cv = Eigen::MatrixXd::Random(dimc, dimv);
  Eigen::MatrixXd Ca = Eigen::MatrixXd::Random(dimc, dimv);
  Eigen::VectorXd la = Eigen::VectorXd::Random(dimv);
  Eigen::VectorXd C_res = Eigen::VectorXd::Random(dimc);
  Eigen::MatrixXd Kaq = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Kav = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Kmuq = Eigen::MatrixXd::Zero(dimc, dimv);
  Eigen::MatrixXd Kmuv = Eigen::MatrixXd::Zero(dimc, dimv);
  Eigen::VectorXd ka = Eigen::VectorXd::Zero(dimv);
  Eigen::VectorXd kmu = Eigen::VectorXd::Zero(dimc);
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dimv+dimc, dimv+dimc);
  M.block(0, 0, dimv, dimv) = pos_mat;
  M.block(0, dimv, dimv, dimc) = Ca.transpose();
  M.block(dimv, 0, dimc, dimv) = Ca;
  M.triangularView<Eigen::StrictlyLower>() 
      = M.transpose().triangularView<Eigen::StrictlyLower>();
  const Eigen::MatrixXd Minv = M.inverse();
  const Eigen::MatrixXd Kaq_ref = - Minv.block(0, 0, dimv, dimv) * Qqa.transpose()
                                  - Minv.block(0, dimv, dimv, dimc) * Cq;
  const Eigen::MatrixXd Kav_ref = - Minv.block(0, 0, dimv, dimv) * Qva.transpose()
                                  - Minv.block(0, dimv, dimv, dimc) * Cv;
  const Eigen::MatrixXd Kmuq_ref = - Minv.block(dimv, 0, dimc, dimv) * Qqa.transpose()
                                   - Minv.block(dimv, dimv, dimc, dimc) * Cq;
  const Eigen::MatrixXd Kmuv_ref = - Minv.block(dimv, 0, dimc, dimv) * Qva.transpose()
                                   - Minv.block(dimv, dimv, dimc, dimc) * Cv;
  const Eigen::VectorXd ka_ref = - Minv.block(0, 0, dimv, dimv) * la
                                 - Minv.block(0, dimv, dimv, dimc) * C_res;
  const Eigen::VectorXd kmu_ref = - Minv.block(dimv, 0, dimc, dimv) * la
                                  - Minv.block(dimv, dimv, dimc, dimc) * C_res;
  RiccatiMatrixInverter inverter(floating_base_robot_);
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(dimv, dimv);
  G = M.topLeftCorner(dimv, dimv);
  Eigen::MatrixXd C_afr = Eigen::MatrixXd::Zero(dimc, dimv);
  C_afr = Ca;
  Eigen::MatrixXd Ginv = Eigen::MatrixXd::Zero(dimv+dimc, dimv+dimc);
  inverter.invert(G, C_afr, Ginv);
  Eigen::MatrixXd Q_afr_qv = Eigen::MatrixXd::Zero(dimv, 2*dimv);
  Q_afr_qv.topLeftCorner(dimv, dimv) = Qqa.transpose();
  Q_afr_qv.topRightCorner(dimv, dimv) = Qva.transpose();
  Eigen::MatrixXd C_qv = Eigen::MatrixXd::Zero(dimc, 2*dimv);
  C_qv.leftCols(dimv) = Cq;
  C_qv.rightCols(dimv) = Cv;
  RiccatiGain gain(floating_base_robot_);
  gain.computeFeedbackGain(Ginv, Q_afr_qv, C_qv);
  Eigen::VectorXd l_afr = Eigen::VectorXd::Zero(dimv);
  l_afr.head(dimv) = la;
  Eigen::VectorXd C = C_res;
  gain.computeFeedforward(Ginv, l_afr, C);
  EXPECT_TRUE(gain.Kaq().isApprox(Kaq_ref));
  EXPECT_TRUE(gain.Kav().isApprox(Kav_ref));
  EXPECT_TRUE(gain.Kmuq().isApprox(Kmuq_ref));
  EXPECT_TRUE(gain.Kmuv().isApprox(Kmuv_ref));
  EXPECT_TRUE(gain.ka().isApprox(ka_ref));
  EXPECT_TRUE(gain.kmu().isApprox(kmu_ref));
  std::cout << "Kaq error:" << std::endl;
  std::cout << gain.Kaq() - Kaq_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kav error:" << std::endl;
  std::cout << gain.Kav() - Kav_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kmuq error:" << std::endl;
  std::cout << gain.Kmuq() - Kmuq_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kmuv error:" << std::endl;
  std::cout << gain.Kmuv() - Kmuv_ref << std::endl;
  std::cout << std::endl;
  std::cout << "ka error:" << std::endl;
  std::cout << gain.ka() - ka_ref << std::endl;
  std::cout << std::endl;
  std::cout << "kmu error:" << std::endl;
  std::cout << gain.kmu() - kmu_ref << std::endl;
  std::cout << std::endl;
}


TEST_F(RiccatiTest, floating_base_with_contacts) {
  const std::vector<int> contact_frames = {14, 24, 34, 44};
  floating_base_robot_ = Robot(floating_base_urdf_, contact_frames, 0, 0);
  const int dimv = floating_base_robot_.dimv();
  const int dimafr = floating_base_robot_.dimv() + 7*floating_base_robot_.num_point_contacts();
  const int dimfr = 7*floating_base_robot_.num_point_contacts();
  const int dimc = floating_base_robot_.dim_passive();
  ASSERT_EQ(dimfr, 28);
  ASSERT_EQ(dimc, 6);
  Eigen::MatrixXd gen_mat = Eigen::MatrixXd::Random(dimafr, dimafr);
  gen_mat.triangularView<Eigen::StrictlyLower>() 
      = gen_mat.transpose().triangularView<Eigen::StrictlyLower>();
  Eigen::MatrixXd pos_mat = gen_mat * gen_mat.transpose(); // Makes pos_mat semi positive define
  pos_mat.noalias() += Eigen::MatrixXd::Identity(dimafr, dimafr); // Makes pos_mat sufficiently positive define
  Eigen::MatrixXd Qqa = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::MatrixXd Qva = Eigen::MatrixXd::Random(dimv, dimv);
  Eigen::MatrixXd Qaa = pos_mat.block(0, 0, dimv, dimv);
  Eigen::MatrixXd Qqfr = Eigen::MatrixXd::Random(dimv, dimfr);
  Eigen::MatrixXd Qvfr = Eigen::MatrixXd::Random(dimv, dimfr);
  Eigen::MatrixXd Qafr = pos_mat.block(0, dimv, dimv, dimfr);
  Eigen::MatrixXd Qfrfr = pos_mat.block(dimv, dimv, dimfr, dimfr);
  Eigen::MatrixXd Cq = Eigen::MatrixXd::Random(dimc, dimv);
  Eigen::MatrixXd Cv = Eigen::MatrixXd::Random(dimc, dimv);
  Eigen::MatrixXd Ca = Eigen::MatrixXd::Random(dimc, dimv);
  Eigen::MatrixXd Cfr = Eigen::MatrixXd::Random(dimc, dimfr);
  Eigen::VectorXd la = Eigen::VectorXd::Random(dimv);
  Eigen::VectorXd lfr = Eigen::VectorXd::Random(dimfr);
  Eigen::VectorXd C_res = Eigen::VectorXd::Random(dimc);
  Eigen::MatrixXd Kaq = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Kav = Eigen::MatrixXd::Zero(dimv, dimv);
  Eigen::MatrixXd Kfrq = Eigen::MatrixXd::Zero(dimfr, dimv);
  Eigen::MatrixXd Kfrv = Eigen::MatrixXd::Zero(dimfr, dimv);
  Eigen::MatrixXd Kmuq = Eigen::MatrixXd::Zero(dimc, dimv);
  Eigen::MatrixXd Kmuv = Eigen::MatrixXd::Zero(dimc, dimv);
  Eigen::VectorXd ka = Eigen::VectorXd::Zero(dimv);
  Eigen::VectorXd kfr = Eigen::VectorXd::Zero(dimfr);
  Eigen::VectorXd kmu = Eigen::VectorXd::Zero(dimc);
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dimafr+dimc, dimafr+dimc);
  M.block(0, 0, dimafr, dimafr) = pos_mat.topLeftCorner(dimafr, dimafr);
  M.block(0, dimafr, dimv, dimc) = Ca.transpose();
  M.block(dimv, dimafr, dimfr, dimc) = Cfr.transpose();
  M.triangularView<Eigen::StrictlyLower>() 
      = M.transpose().triangularView<Eigen::StrictlyLower>();
  const Eigen::MatrixXd Minv = M.inverse();
  const Eigen::MatrixXd Kaq_ref = - Minv.block(0, 0, dimv, dimv) * Qqa.transpose()
                                  - Minv.block(0, dimv, dimv, dimfr) * Qqfr.transpose()
                                  - Minv.block(0, dimv+dimfr, dimv, dimc) * Cq;
  const Eigen::MatrixXd Kav_ref = - Minv.block(0, 0, dimv, dimv) * Qva.transpose()
                                  - Minv.block(0, dimv, dimv, dimfr) * Qvfr.transpose()
                                  - Minv.block(0, dimv+dimfr, dimv, dimc) * Cv;
  const Eigen::MatrixXd Kfrq_ref = - Minv.block(dimv, 0, dimfr, dimv) * Qqa.transpose()
                                   - Minv.block(dimv, dimv, dimfr, dimfr) * Qqfr.transpose()
                                   - Minv.block(dimv, dimv+dimfr, dimfr, dimc) * Cq;
  const Eigen::MatrixXd Kfrv_ref = - Minv.block(dimv, 0, dimfr, dimv) * Qva.transpose()
                                   - Minv.block(dimv, dimv, dimfr, dimfr) * Qvfr.transpose()
                                   - Minv.block(dimv, dimv+dimfr, dimfr, dimc) * Cv;
  const Eigen::MatrixXd Kmuq_ref = - Minv.block(dimv+dimfr, 0, dimc, dimv) * Qqa.transpose()
                                   - Minv.block(dimv+dimfr, dimv, dimc, dimfr) * Qqfr.transpose()
                                   - Minv.block(dimv+dimfr, dimv+dimfr, dimc, dimc) * Cq;
  const Eigen::MatrixXd Kmuv_ref = - Minv.block(dimv+dimfr, 0, dimc, dimv) * Qva.transpose()
                                   - Minv.block(dimv+dimfr, dimv, dimc, dimfr) * Qvfr.transpose()
                                   - Minv.block(dimv+dimfr, dimv+dimfr, dimc, dimc) * Cv;
  const Eigen::VectorXd ka_ref = - Minv.block(0, 0, dimv, dimv) * la
                                 - Minv.block(0, dimv, dimv, dimfr) * lfr
                                 - Minv.block(0, dimv+dimfr, dimv, dimc) * C_res;
  const Eigen::VectorXd kfr_ref = - Minv.block(dimv, 0, dimfr, dimv) * la
                                  - Minv.block(dimv, dimv, dimfr, dimfr) * lfr
                                  - Minv.block(dimv, dimv+dimfr, dimfr, dimc) * C_res;
  const Eigen::VectorXd kmu_ref = - Minv.block(dimv+dimfr, 0, dimc, dimv) * la
                                  - Minv.block(dimv+dimfr, dimv, dimc, dimfr) * lfr
                                  - Minv.block(dimv+dimfr, dimv+dimfr, dimc, dimc) * C_res;
  RiccatiMatrixInverter inverter(floating_base_robot_);
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(dimafr, dimafr);
  G = M.topLeftCorner(dimafr, dimafr);
  Eigen::MatrixXd C_afr = Eigen::MatrixXd::Zero(dimc, dimafr);
  C_afr.leftCols(dimv) = Ca;
  C_afr.rightCols(dimfr) = Cfr;
  Eigen::MatrixXd Ginv = Eigen::MatrixXd::Zero(dimafr+dimc, dimafr+dimc);
  inverter.invert(G, C_afr, Ginv);
  Eigen::MatrixXd Q_afr_qv = Eigen::MatrixXd::Zero(dimafr, 2*dimv);
  Q_afr_qv.topLeftCorner(dimv, dimv) = Qqa.transpose();
  Q_afr_qv.topRightCorner(dimv, dimv) = Qva.transpose();
  Q_afr_qv.bottomLeftCorner(dimfr, dimv) = Qqfr.transpose();
  Q_afr_qv.bottomRightCorner(dimfr, dimv) = Qvfr.transpose();
  Eigen::MatrixXd C_qv = Eigen::MatrixXd::Zero(dimc, 2*dimv);
  C_qv.leftCols(dimv) = Cq;
  C_qv.rightCols(dimv) = Cv;
  RiccatiGain gain(floating_base_robot_);
  gain.computeFeedbackGain(Ginv, Q_afr_qv, C_qv);
  Eigen::VectorXd l_afr = Eigen::VectorXd::Zero(dimafr);
  l_afr.head(dimv) = la;
  l_afr.tail(dimfr) = lfr;
  Eigen::VectorXd C = C_res.head(dimc);
  gain.computeFeedforward(Ginv, l_afr, C);
  EXPECT_TRUE(gain.Kaq().isApprox(Kaq_ref));
  EXPECT_TRUE(gain.Kav().isApprox(Kav_ref));
  EXPECT_TRUE(gain.Kfrq().isApprox(Kfrq_ref));
  EXPECT_TRUE(gain.Kfrv().isApprox(Kfrv_ref));
  EXPECT_TRUE(gain.Kmuq().isApprox(Kmuq_ref));
  EXPECT_TRUE(gain.Kmuv().isApprox(Kmuv_ref));
  EXPECT_TRUE(gain.ka().isApprox(ka_ref));
  EXPECT_TRUE(gain.kfr().isApprox(kfr_ref));
  EXPECT_TRUE(gain.kmu().isApprox(kmu_ref));
  std::cout << "Kaq error:" << std::endl;
  std::cout << gain.Kaq() - Kaq_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kav error:" << std::endl;
  std::cout << gain.Kav() - Kav_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kfrq error:" << std::endl;
  std::cout << gain.Kfrq() - Kfrq_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kfrv error:" << std::endl;
  std::cout << gain.Kfrv() - Kfrv_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kmuq error:" << std::endl;
  std::cout << gain.Kmuq() - Kmuq_ref << std::endl;
  std::cout << std::endl;
  std::cout << "Kmuv error:" << std::endl;
  std::cout << gain.Kmuv() - Kmuv_ref << std::endl;
  std::cout << std::endl;
  std::cout << "ka error:" << std::endl;
  std::cout << gain.ka() - ka_ref << std::endl;
  std::cout << std::endl;
  std::cout << "kfr error:" << std::endl;
  std::cout << gain.kfr() - kfr_ref << std::endl;
  std::cout << std::endl;
  std::cout << "kmu error:" << std::endl;
  std::cout << gain.kmu() - kmu_ref << std::endl;
  std::cout << std::endl;
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}