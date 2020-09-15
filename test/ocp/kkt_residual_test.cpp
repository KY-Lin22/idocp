#include <string>

#include <gtest/gtest.h>

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/kkt_residual.hpp"


namespace idocp {

class KKTResidualTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    fixed_base_urdf_ = "../urdf/iiwa14/iiwa14.urdf";
    floating_base_urdf_ = "../urdf/anymal/anymal.urdf";
  }

  virtual void TearDown() {
  }

  double dtau_;
  std::string fixed_base_urdf_, floating_base_urdf_;
};


TEST_F(KKTResidualTest, fixed_base) {
  Robot robot(fixed_base_urdf_);
  KKTResidual residual(robot);
  const int dimv = robot.dimv();
  const int dimf = 5*robot.num_point_contacts();
  const int dimr = 2*robot.num_point_contacts();
  const int dimfr = dimf + dimr;
  const int dimc = robot.dim_passive();
  EXPECT_EQ(residual.dimKKT(), 5*dimv);
  EXPECT_EQ(residual.dimc(), 0);
  EXPECT_EQ(residual.dimf(), 0);
  EXPECT_EQ(residual.dimr(), 0);
  const Eigen::VectorXd Fq_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd Fv_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd C_res = Eigen::VectorXd::Random(dimc);
  const Eigen::VectorXd la_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd lf_res = Eigen::VectorXd::Random(dimf);
  const Eigen::VectorXd lr_res = Eigen::VectorXd::Random(dimr);
  const Eigen::VectorXd lq_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd lv_res = Eigen::VectorXd::Random(dimv);
  residual.Fq() = Fq_res;
  residual.Fv() = Fv_res;
  residual.C() = C_res;
  residual.la() = la_res;
  residual.lf() = lf_res;
  residual.lr() = lr_res;
  residual.lq() = lq_res;
  residual.lv() = lv_res;
  EXPECT_TRUE(residual.KKT_residual.segment(                0, dimv).isApprox(Fq_res));
  EXPECT_TRUE(residual.KKT_residual.segment(             dimv, dimv).isApprox(Fv_res));
  EXPECT_TRUE(residual.KKT_residual.segment(           2*dimv, dimc).isApprox(C_res));
  EXPECT_TRUE(residual.KKT_residual.segment(      2*dimv+dimc, dimv).isApprox(la_res));
  EXPECT_TRUE(residual.KKT_residual.segment(      3*dimv+dimc, dimf).isApprox(lf_res));
  EXPECT_TRUE(residual.KKT_residual.segment( 3*dimv+dimc+dimf, dimr).isApprox(lr_res));
  EXPECT_TRUE(residual.KKT_residual.segment(3*dimv+dimc+dimfr, dimv).isApprox(lq_res));
  EXPECT_TRUE(residual.KKT_residual.segment(4*dimv+dimc+dimfr, dimv).isApprox(lv_res));
  EXPECT_TRUE(residual.lx().head(dimv).isApprox(lq_res));
  EXPECT_TRUE(residual.lx().tail(dimv).isApprox(lv_res));
  EXPECT_EQ(residual.lx().size(), 2*dimv);
  EXPECT_TRUE(residual.l_afr().segment(        0, dimv).isApprox(la_res));
  EXPECT_TRUE(residual.l_afr().segment(     dimv, dimf).isApprox(lf_res));
  EXPECT_TRUE(residual.l_afr().segment(dimv+dimf, dimr).isApprox(lr_res));
  EXPECT_EQ(residual.l_afr().size(), dimv+dimf+dimr);
  residual.setZero();
  EXPECT_TRUE(residual.KKT_residual.isZero());
}


TEST_F(KKTResidualTest, fixed_base_contact) {
  std::vector<int> contact_frames = {18};
  Robot robot(fixed_base_urdf_, contact_frames, 0, 0);
  KKTResidual residual(robot);
  const int dimv = robot.dimv();
  const int dimf = 5*robot.num_point_contacts();
  const int dimr = 2*robot.num_point_contacts();
  const int dimfr = dimf + dimr;
  const int dimc = robot.dim_passive();
  EXPECT_EQ(residual.dimKKT(), 5*dimv+dimfr);
  EXPECT_EQ(residual.dimc(), 0);
  EXPECT_EQ(residual.dimf(), 5);
  EXPECT_EQ(residual.dimr(), 2);
  const Eigen::VectorXd Fq_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd Fv_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd C_res = Eigen::VectorXd::Random(dimc);
  const Eigen::VectorXd la_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd lf_res = Eigen::VectorXd::Random(dimf);
  const Eigen::VectorXd lr_res = Eigen::VectorXd::Random(dimr);
  const Eigen::VectorXd lq_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd lv_res = Eigen::VectorXd::Random(dimv);
  residual.Fq() = Fq_res;
  residual.Fv() = Fv_res;
  residual.C() = C_res;
  residual.la() = la_res;
  residual.lf() = lf_res;
  residual.lr() = lr_res;
  residual.lq() = lq_res;
  residual.lv() = lv_res;
  EXPECT_TRUE(residual.KKT_residual.segment(                0, dimv).isApprox(Fq_res));
  EXPECT_TRUE(residual.KKT_residual.segment(             dimv, dimv).isApprox(Fv_res));
  EXPECT_TRUE(residual.KKT_residual.segment(           2*dimv, dimc).isApprox(C_res));
  EXPECT_TRUE(residual.KKT_residual.segment(      2*dimv+dimc, dimv).isApprox(la_res));
  EXPECT_TRUE(residual.KKT_residual.segment(      3*dimv+dimc, dimf).isApprox(lf_res));
  EXPECT_TRUE(residual.KKT_residual.segment( 3*dimv+dimc+dimf, dimr).isApprox(lr_res));
  EXPECT_TRUE(residual.KKT_residual.segment(3*dimv+dimc+dimfr, dimv).isApprox(lq_res));
  EXPECT_TRUE(residual.KKT_residual.segment(4*dimv+dimc+dimfr, dimv).isApprox(lv_res));
  EXPECT_TRUE(residual.lx().head(dimv).isApprox(lq_res));
  EXPECT_TRUE(residual.lx().tail(dimv).isApprox(lv_res));
  EXPECT_EQ(residual.lx().size(), 2*dimv);
  EXPECT_TRUE(residual.l_afr().segment(        0, dimv).isApprox(la_res));
  EXPECT_TRUE(residual.l_afr().segment(     dimv, dimf).isApprox(lf_res));
  EXPECT_TRUE(residual.l_afr().segment(dimv+dimf, dimr).isApprox(lr_res));
  EXPECT_EQ(residual.l_afr().size(), dimv+dimf+dimr);
  residual.setZero();
  EXPECT_TRUE(residual.KKT_residual.isZero());
}


TEST_F(KKTResidualTest, floating_base) {
  std::vector<int> contact_frames = {14, 24, 34, 44};
  Robot robot(floating_base_urdf_, contact_frames, 0, 0);
  std::random_device rnd;
  KKTResidual residual(robot);
  const int dimv = robot.dimv();
  const int dimf = 5*robot.num_point_contacts();
  const int dimr = 2*robot.num_point_contacts();
  const int dimfr = dimf + dimr;
  const int dimc = robot.dim_passive();
  EXPECT_EQ(residual.dimKKT(), 5*dimv+dimc+dimfr);
  EXPECT_EQ(residual.dimc(), 6);
  EXPECT_EQ(residual.dimf(), 4*5);
  EXPECT_EQ(residual.dimr(), 4*2);
  const Eigen::VectorXd Fq_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd Fv_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd C_res = Eigen::VectorXd::Random(dimc);
  const Eigen::VectorXd la_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd lf_res = Eigen::VectorXd::Random(dimf);
  const Eigen::VectorXd lr_res = Eigen::VectorXd::Random(dimr);
  const Eigen::VectorXd lq_res = Eigen::VectorXd::Random(dimv);
  const Eigen::VectorXd lv_res = Eigen::VectorXd::Random(dimv);
  residual.Fq() = Fq_res;
  residual.Fv() = Fv_res;
  residual.C() = C_res;
  residual.la() = la_res;
  residual.lf() = lf_res;
  residual.lr() = lr_res;
  residual.lq() = lq_res;
  residual.lv() = lv_res;
  EXPECT_TRUE(residual.KKT_residual.segment(                0, dimv).isApprox(Fq_res));
  EXPECT_TRUE(residual.KKT_residual.segment(             dimv, dimv).isApprox(Fv_res));
  EXPECT_TRUE(residual.KKT_residual.segment(           2*dimv, dimc).isApprox(C_res));
  EXPECT_TRUE(residual.KKT_residual.segment(      2*dimv+dimc, dimv).isApprox(la_res));
  EXPECT_TRUE(residual.KKT_residual.segment(      3*dimv+dimc, dimf).isApprox(lf_res));
  EXPECT_TRUE(residual.KKT_residual.segment( 3*dimv+dimc+dimf, dimr).isApprox(lr_res));
  EXPECT_TRUE(residual.KKT_residual.segment(3*dimv+dimc+dimfr, dimv).isApprox(lq_res));
  EXPECT_TRUE(residual.KKT_residual.segment(4*dimv+dimc+dimfr, dimv).isApprox(lv_res));
  EXPECT_TRUE(residual.lx().head(dimv).isApprox(lq_res));
  EXPECT_TRUE(residual.lx().tail(dimv).isApprox(lv_res));
  EXPECT_EQ(residual.lx().size(), 2*dimv);
  EXPECT_TRUE(residual.l_afr().segment(        0, dimv).isApprox(la_res));
  EXPECT_TRUE(residual.l_afr().segment(     dimv, dimf).isApprox(lf_res));
  EXPECT_TRUE(residual.l_afr().segment(dimv+dimf, dimr).isApprox(lr_res));
  EXPECT_EQ(residual.l_afr().size(), dimv+dimf+dimr);
  residual.setZero();
  EXPECT_TRUE(residual.KKT_residual.isZero());
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}