#include <string>

#include <gtest/gtest.h>

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/kkt_matrix.hpp"


namespace idocp {

class KKTMatrixTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    std::random_device rnd;
    fixed_base_urdf_ = "../urdf/iiwa14/iiwa14.urdf";
    floating_base_urdf_ = "../urdf/anymal/anymal.urdf";
  }

  virtual void TearDown() {
  }

  double dtau_;
  std::string fixed_base_urdf_, floating_base_urdf_;
};


TEST_F(KKTMatrixTest, fixed_base) {
  Robot robot(fixed_base_urdf_);
  KKTMatrix matrix(robot);
  const int dimv = robot.dimv();
  const int dimf = 5*robot.num_point_contacts();
  const int dimr = 2*robot.num_point_contacts();
  const int dimfr = dimf + dimr;
  const int dimc = robot.dim_passive();
  EXPECT_EQ(matrix.dimKKT(), 5*dimv);
  EXPECT_EQ(matrix.dimc(), 0);
  EXPECT_EQ(matrix.dimf(), 0);
  EXPECT_EQ(matrix.dimr(), 0);
  const Eigen::MatrixXd Ca = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Cf = Eigen::MatrixXd::Random(dimc, dimf);
  const Eigen::MatrixXd Cr = Eigen::MatrixXd::Random(dimc, dimr);
  const Eigen::MatrixXd Cq = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Cv = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Qaa = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qaf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qar = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qaq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qav = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qfa = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qff = Eigen::MatrixXd::Random(dimf, dimf);
  const Eigen::MatrixXd Qfr = Eigen::MatrixXd::Random(dimf, dimr);
  const Eigen::MatrixXd Qfq = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qfv = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qra = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qrf = Eigen::MatrixXd::Random(dimr, dimf);
  const Eigen::MatrixXd Qrr = Eigen::MatrixXd::Random(dimr, dimr);
  const Eigen::MatrixXd Qrq = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qrv = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qqa = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qqf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qqr = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qqq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qqv = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qva = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qvf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qvr = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qvq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qvv = Eigen::MatrixXd::Random(dimv, dimv);
  matrix.Ca() = Ca;
  matrix.Cf() = Cf;
  matrix.Cr() = Cr;
  matrix.Cq() = Cq;
  matrix.Cv() = Cv;
  matrix.Qaa() = Qaa;
  matrix.Qaf() = Qaf;
  matrix.Qar() = Qar;
  matrix.Qaq() = Qaq;
  matrix.Qav() = Qav;
  matrix.Qfa() = Qfa;
  matrix.Qff() = Qff;
  matrix.Qfr() = Qfr;
  matrix.Qfq() = Qfq;
  matrix.Qfv() = Qfv;
  matrix.Qra() = Qra;
  matrix.Qrf() = Qrf;
  matrix.Qrr() = Qrr;
  matrix.Qrq() = Qrq;
  matrix.Qrv() = Qrv;
  matrix.Qqa() = Qqa;
  matrix.Qqf() = Qqf;
  matrix.Qqr() = Qqr;
  matrix.Qqq() = Qqq;
  matrix.Qqv() = Qqv;
  matrix.Qva() = Qva;
  matrix.Qvf() = Qvf;
  matrix.Qvr() = Qvr;
  matrix.Qvq() = Qvq;
  matrix.Qvv() = Qvv;
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, 0, dimc, dimv).isApprox(Ca));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv, dimc, dimf).isApprox(Cf));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv+dimf, dimc, dimr).isApprox(Cr));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv+dimf+dimr, dimc, dimv).isApprox(Cq));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, 2*dimv+dimf+dimr, dimc, dimv).isApprox(Cv));
  EXPECT_TRUE(matrix.costHessian().block(0, 0, dimv, dimv).isApprox(Qaa));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv, dimv, dimf).isApprox(Qaf));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv+dimf, dimv, dimr).isApprox(Qar));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv+dimf+dimr, dimv, dimv).isApprox(Qaq));
  EXPECT_TRUE(matrix.costHessian().block(0, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qav));
  EXPECT_TRUE(matrix.costHessian().block(dimv, 0, dimf, dimv).isApprox(Qfa));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv, dimf, dimf).isApprox(Qff));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv+dimf, dimf, dimr).isApprox(Qfr));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv+dimf+dimr, dimf, dimv).isApprox(Qfq));
  EXPECT_TRUE(matrix.costHessian().block(dimv, 2*dimv+dimf+dimr, dimf, dimv).isApprox(Qfv));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, 0, dimr, dimv).isApprox(Qra));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv, dimr, dimf).isApprox(Qrf));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv+dimf, dimr, dimr).isApprox(Qrr));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv+dimf+dimr, dimr, dimv).isApprox(Qrq));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, 2*dimv+dimf+dimr, dimr, dimv).isApprox(Qrv));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, 0, dimv, dimv).isApprox(Qqa));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv, dimv, dimf).isApprox(Qqf));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv+dimf, dimv, dimr).isApprox(Qqr));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv+dimf+dimr, dimv, dimv).isApprox(Qqq));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qqv));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, 0, dimv, dimv).isApprox(Qva));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv, dimv, dimf).isApprox(Qvf));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv+dimf, dimv, dimr).isApprox(Qvr));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv+dimf+dimr, dimv, dimv).isApprox(Qvq));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qvv));
  EXPECT_EQ(matrix.Qxx().rows(), 2*dimv);
  EXPECT_EQ(matrix.Qxx().cols(), 2*dimv);
  EXPECT_TRUE(matrix.Qxx().block(0, 0, dimv, dimv).isApprox(Qqq));
  EXPECT_TRUE(matrix.Qxx().block(0, dimv, dimv, dimv).isApprox(Qqv));
  EXPECT_TRUE(matrix.Qxx().block(dimv, 0, dimv, dimv).isApprox(Qvq));
  EXPECT_TRUE(matrix.Qxx().block(dimv, dimv, dimv, dimv).isApprox(Qvv));
  EXPECT_EQ(matrix.Q_afr_afr().rows(), dimv+dimf+dimr);
  EXPECT_EQ(matrix.Q_afr_afr().cols(), dimv+dimf+dimr);
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, 0, dimv, dimv).isApprox(Qaa));
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, dimv, dimv, dimf).isApprox(Qaf));
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, dimv+dimf, dimv, dimr).isApprox(Qar));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, 0, dimf, dimv).isApprox(Qfa));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, dimv, dimf, dimf).isApprox(Qff));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, dimv+dimf, dimf, dimr).isApprox(Qfr));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, 0, dimr, dimv).isApprox(Qra));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, dimv, dimr, dimf).isApprox(Qrf));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, dimv+dimf, dimr, dimr).isApprox(Qrr));
  EXPECT_EQ(matrix.Q_afr_qv().rows(), dimv+dimf+dimr);
  EXPECT_EQ(matrix.Q_afr_qv().cols(), 2*dimv);
  EXPECT_TRUE(matrix.Q_afr_qv().block(0, 0, dimv, dimv).isApprox(Qaq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(0, dimv, dimv, dimv).isApprox(Qav));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv, 0, dimf, dimv).isApprox(Qfq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv, dimv, dimf, dimv).isApprox(Qfv));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv+dimf, 0, dimr, dimv).isApprox(Qrq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv+dimf, dimv, dimr, dimv).isApprox(Qrv));
  EXPECT_EQ(matrix.C_qv().rows(), dimc);
  EXPECT_EQ(matrix.C_qv().cols(), 2*dimv);
  EXPECT_TRUE(matrix.C_qv().block(0, 0, dimc, dimv).isApprox(Cq));
  EXPECT_TRUE(matrix.C_qv().block(0, dimv, dimc, dimv).isApprox(Cv));
  EXPECT_EQ(matrix.C_afr().rows(), dimc);
  EXPECT_EQ(matrix.C_afr().cols(), dimv+dimf+dimr);
  EXPECT_TRUE(matrix.C_afr().block(0, 0, dimc, dimv).isApprox(Ca));
  EXPECT_TRUE(matrix.C_afr().block(0, dimv, dimc, dimf).isApprox(Cf));
  EXPECT_TRUE(matrix.C_afr().block(0, dimv+dimf, dimc, dimr).isApprox(Cr));
  matrix.setZero();
  EXPECT_TRUE(matrix.costHessian().isZero());
  EXPECT_TRUE(matrix.constraintsJacobian().isZero());
}



TEST_F(KKTMatrixTest, fixed_base_contact) {
  std::vector<int> contact_frames = {18};
  Robot robot(fixed_base_urdf_, contact_frames, 0, 0);
  KKTMatrix matrix(robot);
  const int dimv = robot.dimv();
  const int dimf = 5*robot.num_point_contacts();
  const int dimr = 2*robot.num_point_contacts();
  const int dimfr = dimf + dimr;
  const int dimc = robot.dim_passive();
  EXPECT_EQ(matrix.dimKKT(), 5*dimv+dimfr);
  EXPECT_EQ(matrix.dimc(), 0);
  EXPECT_EQ(matrix.dimf(), 5);
  EXPECT_EQ(matrix.dimr(), 2);
  const Eigen::MatrixXd Ca = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Cf = Eigen::MatrixXd::Random(dimc, dimf);
  const Eigen::MatrixXd Cr = Eigen::MatrixXd::Random(dimc, dimr);
  const Eigen::MatrixXd Cq = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Cv = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Qaa = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qaf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qar = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qaq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qav = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qfa = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qff = Eigen::MatrixXd::Random(dimf, dimf);
  const Eigen::MatrixXd Qfr = Eigen::MatrixXd::Random(dimf, dimr);
  const Eigen::MatrixXd Qfq = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qfv = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qra = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qrf = Eigen::MatrixXd::Random(dimr, dimf);
  const Eigen::MatrixXd Qrr = Eigen::MatrixXd::Random(dimr, dimr);
  const Eigen::MatrixXd Qrq = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qrv = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qqa = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qqf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qqr = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qqq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qqv = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qva = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qvf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qvr = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qvq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qvv = Eigen::MatrixXd::Random(dimv, dimv);
  matrix.Ca() = Ca;
  matrix.Cf() = Cf;
  matrix.Cr() = Cr;
  matrix.Cq() = Cq;
  matrix.Cv() = Cv;
  matrix.Qaa() = Qaa;
  matrix.Qaf() = Qaf;
  matrix.Qar() = Qar;
  matrix.Qaq() = Qaq;
  matrix.Qav() = Qav;
  matrix.Qfa() = Qfa;
  matrix.Qff() = Qff;
  matrix.Qfr() = Qfr;
  matrix.Qfq() = Qfq;
  matrix.Qfv() = Qfv;
  matrix.Qra() = Qra;
  matrix.Qrf() = Qrf;
  matrix.Qrr() = Qrr;
  matrix.Qrq() = Qrq;
  matrix.Qrv() = Qrv;
  matrix.Qqa() = Qqa;
  matrix.Qqf() = Qqf;
  matrix.Qqr() = Qqr;
  matrix.Qqq() = Qqq;
  matrix.Qqv() = Qqv;
  matrix.Qva() = Qva;
  matrix.Qvf() = Qvf;
  matrix.Qvr() = Qvr;
  matrix.Qvq() = Qvq;
  matrix.Qvv() = Qvv;
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, 0, dimc, dimv).isApprox(Ca));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv, dimc, dimf).isApprox(Cf));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv+dimf, dimc, dimr).isApprox(Cr));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv+dimf+dimr, dimc, dimv).isApprox(Cq));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, 2*dimv+dimf+dimr, dimc, dimv).isApprox(Cv));
  EXPECT_TRUE(matrix.costHessian().block(0, 0, dimv, dimv).isApprox(Qaa));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv, dimv, dimf).isApprox(Qaf));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv+dimf, dimv, dimr).isApprox(Qar));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv+dimf+dimr, dimv, dimv).isApprox(Qaq));
  EXPECT_TRUE(matrix.costHessian().block(0, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qav));
  EXPECT_TRUE(matrix.costHessian().block(dimv, 0, dimf, dimv).isApprox(Qfa));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv, dimf, dimf).isApprox(Qff));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv+dimf, dimf, dimr).isApprox(Qfr));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv+dimf+dimr, dimf, dimv).isApprox(Qfq));
  EXPECT_TRUE(matrix.costHessian().block(dimv, 2*dimv+dimf+dimr, dimf, dimv).isApprox(Qfv));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, 0, dimr, dimv).isApprox(Qra));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv, dimr, dimf).isApprox(Qrf));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv+dimf, dimr, dimr).isApprox(Qrr));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv+dimf+dimr, dimr, dimv).isApprox(Qrq));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, 2*dimv+dimf+dimr, dimr, dimv).isApprox(Qrv));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, 0, dimv, dimv).isApprox(Qqa));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv, dimv, dimf).isApprox(Qqf));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv+dimf, dimv, dimr).isApprox(Qqr));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv+dimf+dimr, dimv, dimv).isApprox(Qqq));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qqv));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, 0, dimv, dimv).isApprox(Qva));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv, dimv, dimf).isApprox(Qvf));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv+dimf, dimv, dimr).isApprox(Qvr));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv+dimf+dimr, dimv, dimv).isApprox(Qvq));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qvv));
  EXPECT_EQ(matrix.Qxx().rows(), 2*dimv);
  EXPECT_EQ(matrix.Qxx().cols(), 2*dimv);
  EXPECT_TRUE(matrix.Qxx().block(0, 0, dimv, dimv).isApprox(Qqq));
  EXPECT_TRUE(matrix.Qxx().block(0, dimv, dimv, dimv).isApprox(Qqv));
  EXPECT_TRUE(matrix.Qxx().block(dimv, 0, dimv, dimv).isApprox(Qvq));
  EXPECT_TRUE(matrix.Qxx().block(dimv, dimv, dimv, dimv).isApprox(Qvv));
  EXPECT_EQ(matrix.Q_afr_afr().rows(), dimv+dimf+dimr);
  EXPECT_EQ(matrix.Q_afr_afr().cols(), dimv+dimf+dimr);
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, 0, dimv, dimv).isApprox(Qaa));
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, dimv, dimv, dimf).isApprox(Qaf));
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, dimv+dimf, dimv, dimr).isApprox(Qar));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, 0, dimf, dimv).isApprox(Qfa));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, dimv, dimf, dimf).isApprox(Qff));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, dimv+dimf, dimf, dimr).isApprox(Qfr));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, 0, dimr, dimv).isApprox(Qra));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, dimv, dimr, dimf).isApprox(Qrf));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, dimv+dimf, dimr, dimr).isApprox(Qrr));
  EXPECT_EQ(matrix.Q_afr_qv().rows(), dimv+dimf+dimr);
  EXPECT_EQ(matrix.Q_afr_qv().cols(), 2*dimv);
  EXPECT_TRUE(matrix.Q_afr_qv().block(0, 0, dimv, dimv).isApprox(Qaq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(0, dimv, dimv, dimv).isApprox(Qav));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv, 0, dimf, dimv).isApprox(Qfq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv, dimv, dimf, dimv).isApprox(Qfv));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv+dimf, 0, dimr, dimv).isApprox(Qrq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv+dimf, dimv, dimr, dimv).isApprox(Qrv));
  EXPECT_EQ(matrix.C_qv().rows(), dimc);
  EXPECT_EQ(matrix.C_qv().cols(), 2*dimv);
  EXPECT_TRUE(matrix.C_qv().block(0, 0, dimc, dimv).isApprox(Cq));
  EXPECT_TRUE(matrix.C_qv().block(0, dimv, dimc, dimv).isApprox(Cv));
  EXPECT_EQ(matrix.C_afr().rows(), dimc);
  EXPECT_EQ(matrix.C_afr().cols(), dimv+dimf+dimr);
  EXPECT_TRUE(matrix.C_afr().block(0, 0, dimc, dimv).isApprox(Ca));
  EXPECT_TRUE(matrix.C_afr().block(0, dimv, dimc, dimf).isApprox(Cf));
  EXPECT_TRUE(matrix.C_afr().block(0, dimv+dimf, dimc, dimr).isApprox(Cr));
  matrix.setZero();
  EXPECT_TRUE(matrix.costHessian().isZero());
  EXPECT_TRUE(matrix.constraintsJacobian().isZero());
}


TEST_F(KKTMatrixTest, invert_fixed_base) {
  std::vector<int> contact_frames = {18};
  Robot robot(fixed_base_urdf_, contact_frames, 0, 0);
  KKTMatrix matrix(robot);
  const int dimv = robot.dimv();
  const int dimx = 2*robot.dimv();
  const int dimfr = 7*robot.num_point_contacts();
  const int dim_passive = robot.dim_passive();
  const int dimc = robot.dim_passive();
  const int dimQ = 3*robot.dimv() + 7*robot.num_point_contacts();
  const Eigen::MatrixXd Q_seed_mat = Eigen::MatrixXd::Random(dimQ, dimQ);
  const Eigen::MatrixXd Q_mat = Q_seed_mat * Q_seed_mat.transpose() + Eigen::MatrixXd::Identity(dimQ, dimQ);
  const Eigen::MatrixXd Jc_mat = Eigen::MatrixXd::Random(dimc, dimQ);
  matrix.costHessian() = Q_mat;
  matrix.constraintsJacobian() = Jc_mat;
  const double dtau = std::abs(Eigen::VectorXd::Random(1)[0]);
  const int dimKKT = 5*dimv+dimc+dimfr;
  Eigen::MatrixXd kkt_mat_ref = Eigen::MatrixXd::Zero(dimKKT, dimKKT);
  kkt_mat_ref.bottomRightCorner(dimQ, dimQ) = Q_mat;
  kkt_mat_ref.block(dimx, dimx+dimc, dimc, dimQ) = Jc_mat;
  kkt_mat_ref.block(0, dimx+dimc+dimv+dimfr, dimv, dimv) 
      = -1 * Eigen::MatrixXd::Identity(dimv, dimv);
  kkt_mat_ref.block(0, dimx+dimc+2*dimv+dimfr, dimv, dimv) 
      = dtau * Eigen::MatrixXd::Identity(dimv, dimv);
  kkt_mat_ref.block(dimv, dimx+dimc, dimv, dimv) 
      = dtau * Eigen::MatrixXd::Identity(dimv, dimv);
  kkt_mat_ref.block(dimv, dimx+dimc+2*dimv+dimfr, dimv, dimv) 
      = -1 * Eigen::MatrixXd::Identity(dimv, dimv);
  kkt_mat_ref.triangularView<Eigen::StrictlyLower>() 
      = kkt_mat_ref.transpose().triangularView<Eigen::StrictlyLower>();
  std::cout << kkt_mat_ref << std::endl;
  // const Eigen::MatrixXd kkt_mat_inv_ref = kkt_mat_ref.inverse();
  const Eigen::MatrixXd kkt_mat_inv_ref = kkt_mat_ref.ldlt().solve(Eigen::MatrixXd::Identity(dimKKT, dimKKT));
  Eigen::MatrixXd kkt_mat_inv = Eigen::MatrixXd::Zero(dimKKT, dimKKT);
  matrix.invert(dtau, kkt_mat_inv);
  EXPECT_TRUE(kkt_mat_inv.isApprox(kkt_mat_inv_ref, 1.0e-08));
  std::cout << "error l2 norm = " << (kkt_mat_inv - kkt_mat_inv_ref).lpNorm<2>() << std::endl;
}


TEST_F(KKTMatrixTest, floating_base) {
  std::vector<int> contact_frames = {14, 24, 34, 44};
  Robot robot(floating_base_urdf_, contact_frames, 0, 0);
  KKTMatrix matrix(robot);
  const int dimv = robot.dimv();
  const int dimf = 5*robot.num_point_contacts();
  const int dimr = 2*robot.num_point_contacts();
  const int dimfr = dimf + dimr;
  const int dimc = robot.dim_passive();
  EXPECT_EQ(matrix.dimKKT(), 5*dimv+dimc+dimfr);
  EXPECT_EQ(matrix.dimc(), 6);
  EXPECT_EQ(matrix.dimf(), 5*4);
  EXPECT_EQ(matrix.dimr(), 2*4);
  const Eigen::MatrixXd Ca = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Cf = Eigen::MatrixXd::Random(dimc, dimf);
  const Eigen::MatrixXd Cr = Eigen::MatrixXd::Random(dimc, dimr);
  const Eigen::MatrixXd Cq = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Cv = Eigen::MatrixXd::Random(dimc, dimv);
  const Eigen::MatrixXd Qaa = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qaf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qar = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qaq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qav = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qfa = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qff = Eigen::MatrixXd::Random(dimf, dimf);
  const Eigen::MatrixXd Qfr = Eigen::MatrixXd::Random(dimf, dimr);
  const Eigen::MatrixXd Qfq = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qfv = Eigen::MatrixXd::Random(dimf, dimv);
  const Eigen::MatrixXd Qra = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qrf = Eigen::MatrixXd::Random(dimr, dimf);
  const Eigen::MatrixXd Qrr = Eigen::MatrixXd::Random(dimr, dimr);
  const Eigen::MatrixXd Qrq = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qrv = Eigen::MatrixXd::Random(dimr, dimv);
  const Eigen::MatrixXd Qqa = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qqf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qqr = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qqq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qqv = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qva = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qvf = Eigen::MatrixXd::Random(dimv, dimf);
  const Eigen::MatrixXd Qvr = Eigen::MatrixXd::Random(dimv, dimr);
  const Eigen::MatrixXd Qvq = Eigen::MatrixXd::Random(dimv, dimv);
  const Eigen::MatrixXd Qvv = Eigen::MatrixXd::Random(dimv, dimv);
  matrix.Ca() = Ca;
  matrix.Cf() = Cf;
  matrix.Cr() = Cr;
  matrix.Cq() = Cq;
  matrix.Cv() = Cv;
  matrix.Qaa() = Qaa;
  matrix.Qaf() = Qaf;
  matrix.Qar() = Qar;
  matrix.Qaq() = Qaq;
  matrix.Qav() = Qav;
  matrix.Qfa() = Qfa;
  matrix.Qff() = Qff;
  matrix.Qfr() = Qfr;
  matrix.Qfq() = Qfq;
  matrix.Qfv() = Qfv;
  matrix.Qra() = Qra;
  matrix.Qrf() = Qrf;
  matrix.Qrr() = Qrr;
  matrix.Qrq() = Qrq;
  matrix.Qrv() = Qrv;
  matrix.Qqa() = Qqa;
  matrix.Qqf() = Qqf;
  matrix.Qqr() = Qqr;
  matrix.Qqq() = Qqq;
  matrix.Qqv() = Qqv;
  matrix.Qva() = Qva;
  matrix.Qvf() = Qvf;
  matrix.Qvr() = Qvr;
  matrix.Qvq() = Qvq;
  matrix.Qvv() = Qvv;
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, 0, dimc, dimv).isApprox(Ca));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv, dimc, dimf).isApprox(Cf));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv+dimf, dimc, dimr).isApprox(Cr));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, dimv+dimf+dimr, dimc, dimv).isApprox(Cq));
  EXPECT_TRUE(matrix.constraintsJacobian().block(0, 2*dimv+dimf+dimr, dimc, dimv).isApprox(Cv));
  EXPECT_TRUE(matrix.costHessian().block(0, 0, dimv, dimv).isApprox(Qaa));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv, dimv, dimf).isApprox(Qaf));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv+dimf, dimv, dimr).isApprox(Qar));
  EXPECT_TRUE(matrix.costHessian().block(0, dimv+dimf+dimr, dimv, dimv).isApprox(Qaq));
  EXPECT_TRUE(matrix.costHessian().block(0, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qav));
  EXPECT_TRUE(matrix.costHessian().block(dimv, 0, dimf, dimv).isApprox(Qfa));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv, dimf, dimf).isApprox(Qff));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv+dimf, dimf, dimr).isApprox(Qfr));
  EXPECT_TRUE(matrix.costHessian().block(dimv, dimv+dimf+dimr, dimf, dimv).isApprox(Qfq));
  EXPECT_TRUE(matrix.costHessian().block(dimv, 2*dimv+dimf+dimr, dimf, dimv).isApprox(Qfv));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, 0, dimr, dimv).isApprox(Qra));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv, dimr, dimf).isApprox(Qrf));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv+dimf, dimr, dimr).isApprox(Qrr));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, dimv+dimf+dimr, dimr, dimv).isApprox(Qrq));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf, 2*dimv+dimf+dimr, dimr, dimv).isApprox(Qrv));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, 0, dimv, dimv).isApprox(Qqa));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv, dimv, dimf).isApprox(Qqf));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv+dimf, dimv, dimr).isApprox(Qqr));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, dimv+dimf+dimr, dimv, dimv).isApprox(Qqq));
  EXPECT_TRUE(matrix.costHessian().block(dimv+dimf+dimr, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qqv));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, 0, dimv, dimv).isApprox(Qva));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv, dimv, dimf).isApprox(Qvf));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv+dimf, dimv, dimr).isApprox(Qvr));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, dimv+dimf+dimr, dimv, dimv).isApprox(Qvq));
  EXPECT_TRUE(matrix.costHessian().block(2*dimv+dimf+dimr, 2*dimv+dimf+dimr, dimv, dimv).isApprox(Qvv));
  EXPECT_EQ(matrix.Qxx().rows(), 2*dimv);
  EXPECT_EQ(matrix.Qxx().cols(), 2*dimv);
  EXPECT_TRUE(matrix.Qxx().block(0, 0, dimv, dimv).isApprox(Qqq));
  EXPECT_TRUE(matrix.Qxx().block(0, dimv, dimv, dimv).isApprox(Qqv));
  EXPECT_TRUE(matrix.Qxx().block(dimv, 0, dimv, dimv).isApprox(Qvq));
  EXPECT_TRUE(matrix.Qxx().block(dimv, dimv, dimv, dimv).isApprox(Qvv));
  EXPECT_EQ(matrix.Q_afr_afr().rows(), dimv+dimf+dimr);
  EXPECT_EQ(matrix.Q_afr_afr().cols(), dimv+dimf+dimr);
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, 0, dimv, dimv).isApprox(Qaa));
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, dimv, dimv, dimf).isApprox(Qaf));
  EXPECT_TRUE(matrix.Q_afr_afr().block(0, dimv+dimf, dimv, dimr).isApprox(Qar));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, 0, dimf, dimv).isApprox(Qfa));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, dimv, dimf, dimf).isApprox(Qff));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv, dimv+dimf, dimf, dimr).isApprox(Qfr));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, 0, dimr, dimv).isApprox(Qra));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, dimv, dimr, dimf).isApprox(Qrf));
  EXPECT_TRUE(matrix.Q_afr_afr().block(dimv+dimf, dimv+dimf, dimr, dimr).isApprox(Qrr));
  EXPECT_EQ(matrix.Q_afr_qv().rows(), dimv+dimf+dimr);
  EXPECT_EQ(matrix.Q_afr_qv().cols(), 2*dimv);
  EXPECT_TRUE(matrix.Q_afr_qv().block(0, 0, dimv, dimv).isApprox(Qaq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(0, dimv, dimv, dimv).isApprox(Qav));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv, 0, dimf, dimv).isApprox(Qfq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv, dimv, dimf, dimv).isApprox(Qfv));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv+dimf, 0, dimr, dimv).isApprox(Qrq));
  EXPECT_TRUE(matrix.Q_afr_qv().block(dimv+dimf, dimv, dimr, dimv).isApprox(Qrv));
  EXPECT_EQ(matrix.C_qv().rows(), dimc);
  EXPECT_EQ(matrix.C_qv().cols(), 2*dimv);
  EXPECT_TRUE(matrix.C_qv().block(0, 0, dimc, dimv).isApprox(Cq));
  EXPECT_TRUE(matrix.C_qv().block(0, dimv, dimc, dimv).isApprox(Cv));
  EXPECT_EQ(matrix.C_afr().rows(), dimc);
  EXPECT_EQ(matrix.C_afr().cols(), dimv+dimf+dimr);
  EXPECT_TRUE(matrix.C_afr().block(0, 0, dimc, dimv).isApprox(Ca));
  EXPECT_TRUE(matrix.C_afr().block(0, dimv, dimc, dimf).isApprox(Cf));
  EXPECT_TRUE(matrix.C_afr().block(0, dimv+dimf, dimc, dimr).isApprox(Cr));
  matrix.setZero();
  EXPECT_TRUE(matrix.costHessian().isZero());
  EXPECT_TRUE(matrix.constraintsJacobian().isZero());
}


TEST_F(KKTMatrixTest, invert_floating_base) {
  std::vector<int> contact_frames = {14, 24, 34, 44};
  Robot robot(floating_base_urdf_, contact_frames, 0, 0);
  KKTMatrix matrix(robot);
  const int dimv = robot.dimv();
  const int dimx = 2*robot.dimv();
  const int dimfr = 7*robot.num_point_contacts();
  const int dim_passive = robot.dim_passive();
  const int dimc = robot.dim_passive();
  const int dimQ = 3*robot.dimv() + 7*robot.num_point_contacts();
  const Eigen::MatrixXd Q_seed_mat = Eigen::MatrixXd::Random(dimQ, dimQ);
  const Eigen::MatrixXd Q_mat = Q_seed_mat * Q_seed_mat.transpose() + Eigen::MatrixXd::Identity(dimQ, dimQ);
  const Eigen::MatrixXd Jc_mat = Eigen::MatrixXd::Random(dimc, dimQ);
  matrix.costHessian() = Q_mat;
  matrix.constraintsJacobian() = Jc_mat;
  const Eigen::MatrixXd Fqq_seed_mat = Eigen::MatrixXd::Random(6, 6);
  const Eigen::MatrixXd Fqq_mat = Fqq_seed_mat * Fqq_seed_mat.transpose();
  matrix.Fqq = -1 * Eigen::MatrixXd::Identity(dimv, dimv);
  matrix.Fqq.topLeftCorner(6, 6) = - Fqq_mat;
  const double dtau = std::abs(Eigen::VectorXd::Random(1)[0]);
  const int dimKKT = 5*dimv+dimc+dimfr;
  Eigen::MatrixXd kkt_mat_ref = Eigen::MatrixXd::Zero(dimKKT, dimKKT);
  kkt_mat_ref.bottomRightCorner(dimQ, dimQ) = Q_mat;
  kkt_mat_ref.block(dimx, dimx+dimc, dimc, dimQ) = Jc_mat;
  kkt_mat_ref.block(0, dimx+dimc+dimv+dimfr, dimv, dimv) 
      = -1 * Eigen::MatrixXd::Identity(dimv, dimv);
  kkt_mat_ref.block(0, dimx+dimc+dimv+dimfr, dimv, dimv).topLeftCorner(6, 6)
      = - Fqq_mat;
  kkt_mat_ref.block(0, dimx+dimc+2*dimv+dimfr, dimv, dimv) 
      = dtau * Eigen::MatrixXd::Identity(dimv, dimv);
  kkt_mat_ref.block(dimv, dimx+dimc, dimv, dimv) 
      = dtau * Eigen::MatrixXd::Identity(dimv, dimv);
  kkt_mat_ref.block(dimv, dimx+dimc+2*dimv+dimfr, dimv, dimv) 
      = -1 * Eigen::MatrixXd::Identity(dimv, dimv);
  kkt_mat_ref.triangularView<Eigen::StrictlyLower>() 
      = kkt_mat_ref.transpose().triangularView<Eigen::StrictlyLower>();
  std::cout << kkt_mat_ref << std::endl;
  const Eigen::MatrixXd kkt_mat_inv_ref = kkt_mat_ref.inverse();
  Eigen::MatrixXd kkt_mat_inv = Eigen::MatrixXd::Zero(dimKKT, dimKKT);
  matrix.invert(dtau, kkt_mat_inv);
  EXPECT_TRUE(kkt_mat_inv.isApprox(kkt_mat_inv_ref, 1.0e-08));
  std::cout << "error l2 norm = " << (kkt_mat_inv - kkt_mat_inv_ref).lpNorm<2>() << std::endl;
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}