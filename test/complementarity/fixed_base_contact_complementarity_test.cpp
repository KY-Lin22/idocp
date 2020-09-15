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
#include "idocp/complementarity/contact_complementarity.hpp"
#include "idocp/complementarity/contact_force_inequality.hpp"
#include "idocp/complementarity/baumgarte_inequality.hpp"
#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {

class FixedBaseContactComplementarityTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    mu_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    max_complementarity_violation_ = 1.0e-04;
    barrier_ = 1.0e-04;
    dtau_ = std::abs(Eigen::VectorXd::Random(1)[0]);
    fraction_to_boundary_rate_ = 0.995;
    const std::vector<int> contact_frames = {18};
    const std::string urdf = "../urdf/iiwa14/iiwa14.urdf";
    robot_ = Robot(urdf, contact_frames, 0, 0);
    contact_complementarity_ = ContactComplementarity(
        robot_, mu_, max_complementarity_violation_, barrier_, 
        fraction_to_boundary_rate_);
    force_inequality_ = ContactForceInequality(robot_, mu_);
    baum_inequality_ = BaumgarteInequality(robot_);
    force_data_ = ConstraintComponentData(6*robot_.num_point_contacts());
    baum_data_ = ConstraintComponentData(6*robot_.num_point_contacts());
    complementarity_data_ = ConstraintComponentData(6*robot_.num_point_contacts());
    dimc_ = 6 * robot_.num_point_contacts();
  }

  virtual void TearDown() {
  }

  double mu_, max_complementarity_violation_, barrier_, dtau_, 
         fraction_to_boundary_rate_;
  Robot robot_;
  ContactComplementarity contact_complementarity_;
  ContactForceInequality force_inequality_;
  BaumgarteInequality baum_inequality_;
  ConstraintComponentData force_data_, baum_data_, complementarity_data_;
  int dimc_;
};


TEST_F(FixedBaseContactComplementarityTest, isFeasible) {
  SplitSolution s(robot_);
  ASSERT_TRUE(s.f.size() == 5);
  ASSERT_TRUE(s.f_3D.size() == 3);
  ASSERT_TRUE(s.r.size() == 2);
  ASSERT_TRUE(robot_.num_point_contacts() == 1);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  robot_.updateKinematics(s.q, s.v, s.a);
  s.f = - Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  EXPECT_FALSE(contact_complementarity_.isFeasible(robot_, s));
  contact_complementarity_.setSlackAndDual(robot_, dtau_, s);
  contact_complementarity_.computeResidual(robot_, dtau_, s);
  EXPECT_FALSE(contact_complementarity_.residualL1Nrom() == 0);
  EXPECT_FALSE(contact_complementarity_.squaredKKTErrorNorm() == 0);
  s.set_f_3D();
  s.f(4) = 1.1 * std::sqrt((s.f_3D(0)*s.f_3D(0)+s.f_3D(1)*s.f_3D(1))/(mu_*mu_));
  s.set_f_3D();
  s.r = - Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  EXPECT_FALSE(contact_complementarity_.isFeasible(robot_, s));
  contact_complementarity_.setSlackAndDual(robot_, dtau_, s);
  contact_complementarity_.computeResidual(robot_, dtau_, s);
  EXPECT_TRUE(contact_complementarity_.residualL1Nrom() > 0);
  EXPECT_TRUE(contact_complementarity_.squaredKKTErrorNorm() > 0);
}


TEST_F(FixedBaseContactComplementarityTest, setSlackAndDual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  robot_.updateKinematics(s.q, s.v, s.a);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  s.f(4) = 1.1 * std::sqrt((s.f_3D(0)*s.f_3D(0)+s.f_3D(1)*s.f_3D(1))/(mu_*mu_));
  s.set_f_3D();
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  contact_complementarity_.setSlackAndDual(robot_, dtau_, s);
  EXPECT_FALSE(std::isnan(contact_complementarity_.costSlackBarrier()));
}


TEST_F(FixedBaseContactComplementarityTest, computeResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  robot_.updateKinematics(s.q, s.v, s.a);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  contact_complementarity_.computeResidual(robot_, dtau_, s);
  const double L1Norm = contact_complementarity_.residualL1Nrom();
  const double squaredNorm = contact_complementarity_.squaredKKTErrorNorm();
  force_inequality_.computePrimalResidual(robot_, dtau_, s, force_data_);
  baum_inequality_.computePrimalResidual(robot_, dtau_, s, baum_data_);
  const double L1Norm_ref = force_data_.residual.lpNorm<1>() 
                              + baum_data_.residual.lpNorm<1>() 
                              + Eigen::VectorXd::Constant(dimc_, barrier_).lpNorm<1>() 
                              + Eigen::VectorXd::Constant(dimc_, barrier_).lpNorm<1>() 
                              + Eigen::VectorXd::Constant(dimc_, barrier_).lpNorm<1>() 
                              + Eigen::VectorXd::Constant(dimc_, max_complementarity_violation_).lpNorm<1>();
  const double squaredNorm_ref = force_data_.residual.squaredNorm() 
                                  + baum_data_.residual.squaredNorm() 
                                  + Eigen::VectorXd::Constant(dimc_, barrier_).squaredNorm() 
                                  + Eigen::VectorXd::Constant(dimc_, barrier_).squaredNorm() 
                                  + Eigen::VectorXd::Constant(dimc_, barrier_).squaredNorm() 
                                  + Eigen::VectorXd::Constant(dimc_, max_complementarity_violation_).squaredNorm();
  EXPECT_DOUBLE_EQ(L1Norm, L1Norm_ref);
  EXPECT_DOUBLE_EQ(squaredNorm, squaredNorm_ref);
}


TEST_F(FixedBaseContactComplementarityTest, augmentDualResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  robot_.updateKinematics(s.q, s.v, s.a);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  KKTResidual kkt_residual(robot_);
  contact_complementarity_.augmentDualResidual(robot_, dtau_, s, kkt_residual);
  EXPECT_TRUE(kkt_residual.KKT_residual.isZero());
  contact_complementarity_.setSlackAndDual(robot_, dtau_, s);
  contact_complementarity_.augmentDualResidual(robot_, dtau_, s, kkt_residual);
  KKTResidual kkt_residual_ref(robot_);
  force_inequality_.setSlack(robot_, dtau_, s, force_data_);
  baum_inequality_.setSlack(robot_, dtau_, s, baum_data_);
  for (int i=0; i<dimc_; ++i) {
    if (force_data_.slack.coeff(i) < barrier_) force_data_.slack.coeffRef(i) = barrier_;
    if (baum_data_.slack.coeff(i) < barrier_) baum_data_.slack.coeffRef(i) = barrier_;
  }
  force_data_.dual.array() 
      = barrier_ / force_data_.slack.array() 
          - baum_data_.slack.array() * complementarity_data_.dual.array();
  baum_data_.dual.array() 
      = barrier_ / baum_data_.slack.array() 
          - force_data_.slack.array() 
              * complementarity_data_.dual.array();
  for (int i=0; i<dimc_; ++i) {
    if (force_data_.dual.coeff(i) < barrier_) force_data_.dual.coeffRef(i) = barrier_;
    if (baum_data_.dual.coeff(i) < barrier_) baum_data_.dual.coeffRef(i) = barrier_;
  }
  Eigen::MatrixXd force_jac_df(Eigen::MatrixXd::Zero(dimc_, s.f.size()));
  force_jac_df(0, 0) = dtau_;
  force_jac_df(1, 1) = dtau_;
  force_jac_df(2, 2) = dtau_;
  force_jac_df(3, 3) = dtau_;
  force_jac_df(4, 4) = dtau_;
  force_jac_df(5, 0) = - 2 * dtau_ * s.f_3D(0);
  force_jac_df(5, 1) = 2 * dtau_ * s.f_3D(0);
  force_jac_df(5, 2) = - 2 * dtau_ * s.f_3D(1);
  force_jac_df(5, 3) = 2 * dtau_ * s.f_3D(1);
  force_jac_df(5, 4) = 2 * mu_ * mu_ * dtau_ * s.f_3D(2);
  Eigen::MatrixXd dbaum_dq(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_dv(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  Eigen::MatrixXd dbaum_da(Eigen::MatrixXd::Zero(3, robot_.dimv()));
  robot_.computeBaumgarteDerivatives(dbaum_dq, dbaum_dv, dbaum_da); 
  Eigen::MatrixXd baum_jac_da(Eigen::MatrixXd::Zero(dimc_, robot_.dimv()));
  Eigen::MatrixXd baum_jac_dr(Eigen::MatrixXd::Zero(dimc_, s.r.size()));
  Eigen::MatrixXd baum_jac_dq(Eigen::MatrixXd::Zero(dimc_, robot_.dimv()));
  Eigen::MatrixXd baum_jac_dv(Eigen::MatrixXd::Zero(dimc_, robot_.dimv()));
  baum_jac_da.row(0) =   dtau_ * dbaum_da.row(0);
  baum_jac_da.row(1) = - dtau_ * dbaum_da.row(0);
  baum_jac_da.row(2) =   dtau_ * dbaum_da.row(1);
  baum_jac_da.row(3) = - dtau_ * dbaum_da.row(1);
  baum_jac_da.row(4) =   dtau_ * dbaum_da.row(2);
  baum_jac_da.row(5).setZero();
  baum_jac_dr(0, 0) = dtau_;
  baum_jac_dr(1, 0) = dtau_;
  baum_jac_dr(2, 1) = dtau_;
  baum_jac_dr(3, 1) = dtau_;
  baum_jac_dr(5, 0) = 2 * dtau_ * s.r(0);
  baum_jac_dr(5, 1) = 2 * dtau_ * s.r(1);
  baum_jac_dq.row(0) =   dtau_ * dbaum_dq.row(0);
  baum_jac_dq.row(1) = - dtau_ * dbaum_dq.row(0);
  baum_jac_dq.row(2) =   dtau_ * dbaum_dq.row(1);
  baum_jac_dq.row(3) = - dtau_ * dbaum_dq.row(1);
  baum_jac_dq.row(4) =   dtau_ * dbaum_dq.row(2);
  baum_jac_dq.row(5).setZero();
  baum_jac_dv.row(0) =   dtau_ * dbaum_dv.row(0);
  baum_jac_dv.row(1) = - dtau_ * dbaum_dv.row(0);
  baum_jac_dv.row(2) =   dtau_ * dbaum_dv.row(1);
  baum_jac_dv.row(3) = - dtau_ * dbaum_dv.row(1);
  baum_jac_dv.row(4) =   dtau_ * dbaum_dv.row(2);
  baum_jac_dv.row(5).setZero();
  kkt_residual_ref.la() = - baum_jac_da.transpose() * baum_data_.dual;
  kkt_residual_ref.lf() = - force_jac_df.transpose() * force_data_.dual;
  kkt_residual_ref.lr() = - baum_jac_dr.transpose() * baum_data_.dual;
  kkt_residual_ref.lq() = - baum_jac_dq.transpose() * baum_data_.dual;
  kkt_residual_ref.lv() = - baum_jac_dv.transpose() * baum_data_.dual;
  EXPECT_TRUE(kkt_residual.KKT_residual.isApprox(kkt_residual_ref.KKT_residual));
}

TEST_F(FixedBaseContactComplementarityTest, computeResidual) {
  SplitSolution s(robot_);
  s.q = Eigen::VectorXd::Random(robot_.dimq());
  robot_.generateFeasibleConfiguration(s.q);
  s.v = Eigen::VectorXd::Random(robot_.dimv());
  s.a = Eigen::VectorXd::Random(robot_.dimv());
  robot_.updateKinematics(s.q, s.v, s.a);
  s.f = Eigen::VectorXd::Random(5*robot_.num_point_contacts()).array().abs();
  s.r = Eigen::VectorXd::Random(2*robot_.num_point_contacts()).array().abs();
  s.set_f_3D();
  contact_complementarity_.computeResidual(robot_, dtau_, s);
  const double L1Norm = contact_complementarity_.residualL1Nrom();
  const double squaredNorm = contact_complementarity_.squaredKKTErrorNorm();
  force_inequality_.computePrimalResidual(robot_, dtau_, s, force_data_);
  baum_inequality_.computePrimalResidual(robot_, dtau_, s, baum_data_);
  const double L1Norm_ref = force_data_.residual.lpNorm<1>() 
                              + baum_data_.residual.lpNorm<1>() 
                              + Eigen::VectorXd::Constant(dimc_, barrier_).lpNorm<1>() 
                              + Eigen::VectorXd::Constant(dimc_, barrier_).lpNorm<1>() 
                              + Eigen::VectorXd::Constant(dimc_, barrier_).lpNorm<1>() 
                              + Eigen::VectorXd::Constant(dimc_, max_complementarity_violation_).lpNorm<1>();
  const double squaredNorm_ref = force_data_.residual.squaredNorm() 
                                  + baum_data_.residual.squaredNorm() 
                                  + Eigen::VectorXd::Constant(dimc_, barrier_).squaredNorm() 
                                  + Eigen::VectorXd::Constant(dimc_, barrier_).squaredNorm() 
                                  + Eigen::VectorXd::Constant(dimc_, barrier_).squaredNorm() 
                                  + Eigen::VectorXd::Constant(dimc_, max_complementarity_violation_).squaredNorm();
  EXPECT_DOUBLE_EQ(L1Norm, L1Norm_ref);
  EXPECT_DOUBLE_EQ(squaredNorm, squaredNorm_ref);
}


} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}