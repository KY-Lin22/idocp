#include <string>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "Eigen/Core"

#include "idocp/complementarity/complementarity_constraint.hpp"
#include "idocp/constraints/constraint_component_data.hpp"

namespace idocp {

class ComplementarityConstraintTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    dimc = 10;
    max_complementarity_violation = 1.0e-02;
    barrier = 1.0e-04;
    fraction_to_boundary_rate = 0.995;
    complementarity = ComplementarityConstraint(dimc, 
                                                max_complementarity_violation, 
                                                barrier, 
                                                fraction_to_boundary_rate);
    data1 = ConstraintComponentData(dimc);
    data2 = ConstraintComponentData(dimc);
    datacc = ConstraintComponentData(dimc);
  }

  virtual void TearDown() {
  }

  int dimc;
  double max_complementarity_violation, barrier, fraction_to_boundary_rate;
  ComplementarityConstraint complementarity;
  ConstraintComponentData data1, data2, datacc;
};


TEST_F(ComplementarityConstraintTest, isFeasible) {
  data1.slack = Eigen::VectorXd::Constant(dimc, 1.1*std::sqrt(max_complementarity_violation));
  data2.slack = Eigen::VectorXd::Constant(dimc, 1.1*std::sqrt(max_complementarity_violation));
  EXPECT_FALSE(complementarity.isFeasible(data1, data2));
  data1.slack = Eigen::VectorXd::Constant(dimc, 0.9*std::sqrt(max_complementarity_violation));
  data2.slack = Eigen::VectorXd::Constant(dimc, 0.9*std::sqrt(max_complementarity_violation));
  EXPECT_TRUE(complementarity.isFeasible(data1, data2));
}


TEST_F(ComplementarityConstraintTest, setSlackAndDual) {
  complementarity.setSlackAndDual(data1, data2, datacc);
  EXPECT_TRUE(data1.slack.isApprox(Eigen::VectorXd::Constant(dimc, barrier)));
  EXPECT_TRUE(data2.slack.isApprox(Eigen::VectorXd::Constant(dimc, barrier)));
  ConstraintComponentData datacc_ref(dimc);
  datacc_ref.slack.array() = max_complementarity_violation - barrier * barrier;
  EXPECT_TRUE(datacc_ref.slack.isApprox(datacc.slack));
  datacc_ref.dual.array() = barrier / datacc_ref.slack.array();
  EXPECT_TRUE(datacc_ref.dual.isApprox(datacc.dual));
  ConstraintComponentData data1_ref(dimc), data2_ref(dimc);
  data1_ref.dual.array() = barrier / data1.slack.array() - data2.slack.array() * datacc.dual.array();
  data2_ref.dual.array() = barrier / data2.slack.array() - data1.slack.array() * datacc.dual.array();
  EXPECT_TRUE(data1_ref.dual.isApprox(data1.dual));
  EXPECT_TRUE(data2_ref.dual.isApprox(data2.dual));
}


TEST_F(ComplementarityConstraintTest, computeComplementarityResidual) {
  data1.slack = Eigen::VectorXd::Random(dimc).array().abs();
  data2.slack = Eigen::VectorXd::Random(dimc).array().abs();
  datacc.slack = Eigen::VectorXd::Random(dimc).array().abs();
  complementarity.computeComplementarityResidual(data1, data2, datacc);
  ConstraintComponentData datacc_ref(dimc);
  datacc_ref.residual.array() = datacc.slack.array() 
                                  + data1.slack.array() * data2.slack.array() 
                                  - max_complementarity_violation;
  EXPECT_TRUE(datacc_ref.residual.isApprox(datacc.residual));
}


TEST_F(ComplementarityConstraintTest, computeDualities) {
  data1.slack = Eigen::VectorXd::Random(dimc).array().abs();
  data2.slack = Eigen::VectorXd::Random(dimc).array().abs();
  datacc.slack = Eigen::VectorXd::Random(dimc).array().abs();
  data1.residual = Eigen::VectorXd::Random(dimc);
  data2.residual = Eigen::VectorXd::Random(dimc);
  datacc.residual = Eigen::VectorXd::Random(dimc);
  complementarity.computeDualities(data1, data2, datacc);
  ConstraintComponentData data1_ref(dimc), data2_ref(dimc), datacc_ref(dimc);
  data1_ref.duality.array()
      = data1.slack.array() * data1.dual.array()
          + data1.slack.array() * data2.slack.array() * datacc.dual.array() - barrier;
  data2_ref.duality.array()
      = data2.slack.array() * data2.dual.array()
          + data1.slack.array() * data2.slack.array() * datacc.dual.array() - barrier;
  datacc_ref.duality.array() = datacc.slack.array() * datacc.dual.array() - barrier;
  EXPECT_TRUE(data1.duality.isApprox(data1_ref.duality));
  EXPECT_TRUE(data2.duality.isApprox(data2_ref.duality));
  EXPECT_TRUE(datacc.duality.isApprox(datacc_ref.duality));
}


TEST_F(ComplementarityConstraintTest, condenseSlackAndDual) {
  data1.slack = Eigen::VectorXd::Random(dimc).array().abs();
  data2.slack = Eigen::VectorXd::Random(dimc).array().abs();
  datacc.slack = Eigen::VectorXd::Random(dimc).array().abs();
  data1.residual = Eigen::VectorXd::Random(dimc);
  data2.residual = Eigen::VectorXd::Random(dimc);
  datacc.residual = Eigen::VectorXd::Random(dimc);
  data1.dual = Eigen::VectorXd::Random(dimc).array().abs();
  data2.dual = Eigen::VectorXd::Random(dimc).array().abs();
  datacc.dual = Eigen::VectorXd::Random(dimc).array().abs();
  data1.duality = Eigen::VectorXd::Random(dimc);
  data2.duality = Eigen::VectorXd::Random(dimc);
  datacc.duality = Eigen::VectorXd::Random(dimc);
  Eigen::VectorXd diagonal11 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd diagonal12 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd diagonal22 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd dual1 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd dual2 = Eigen::VectorXd::Zero(dimc);
  complementarity.condenseSlackAndDual(data1, data2, datacc, diagonal11, 
                                       diagonal12, diagonal22, dual1, dual2);
  Eigen::VectorXd Sigma1 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd Sigma2 = Eigen::VectorXd::Zero(dimc);
  Sigma1.array() = (data1.dual.array() + data2.slack.array() * datacc.dual.array()) / data1.slack.array();
  Sigma2.array() = (data2.dual.array() + data1.slack.array() * datacc.dual.array()) / data2.slack.array();
  Eigen::VectorXd Lambda11 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd Lambda12 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd Lambda22 = Eigen::VectorXd::Zero(dimc);
  Lambda11.array() = data1.slack.array() * datacc.dual.array() * data1.slack.array() / datacc.slack.array();
  Lambda12.array() = data1.slack.array() * datacc.dual.array() * data2.slack.array() / datacc.slack.array();
  Lambda22.array() = data2.slack.array() * datacc.dual.array() * data2.slack.array() / datacc.slack.array();
  Eigen::VectorXd diagonal11_ref = Sigma1 + Lambda22;
  Eigen::VectorXd diagonal12_ref = datacc.dual + Lambda12;
  Eigen::VectorXd diagonal22_ref = Sigma2 + Lambda11;
  EXPECT_TRUE(diagonal11.isApprox(diagonal11_ref));
  EXPECT_TRUE(diagonal12.isApprox(diagonal12_ref));
  EXPECT_TRUE(diagonal22.isApprox(diagonal22_ref));
  Eigen::VectorXd dcc = Eigen::VectorXd::Zero(dimc);
  dcc = datacc.dual.array() / datacc.slack.array();
  Eigen::VectorXd dual1_ref = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd dual2_ref = Eigen::VectorXd::Zero(dimc);
  dual1_ref.array()
      = diagonal11_ref.array() * data1.residual.array() 
          + diagonal12_ref.array() * data2.residual.array() 
          - data2.slack.array() * dcc.array() * datacc.residual.array()
          + data2.slack.array() * datacc.duality.array() / datacc.slack.array()
          - data1.duality.array() / data1.slack.array();
  dual2_ref.array()
      = diagonal12_ref.array() * data1.residual.array() 
          + diagonal22_ref.array() * data2.residual.array() 
          - data1.slack.array() * dcc.array() * datacc.residual.array()
          + data1.slack.array() * datacc.duality.array() / datacc.slack.array()
          - data2.duality.array() / data2.slack.array();
  EXPECT_TRUE(dual1.isApprox(dual1_ref));
  EXPECT_TRUE(dual2.isApprox(dual2_ref));
}


TEST_F(ComplementarityConstraintTest, computeDirections) {
  data1.slack = Eigen::VectorXd::Random(dimc).array().abs();
  data2.slack = Eigen::VectorXd::Random(dimc).array().abs();
  datacc.slack = Eigen::VectorXd::Random(dimc).array().abs();
  data1.residual = Eigen::VectorXd::Random(dimc);
  data2.residual = Eigen::VectorXd::Random(dimc);
  datacc.residual = Eigen::VectorXd::Random(dimc);
  data1.dual = Eigen::VectorXd::Random(dimc).array().abs();
  data2.dual = Eigen::VectorXd::Random(dimc).array().abs();
  datacc.dual = Eigen::VectorXd::Random(dimc).array().abs();
  data1.duality = Eigen::VectorXd::Random(dimc);
  data2.duality = Eigen::VectorXd::Random(dimc);
  datacc.duality = Eigen::VectorXd::Random(dimc);
  data1.dslack = Eigen::VectorXd::Random(dimc).array();
  data2.dslack = Eigen::VectorXd::Random(dimc).array();
  Eigen::VectorXd diagonal11 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd diagonal12 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd diagonal22 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd dual1 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd dual2 = Eigen::VectorXd::Zero(dimc);
  complementarity.condenseSlackAndDual(data1, data2, datacc, diagonal11, 
                                       diagonal12, diagonal22, dual1, dual2);
  complementarity.computeDirections(data1, data2, datacc);
  Eigen::VectorXd Sigma1 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd Sigma2 = Eigen::VectorXd::Zero(dimc);
  Sigma1.array() = (data1.dual.array() + data2.slack.array() * datacc.dual.array()) / data1.slack.array();
  Sigma2.array() = (data2.dual.array() + data1.slack.array() * datacc.dual.array()) / data2.slack.array();
  Eigen::VectorXd dslackcc = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd ddualcc = Eigen::VectorXd::Zero(dimc);
  dslackcc.array() = - data1.slack.array() * data2.dslack.array() 
                     - data2.slack.array() * data1.dslack.array() 
                     - datacc.residual.array();
  ddualcc.array() = - datacc.dual.array() * datacc.dslack.array() / datacc.slack.array()
                    - datacc.duality.array() / datacc.slack.array();
  EXPECT_TRUE(datacc.dslack.isApprox(dslackcc));
  EXPECT_TRUE(datacc.ddual.isApprox(ddualcc));
  Eigen::VectorXd ddual1 = Eigen::VectorXd::Zero(dimc);
  Eigen::VectorXd ddual2 = Eigen::VectorXd::Zero(dimc);
  ddual1.array() = - Sigma1.array() * data1.dslack.array() 
                   - datacc.dual.array() * data2.dslack.array()
                   - data2.slack.array() * datacc.ddual.array()
                   - data1.duality.array() / data1.slack.array();
  ddual2.array() = - datacc.dual.array() * data1.dslack.array()
                   - Sigma2.array() * data2.dslack.array() 
                   - data1.slack.array() * datacc.ddual.array()
                   - data2.duality.array() / data2.slack.array();
  EXPECT_TRUE(data1.ddual.isApprox(ddual1));
  EXPECT_TRUE(data2.ddual.isApprox(ddual2));
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}