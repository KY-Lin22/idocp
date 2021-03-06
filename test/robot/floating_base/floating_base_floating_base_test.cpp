#include <string>
#include <random>
#include <utility>
#include <vector>
#include <memory>

#include <gtest/gtest.h>
#include "Eigen/Core"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"

#include "idocp/robot/floating_base.hpp"


namespace idocp {

class FloatingBaseFloatingBaseTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    std::random_device rnd;
    urdf_ = "../../urdf/anymal/anymal.urdf";
    pinocchio::urdf::buildModel(urdf_, model_);
    data_ = pinocchio::Data(model_);
    dimq_ = model_.nq;
    dimv_ = model_.nv;
    u_ = Eigen::VectorXd::Random(dimv_);
  }

  virtual void TearDown() {
  }

  std::string urdf_;
  pinocchio::Model model_;
  pinocchio::Data data_;
  int dimq_, dimv_;
  Eigen::VectorXd u_;
};


TEST_F(FloatingBaseFloatingBaseTest, constructor) {
  FloatingBase floating_base(model_);
  EXPECT_EQ(floating_base.dim_passive(), 6);
  EXPECT_FALSE(floating_base.passive_joint_indices().empty());
  for (int i=0; i<floating_base.dim_passive(); ++i) {
    EXPECT_EQ(floating_base.passive_joint_indices()[i], i);
  }
  EXPECT_TRUE(floating_base.has_floating_base());
}


TEST_F(FloatingBaseFloatingBaseTest, moveAssign) {
  FloatingBase floating_base(model_);
  FloatingBase floating_base_ref = std::move(floating_base);
  EXPECT_EQ(floating_base_ref.dim_passive(), 6);
  EXPECT_FALSE(floating_base_ref.passive_joint_indices().empty());
  for (int i=0; i<floating_base_ref.dim_passive(); ++i) {
    EXPECT_EQ(floating_base_ref.passive_joint_indices()[i], i);
  }
  EXPECT_TRUE(floating_base_ref.has_floating_base());
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}