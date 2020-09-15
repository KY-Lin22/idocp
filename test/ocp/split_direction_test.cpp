#include <string>

#include <gtest/gtest.h>

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/split_direction.hpp"


namespace idocp {

class SplitDirectionTest : public ::testing::Test {
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


TEST_F(SplitDirectionTest, fixed_base) {
  Robot robot(fixed_base_urdf_);
  SplitDirection d(robot);
  EXPECT_EQ(d.dimKKT(), 5*robot.dimv()+robot.dim_passive()+7*robot.num_point_contacts());
  EXPECT_EQ(d.dimf(), 5*robot.num_point_contacts());
  EXPECT_EQ(d.dimr(), 2*robot.num_point_contacts());
  const Eigen::VectorXd split_direction = Eigen::VectorXd::Random(d.dimKKT());
  d.split_direction = split_direction;
  const Eigen::VectorXd dlmd = split_direction.segment(                                                   0, robot.dimv());
  const Eigen::VectorXd dgmm = split_direction.segment(                                        robot.dimv(), robot.dimv());
  const Eigen::VectorXd dmu =  split_direction.segment(                                      2*robot.dimv(), robot.dim_passive());
  const Eigen::VectorXd da =   split_direction.segment(                  2*robot.dimv()+robot.dim_passive(), robot.dimv());
  const Eigen::VectorXd df =   split_direction.segment(                  3*robot.dimv()+robot.dim_passive(), d.dimf());
  const Eigen::VectorXd dr =   split_direction.segment(         3*robot.dimv()+robot.dim_passive()+d.dimf(), d.dimr());
  const Eigen::VectorXd dfr =  split_direction.segment(                  3*robot.dimv()+robot.dim_passive(), d.dimf()+d.dimr());
  const Eigen::VectorXd dq =   split_direction.segment(3*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), robot.dimv());
  const Eigen::VectorXd dv =   split_direction.segment(4*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), robot.dimv());
  const Eigen::VectorXd dx =   split_direction.segment(3*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), 2*robot.dimv());
  EXPECT_TRUE(dlmd.isApprox(d.dlmd()));
  EXPECT_TRUE(dgmm.isApprox(d.dgmm()));
  EXPECT_TRUE(dmu.isApprox(d.dmu()));
  EXPECT_TRUE(da.isApprox(d.da()));
  EXPECT_TRUE(df.isApprox(d.df()));
  EXPECT_TRUE(dr.isApprox(d.dr()));
  EXPECT_TRUE(dfr.isApprox(d.dfr()));
  EXPECT_TRUE(dq.isApprox(d.dq()));
  EXPECT_TRUE(dv.isApprox(d.dv()));
  EXPECT_TRUE(dx.isApprox(d.dx()));
  d.setZero();
  EXPECT_TRUE(d.split_direction.isZero());
}


TEST_F(SplitDirectionTest, fixed_base_contact) {
  std::vector<int> contact_frames = {18};
  Robot robot(fixed_base_urdf_, contact_frames, 0, 0);
  SplitDirection d(robot);
  EXPECT_EQ(d.dimKKT(), 5*robot.dimv()+robot.dim_passive()+7*robot.num_point_contacts());
  EXPECT_EQ(d.dimf(), 5*robot.num_point_contacts());
  EXPECT_EQ(d.dimr(), 2*robot.num_point_contacts());
  const Eigen::VectorXd split_direction = Eigen::VectorXd::Random(d.dimKKT());
  d.split_direction = split_direction;
  const Eigen::VectorXd dlmd = split_direction.segment(                                                   0, robot.dimv());
  const Eigen::VectorXd dgmm = split_direction.segment(                                        robot.dimv(), robot.dimv());
  const Eigen::VectorXd dmu =  split_direction.segment(                                      2*robot.dimv(), robot.dim_passive());
  const Eigen::VectorXd da =   split_direction.segment(                  2*robot.dimv()+robot.dim_passive(), robot.dimv());
  const Eigen::VectorXd df =   split_direction.segment(                  3*robot.dimv()+robot.dim_passive(), d.dimf());
  const Eigen::VectorXd dr =   split_direction.segment(         3*robot.dimv()+robot.dim_passive()+d.dimf(), d.dimr());
  const Eigen::VectorXd dfr =  split_direction.segment(                  3*robot.dimv()+robot.dim_passive(), d.dimf()+d.dimr());
  const Eigen::VectorXd dq =   split_direction.segment(3*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), robot.dimv());
  const Eigen::VectorXd dv =   split_direction.segment(4*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), robot.dimv());
  const Eigen::VectorXd dx =   split_direction.segment(3*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), 2*robot.dimv());
  EXPECT_TRUE(dlmd.isApprox(d.dlmd()));
  EXPECT_TRUE(dgmm.isApprox(d.dgmm()));
  EXPECT_TRUE(dmu.isApprox(d.dmu()));
  EXPECT_TRUE(da.isApprox(d.da()));
  EXPECT_TRUE(df.isApprox(d.df()));
  EXPECT_TRUE(dr.isApprox(d.dr()));
  EXPECT_TRUE(dfr.isApprox(d.dfr()));
  EXPECT_TRUE(dq.isApprox(d.dq()));
  EXPECT_TRUE(dv.isApprox(d.dv()));
  EXPECT_TRUE(dx.isApprox(d.dx()));
  d.setZero();
  EXPECT_TRUE(d.split_direction.isZero());
}


TEST_F(SplitDirectionTest, floating_base) {
  std::vector<int> contact_frames = {14, 24, 34, 44};
  Robot robot(floating_base_urdf_, contact_frames, 0, 0);
  SplitDirection d(robot);
  EXPECT_EQ(d.dimKKT(), 5*robot.dimv()+robot.dim_passive()+7*robot.num_point_contacts());
  EXPECT_EQ(d.dimf(), 5*robot.num_point_contacts());
  EXPECT_EQ(d.dimr(), 2*robot.num_point_contacts());
  const Eigen::VectorXd split_direction = Eigen::VectorXd::Random(d.dimKKT());
  d.split_direction = split_direction;
  const Eigen::VectorXd dlmd = split_direction.segment(                                                   0, robot.dimv());
  const Eigen::VectorXd dgmm = split_direction.segment(                                        robot.dimv(), robot.dimv());
  const Eigen::VectorXd dmu =  split_direction.segment(                                      2*robot.dimv(), robot.dim_passive());
  const Eigen::VectorXd da =   split_direction.segment(                  2*robot.dimv()+robot.dim_passive(), robot.dimv());
  const Eigen::VectorXd df =   split_direction.segment(                  3*robot.dimv()+robot.dim_passive(), d.dimf());
  const Eigen::VectorXd dr =   split_direction.segment(         3*robot.dimv()+robot.dim_passive()+d.dimf(), d.dimr());
  const Eigen::VectorXd dfr =  split_direction.segment(                  3*robot.dimv()+robot.dim_passive(), d.dimf()+d.dimr());
  const Eigen::VectorXd dq =   split_direction.segment(3*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), robot.dimv());
  const Eigen::VectorXd dv =   split_direction.segment(4*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), robot.dimv());
  const Eigen::VectorXd dx =   split_direction.segment(3*robot.dimv()+robot.dim_passive()+d.dimf()+d.dimr(), 2*robot.dimv());
  EXPECT_TRUE(dlmd.isApprox(d.dlmd()));
  EXPECT_TRUE(dgmm.isApprox(d.dgmm()));
  EXPECT_TRUE(dmu.isApprox(d.dmu()));
  EXPECT_TRUE(da.isApprox(d.da()));
  EXPECT_TRUE(df.isApprox(d.df()));
  EXPECT_TRUE(dr.isApprox(d.dr()));
  EXPECT_TRUE(dfr.isApprox(d.dfr()));
  EXPECT_TRUE(dq.isApprox(d.dq()));
  EXPECT_TRUE(dv.isApprox(d.dv()));
  EXPECT_TRUE(dx.isApprox(d.dx()));
  d.setZero();
  EXPECT_TRUE(d.split_direction.isZero());
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}