#include <string>

#include <gtest/gtest.h>

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"
#include "idocp/ocp/split_solution.hpp"


namespace idocp {

class SplitSolutionTest : public ::testing::Test {
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


TEST_F(SplitSolutionTest, fixed_base) {
  Robot robot(fixed_base_urdf_);
  SplitSolution s(robot);
  EXPECT_EQ(s.lmd.size(), 7);
  EXPECT_EQ(s.gmm.size(), 7);
  EXPECT_EQ(s.mu.size(), 0);
  EXPECT_EQ(s.a.size(), 7);
  EXPECT_EQ(s.f.size(), 0);
  EXPECT_EQ(s.f_3D.size(), 0);
  EXPECT_EQ(s.r.size(), 0);
  EXPECT_EQ(s.q.size(), 7);
  EXPECT_EQ(s.v.size(), 7);
  EXPECT_EQ(s.u.size(), 7);
  EXPECT_EQ(s.beta.size(), 7);
  const Eigen::VectorXd lmd = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd gmm = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd mu = Eigen::VectorXd::Random(robot.dim_passive());
  const Eigen::VectorXd a = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd f = Eigen::VectorXd::Random(0);
  const Eigen::VectorXd f_3D = Eigen::VectorXd::Random(0);
  const Eigen::VectorXd r = Eigen::VectorXd::Random(0);
  const Eigen::VectorXd q = Eigen::VectorXd::Random(robot.dimq());
  const Eigen::VectorXd v = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd u = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(robot.dimv());
  s.lmd = lmd;
  s.gmm = gmm;
  s.mu = mu;
  s.a = a;
  s.f = f;
  s.f_3D = f_3D;
  s.r = r;
  s.q = q;
  s.v = v;
  s.u = u;
  s.beta = beta;
  EXPECT_TRUE(s.lmd.isApprox(lmd));
  EXPECT_TRUE(s.gmm.isApprox(gmm));
  EXPECT_TRUE(s.mu.isApprox(mu));
  EXPECT_TRUE(s.a.isApprox(a));
  EXPECT_TRUE(s.f.isApprox(f));
  EXPECT_TRUE(s.q.isApprox(q));
  EXPECT_TRUE(s.v.isApprox(v));
  EXPECT_TRUE(s.u.isApprox(u));
  EXPECT_TRUE(s.beta.isApprox(beta));
}


TEST_F(SplitSolutionTest, fixed_base_contact) {
  std::vector<int> contact_frames = {18};
  Robot robot(fixed_base_urdf_, contact_frames, 0, 0);
  SplitSolution s(robot);
  EXPECT_EQ(s.lmd.size(), 7);
  EXPECT_EQ(s.gmm.size(), 7);
  EXPECT_EQ(s.mu.size(), 0);
  EXPECT_EQ(s.a.size(), 7);
  EXPECT_EQ(s.f.size(), 5);
  EXPECT_EQ(s.f_3D.size(), 3);
  EXPECT_EQ(s.r.size(), 2);
  EXPECT_EQ(s.q.size(), 7);
  EXPECT_EQ(s.v.size(), 7);
  EXPECT_EQ(s.u.size(), 7);
  EXPECT_EQ(s.beta.size(), 7);
  const Eigen::VectorXd lmd = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd gmm = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd mu = Eigen::VectorXd::Random(robot.dim_passive());
  const Eigen::VectorXd a = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd f = Eigen::VectorXd::Random(5);
  const Eigen::VectorXd f_3D = Eigen::VectorXd::Random(3);
  const Eigen::VectorXd r = Eigen::VectorXd::Random(2);
  const Eigen::VectorXd q = Eigen::VectorXd::Random(robot.dimq());
  const Eigen::VectorXd v = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd u = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(robot.dimv());
  s.lmd = lmd;
  s.gmm = gmm;
  s.mu = mu;
  s.a = a;
  s.f = f;
  s.f_3D = f_3D;
  s.r = r;
  s.q = q;
  s.v = v;
  s.u = u;
  s.beta = beta;
  EXPECT_TRUE(s.lmd.isApprox(lmd));
  EXPECT_TRUE(s.gmm.isApprox(gmm));
  EXPECT_TRUE(s.mu.isApprox(mu));
  EXPECT_TRUE(s.a.isApprox(a));
  EXPECT_TRUE(s.f.isApprox(f));
  EXPECT_TRUE(s.q.isApprox(q));
  EXPECT_TRUE(s.v.isApprox(v));
  EXPECT_TRUE(s.u.isApprox(u));
  EXPECT_TRUE(s.beta.isApprox(beta));
  s.set_f_3D();
  Eigen::VectorXd f_3D_ref = Eigen::VectorXd::Zero(robot.dimf());
  f_3D_ref(0) = s.f(0) - s.f(1);
  f_3D_ref(1) = s.f(2) - s.f(3);
  f_3D_ref(2) = s.f(4);
  EXPECT_TRUE(s.f_3D.isApprox(f_3D_ref));
}


TEST_F(SplitSolutionTest, floating_base) {
  std::vector<int> contact_frames = {14, 24, 34, 44};
  Robot robot(floating_base_urdf_, contact_frames, 0, 0);
  SplitSolution s(robot);
  EXPECT_EQ(s.lmd.size(), 18);
  EXPECT_EQ(s.gmm.size(), 18);
  EXPECT_EQ(s.mu.size(), 6);
  EXPECT_EQ(s.a.size(), 18);
  EXPECT_EQ(s.f.size(), 20);
  EXPECT_EQ(s.f_3D.size(), 12);
  EXPECT_EQ(s.r.size(), 8);
  EXPECT_EQ(s.q.size(), 19);
  EXPECT_EQ(s.v.size(), 18);
  EXPECT_EQ(s.u.size(), 18);
  EXPECT_EQ(s.beta.size(), 18);
  const Eigen::VectorXd lmd = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd gmm = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd mu = Eigen::VectorXd::Random(robot.dim_passive());
  const Eigen::VectorXd a = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd f = Eigen::VectorXd::Random(20);
  const Eigen::VectorXd f_3D = Eigen::VectorXd::Random(12);
  const Eigen::VectorXd r = Eigen::VectorXd::Random(8);
  const Eigen::VectorXd q = Eigen::VectorXd::Random(robot.dimq());
  const Eigen::VectorXd v = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd u = Eigen::VectorXd::Random(robot.dimv());
  const Eigen::VectorXd beta = Eigen::VectorXd::Random(robot.dimv());
  s.lmd = lmd;
  s.gmm = gmm;
  s.mu = mu;
  s.a = a;
  s.f = f;
  s.f_3D = f_3D;
  s.r = r;
  s.q = q;
  s.v = v;
  s.u = u;
  s.beta = beta;
  EXPECT_TRUE(s.lmd.isApprox(lmd));
  EXPECT_TRUE(s.gmm.isApprox(gmm));
  EXPECT_TRUE(s.mu.isApprox(mu));
  EXPECT_TRUE(s.a.isApprox(a));
  EXPECT_TRUE(s.f.isApprox(f));
  EXPECT_TRUE(s.q.isApprox(q));
  EXPECT_TRUE(s.v.isApprox(v));
  EXPECT_TRUE(s.u.isApprox(u));
  EXPECT_TRUE(s.beta.isApprox(beta));
  s.set_f_3D();
  Eigen::VectorXd f_3D_ref = Eigen::VectorXd::Zero(robot.dimf());
  for (int i=0; i<robot.num_point_contacts(); ++i) {
    f_3D_ref(3*i  ) = s.f(5*i  ) - s.f(5*i+1);
    f_3D_ref(3*i+1) = s.f(5*i+2) - s.f(5*i+3);
    f_3D_ref(3*i+2) = s.f(5*i+4);
  }
  EXPECT_TRUE(s.f_3D.isApprox(f_3D_ref));
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}