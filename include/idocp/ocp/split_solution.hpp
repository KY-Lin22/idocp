#ifndef IDOCP_SPLIT_SOLUTION_HPP_
#define IDOCP_SPLIT_SOLUTION_HPP_

#include "Eigen/Core"

#include "idocp/robot/robot.hpp"


namespace idocp {

class SplitSolution {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SplitSolution(const Robot& robot);

  SplitSolution();

  ~SplitSolution();

  SplitSolution(const SplitSolution&) = default;

  SplitSolution& operator=(const SplitSolution&) = default;
 
  SplitSolution(SplitSolution&&) noexcept = default;

  SplitSolution& operator=(SplitSolution&&) noexcept = default;

  void set_f_3D();

  Eigen::VectorXd lmd, gmm, mu, a, f, f_3D, r, q, v, u, beta;

private:
  static constexpr int kDimf_3D = 3;
  static constexpr int kDimf = 5;
  static constexpr int kDimr = 2;
  int num_point_contacts_;

};

} // namespace idocp 

#include "idocp/ocp/split_solution.hxx"

#endif // IDOCP_SPLIT_SOLUTION_HPP_