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

  void setContactStatus(const Robot& robot);

  void set_f();

  int dimc() const;

  int dimf() const;

  Eigen::VectorXd lmd, gmm, mu, a, f, fxp, fxm, fyp, fym, fz, rx, ry, q, v, u, beta;

private:
  int dimc_, dimf_, max_point_contacts_;

};

} // namespace idocp 

#include "idocp/ocp/split_solution.hxx"

#endif // IDOCP_SPLIT_SOLUTION_HPP_