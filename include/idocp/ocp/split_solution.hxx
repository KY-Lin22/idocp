#ifndef IDOCP_SPLIT_SOLUTION_HXX_
#define IDOCP_SPLIT_SOLUTION_HXX_

#include "idocp/ocp/split_solution.hpp"

namespace idocp {

inline SplitSolution::SplitSolution(const Robot& robot) 
  : lmd(Eigen::VectorXd::Zero(robot.dimv())),
    gmm(Eigen::VectorXd::Zero(robot.dimv())),
    mu(Eigen::VectorXd::Zero(robot.dim_passive())),
    a(Eigen::VectorXd::Zero(robot.dimv())),
    f_3D(Eigen::VectorXd::Zero(kDimf_3D*robot.num_point_contacts())),
    f(Eigen::VectorXd::Zero(kDimf*robot.num_point_contacts())),
    r(Eigen::VectorXd::Zero(kDimr*robot.num_point_contacts())),
    q(Eigen::VectorXd::Zero(robot.dimq())),
    v(Eigen::VectorXd::Zero(robot.dimv())),
    u(Eigen::VectorXd::Zero(robot.dimv())),
    beta(Eigen::VectorXd::Zero(robot.dimv())),
    num_point_contacts_(robot.num_point_contacts()) {
  robot.normalizeConfiguration(q);
}


inline SplitSolution::SplitSolution() 
  : lmd(),
    gmm(),
    mu(),
    a(),
    f_3D(),
    f(),
    r(),
    q(),
    v(),
    u(),
    beta(),
    num_point_contacts_(0) {
}


inline SplitSolution::~SplitSolution() {
}


inline void SplitSolution::set_f_3D() {
  for (int i=0; i<num_point_contacts_; ++i) {
    f_3D.coeffRef(kDimf_3D*i  ) = f.coeff(kDimf*i  ) - f.coeff(kDimf*i+1);
    f_3D.coeffRef(kDimf_3D*i+1) = f.coeff(kDimf*i+2) - f.coeff(kDimf*i+3);
    f_3D.coeffRef(kDimf_3D*i+2) = f.coeff(kDimf*i+4);
  }
}

} // namespace idocp 

#endif // IDOCP_SPLIT_SOLUTION_HXX_