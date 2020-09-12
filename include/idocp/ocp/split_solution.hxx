#ifndef IDOCP_SPLIT_SOLUTION_HXX_
#define IDOCP_SPLIT_SOLUTION_HXX_

namespace idocp {

inline SplitSolution::SplitSolution(const Robot& robot) 
  : lmd(Eigen::VectorXd::Zero(robot.dimv())),
    gmm(Eigen::VectorXd::Zero(robot.dimv())),
    mu(Eigen::VectorXd::Zero(robot.dim_passive())),
    a(Eigen::VectorXd::Zero(robot.dimv())),
    f(Eigen::VectorXd::Zero(3*robot.num_point_contacts())),
    f_verbose(Eigen::VectorXd::Zero(7*robot.num_point_contacts())),
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
    f(),
    f_verbose(),
    q(),
    v(),
    u(),
    beta(),
    num_point_contacts_(0) {
}


inline SplitSolution::~SplitSolution() {
}


inline void SplitSolution::set_f() {
  for (int i=0; i<num_point_contacts_; ++i) {
    f.coeffRef(i*3  ) = f_verbose.coeff(i*7  ) - f_verbose.coeff(i*7+1);
    f.coeffRef(i*3+1) = f_verbose.coeff(i*7+2) - f_verbose.coeff(i*7+3);
    f.coeffRef(i*3+2) = f_verbose.coeff(i*7+4);
  }
}

} // namespace idocp 

#endif // IDOCP_SPLIT_SOLUTION_HXX_