#ifndef IDOCP_SPLIT_SOLUTION_HXX_
#define IDOCP_SPLIT_SOLUTION_HXX_

namespace idocp {

inline SplitSolution::SplitSolution(const Robot& robot) 
  : lmd(Eigen::VectorXd::Zero(robot.dimv())),
    gmm(Eigen::VectorXd::Zero(robot.dimv())),
    mu(Eigen::VectorXd::Zero(robot.dim_passive())),
    a(Eigen::VectorXd::Zero(robot.dimv())),
    f(Eigen::VectorXd::Zero(3*robot.max_point_contacts())),
    fxp(Eigen::VectorXd::Zero(robot.max_point_contacts())),
    fxm(Eigen::VectorXd::Zero(robot.max_point_contacts())),
    fyp(Eigen::VectorXd::Zero(robot.max_point_contacts())),
    fym(Eigen::VectorXd::Zero(robot.max_point_contacts())),
    fz(Eigen::VectorXd::Zero(robot.max_point_contacts())),
    vx(Eigen::VectorXd::Zero(robot.max_point_contacts())),
    vy(Eigen::VectorXd::Zero(robot.max_point_contacts())),
    q(Eigen::VectorXd::Zero(robot.dimq())),
    v(Eigen::VectorXd::Zero(robot.dimv())),
    u(Eigen::VectorXd::Zero(robot.dimv())),
    beta(Eigen::VectorXd::Zero(robot.dimv())),
    dimc_(robot.dim_passive()),
    dimf_(7*robot.max_point_contacts()) {
  robot.normalizeConfiguration(q);
}


inline SplitSolution::SplitSolution() 
  : lmd(),
    gmm(),
    mu(),
    a(),
    f(),
    q(),
    v(),
    u(),
    beta(),
    dimc_(0),
    dimf_(0) {
}


inline SplitSolution::~SplitSolution() {
}


inline void SplitSolution::setContactStatus(const Robot& robot) {
  dimc_ = robot.dim_passive();
  dimf_ = robot.dimf();
}


inline void SplitSolution::set_f() {
  for (int i=0; i<max_point_contacts_; ++i) {
    f.coeffRef(i*3  ) = fxp.coeff(i) - fxm.coeff(i);
    f.coeffRef(i*3+1) = fyp.coeff(i) - fym.coeff(i);
    f.coeffRef(i*3+2) = fz.coeff(i);
  }
}


inline int SplitSolution::dimc() const {
  return dimc_;
}


inline int SplitSolution::dimf() const {
  return dimf_;
}

} // namespace idocp 

#endif // IDOCP_SPLIT_SOLUTION_HXX_