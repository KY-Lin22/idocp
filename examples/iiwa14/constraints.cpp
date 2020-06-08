#include "constraints.hpp"


namespace invdynocp {
namespace iiwa14 {

Constraints::Constraints(const Robot* robot_ptr)
  : ConstraintsInterface(),
    configuration_space_constraints_(robot_ptr) {
}


void Constraints::C(const Robot* robot_ptr, const double t, const double dtau, 
                    const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                    const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                    Eigen::VectorXd& C) {
}


void Constraints::Cq(const Robot* robot_ptr, const double t, const double dtau,
                     const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                     const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                     Eigen::MatrixXd& Cq) {
}


void Constraints::Cq(const Robot* robot_ptr, const double t, const double dtau,
                     const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                     const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                     const Eigen::VectorXd& fext, 
                     Eigen::MatrixXd& Cq) {
}


void Constraints::Cv(const Robot* robot_ptr, const double t, const double dtau,
                     const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                     const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                     Eigen::MatrixXd& Cv) {
}


void Constraints::Cv(const Robot* robot_ptr, const double t, const double dtau,
                     const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                     const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                     const Eigen::VectorXd& fext, 
                     Eigen::MatrixXd& Cv) {
}


void Constraints::Ca(const Robot* robot_ptr, const double t, const double dtau,
                     const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                     const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                     Eigen::MatrixXd& Cv) {
}


void Constraints::Ca(const Robot* robot_ptr, const double t, const double dtau,
                     const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                     const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                     const Eigen::VectorXd& fext, 
                     Eigen::MatrixXd& Ca) {
}


void Constraints::Cu(const Robot* robot_ptr, const double t, const double dtau,
                     const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                     const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                     Eigen::MatrixXd& Cu) {
}


void Constraints::Cu(const Robot* robot_ptr, const double t, const double dtau,
                     const Eigen::VectorXd& q, const Eigen::VectorXd& v, 
                     const Eigen::VectorXd& a, const Eigen::VectorXd& u, 
                     const Eigen::VectorXd& fext, 
                     Eigen::MatrixXd& Cu) {
}


void Constraints::Cfext(const Robot* robot_ptr, const double t, 
                        const double dtau, const Eigen::VectorXd& q, 
                        const Eigen::VectorXd& v, const Eigen::VectorXd& a, 
                        const Eigen::VectorXd& u, const Eigen::VectorXd& fext, 
                        Eigen::MatrixXd& Cfext) {
}

unsigned int Constraints::dimc() const {
  return 0;
}

} // namespace iiwa14
} // namespace invdynocp