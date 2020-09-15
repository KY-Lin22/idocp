#ifndef IDOCP_CONSTRAINT_COMPONENT_DATA_HPP_
#define IDOCP_CONSTRAINT_COMPONENT_DATA_HPP_

#include <vector>
#include <exception>
#include <iostream>

#include "Eigen/Core"


namespace idocp {

class ConstraintComponentData {
public:
  ConstraintComponentData(const int dimc)
    : slack(Eigen::VectorXd::Zero(dimc)),
      dual(Eigen::VectorXd::Zero(dimc)),
      residual(Eigen::VectorXd::Zero(dimc)),
      duality(Eigen::VectorXd::Zero(dimc)),
      dslack(Eigen::VectorXd::Zero(dimc)),
      ddual(Eigen::VectorXd::Zero(dimc)),
      dimc_(dimc) {
    try {
      if (dimc < 0) {
        throw std::out_of_range("invalid argment: dimc must not be negative");
      }
    }
    catch(const std::exception& e) {
      std::cerr << e.what() << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  ConstraintComponentData()
    : slack(),
      dual(),
      residual(),
      duality(),
      dslack(),
      ddual(),
      dimc_(0) {
  }

  ~ConstraintComponentData() {
  }

  // Use default copy constructor.
  ConstraintComponentData(const ConstraintComponentData&) = default;

  // Use default copy coperator.
  ConstraintComponentData& operator=(const ConstraintComponentData&) = default;

  // Use default move constructor.
  ConstraintComponentData(ConstraintComponentData&&) noexcept = default;

  // Use default move assign coperator.
  ConstraintComponentData& operator=(ConstraintComponentData&&) noexcept 
      = default;

  int dimc() const {
    return dimc_;
  }

  Eigen::VectorXd slack, dual, residual, duality, dslack, ddual;

private:
  int dimc_;

};

} // namespace idocp


#endif // IDOCP_CONSTRAINT_COMPONENT_DATA_HPP_