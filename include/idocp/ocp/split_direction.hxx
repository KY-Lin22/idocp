#ifndef IDOCP_SPLIT_DIRECTION_HXX_
#define IDOCP_SPLIT_DIRECTION_HXX_

#include "idocp/ocp/split_direction.hpp"

namespace idocp {

inline SplitDirection::SplitDirection(const Robot& robot) 
  : du(robot.dimv()),
    dbeta(robot.dimv()),
    split_direction(Eigen::VectorXd::Zero(
        5*robot.dimv()+robot.dim_passive()+7*robot.num_point_contacts())),
    dimv_(robot.dimv()), 
    dimx_(2*robot.dimv()), 
    num_point_contacts_(robot.num_point_contacts()),
    dimf_verbose_(7*robot.num_point_contacts()), 
    dimc_(robot.dim_passive()),
    dimKKT_(5*robot.dimv()+robot.dim_passive()+7*robot.num_point_contacts()) {
}


inline SplitDirection::SplitDirection() 
  : du(),
    dbeta(),
    split_direction(),
    dimv_(0), 
    dimx_(0), 
    num_point_contacts_(0),
    dimf_verbose_(0), 
    dimc_(0),
    dimKKT_(0) {
}


inline SplitDirection::~SplitDirection() {
}


inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dlmd() {
  return split_direction.head(dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dgmm() {
  return split_direction.segment(dimv_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dmu() {
  return split_direction.segment(dimx_, dimc_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::da() {
  return split_direction.segment(dimx_+dimc_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::df() {
  return split_direction.segment(dimx_+dimc_+dimv_, dimf_verbose_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dq() {
  return split_direction.segment(dimx_+dimc_+dimv_+dimf_verbose_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dv() {
  return split_direction.segment(dimx_+dimc_+2*dimv_+dimf_verbose_, dimv_);
}


inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dx() {
  return split_direction.segment(dimx_+dimc_+dimv_+dimf_verbose_, dimx_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
SplitDirection::dlmd() const {
  return split_direction.head(dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
SplitDirection::dgmm() const {
  return split_direction.segment(dimv_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
SplitDirection::dmu() const {
  return split_direction.segment(dimx_, dimc_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
SplitDirection::da() const {
  return split_direction.segment(dimx_+dimc_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
SplitDirection::df() const {
  return split_direction.segment(dimx_+dimc_+dimv_, dimf_verbose_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
SplitDirection::dq() const {
  return split_direction.segment(dimx_+dimc_+dimv_+dimf_verbose_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
SplitDirection::dv() const {
  return split_direction.segment(dimx_+dimc_+2*dimv_+dimf_verbose_, dimv_);
}


inline const Eigen::VectorBlock<const Eigen::VectorXd> 
SplitDirection::dx() const {
  return split_direction.segment(dimx_+dimc_+dimv_+dimf_verbose_, dimx_);
}


inline void SplitDirection::setZero() {
  split_direction.setZero();
}


inline int SplitDirection::dimKKT() const {
  return dimKKT_;
}


inline int SplitDirection::dimc() const {
  return dimc_;
}


inline int SplitDirection::dimf_verbose() const {
  return dimf_verbose_;
}

} // namespace idocp 

#endif // IDOCP_SPLIT_DIRECTION_HXX_ 