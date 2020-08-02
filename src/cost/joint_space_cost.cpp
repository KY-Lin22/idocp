#include "cost/joint_space_cost.hpp"

#include <assert.h>


namespace idocp {

JointSpaceCost::JointSpaceCost(const Robot& robot, 
                               const Eigen::VectorXd& q_weight,  
                               const Eigen::VectorXd& v_weight,  
                               const Eigen::VectorXd& a_weight,  
                               const Eigen::VectorXd& u_weight,
                               const Eigen::VectorXd& qf_weight,  
                               const Eigen::VectorXd& vf_weight)
  : has_floating_base_(robot.has_floating_base()),
    dimq_(robot.dimq()),
    dimv_(robot.dimv()),
    q_ref_(Eigen::VectorXd::Zero(robot.dimq())),
    v_ref_(Eigen::VectorXd::Zero(robot.dimv())),
    a_ref_(Eigen::VectorXd::Zero(robot.dimv())),
    u_ref_(Eigen::VectorXd::Zero(robot.dimv())),
    q_weight_(q_weight),
    v_weight_(v_weight),
    a_weight_(a_weight),
    u_weight_(u_weight),
    qf_weight_(qf_weight),
    vf_weight_(vf_weight),
    lq_configuration_(),
    lqq_configuration_(),
    phiqq_configuration_(),
    configuration_Jacobian_() {
  assert(q_weight.size() == dimq_);
  assert(v_weight.size() == dimv_);
  assert(a_weight.size() == dimv_);
  assert(u_weight.size() == dimv_);
  assert(qf_weight.size() == dimq_);
  assert(vf_weight.size() == dimv_);
  if (robot.has_floating_base()) {
    lq_configuration_.resize(robot.dimq());
    lq_configuration_.setZero();
    lqq_configuration_.resize(robot.dimq(), robot.dimq());
    lqq_configuration_.setZero();
    for (int i=0; i<robot.dimq(); ++i) {
      lqq_configuration_.coeffRef(i, i) = q_weight.coeff(i);
    }
    phiqq_configuration_.resize(robot.dimq(), robot.dimq());
    phiqq_configuration_.setZero();
    for (int i=0; i<robot.dimq(); ++i) {
      phiqq_configuration_.coeffRef(i, i) = qf_weight.coeff(i);
    }
    configuration_Jacobian_.resize(robot.dimq(), robot.dimv());
    configuration_Jacobian_.setZero();
  }
}


JointSpaceCost::JointSpaceCost(const Robot& robot, const Eigen::VectorXd& q_ref,  
                               const Eigen::VectorXd& v_ref, 
                               const Eigen::VectorXd& a_ref,  
                               const Eigen::VectorXd& u_ref, 
                               const Eigen::VectorXd& q_weight,  
                               const Eigen::VectorXd& v_weight, 
                               const Eigen::VectorXd& a_weight,  
                               const Eigen::VectorXd& u_weight,
                               const Eigen::VectorXd& qf_weight,  
                               const Eigen::VectorXd& vf_weight)
  : has_floating_base_(robot.has_floating_base()),
    dimq_(robot.dimq()),
    dimv_(robot.dimv()),
    q_ref_(q_ref),
    v_ref_(v_ref),
    a_ref_(a_ref),
    u_ref_(u_ref),
    q_weight_(q_weight),
    v_weight_(v_weight),
    a_weight_(a_weight),
    u_weight_(u_weight),
    qf_weight_(qf_weight),
    vf_weight_(vf_weight),
    lq_configuration_(),
    lqq_configuration_(),
    phiqq_configuration_(),
    configuration_Jacobian_() {
  assert(q_weight.size() == dimq_);
  assert(v_weight.size() == dimv_);
  assert(a_weight.size() == dimv_);
  assert(u_weight.size() == dimv_);
  assert(qf_weight.size() == dimq_);
  assert(vf_weight.size() == dimv_);
  if (robot.has_floating_base()) {
    lq_configuration_.resize(robot.dimq());
    lq_configuration_.setZero();
    lqq_configuration_.resize(robot.dimq(), robot.dimq());
    lqq_configuration_.setZero();
    for (int i=0; i<robot.dimq(); ++i) {
      lqq_configuration_.coeffRef(i, i) = q_weight.coeff(i);
    }
    phiqq_configuration_.resize(robot.dimq(), robot.dimq());
    phiqq_configuration_.setZero();
    for (int i=0; i<robot.dimq(); ++i) {
      phiqq_configuration_.coeffRef(i, i) = qf_weight.coeff(i);
    }
    configuration_Jacobian_.resize(robot.dimq(), robot.dimv());
    configuration_Jacobian_.setZero();
  }
}


JointSpaceCost::JointSpaceCost()
  : has_floating_base_(false),
    dimq_(0),
    dimv_(0),
    q_ref_(),
    v_ref_(),
    a_ref_(),
    u_ref_(),
    q_weight_(),
    v_weight_(),
    a_weight_(),
    u_weight_(),
    qf_weight_(),
    vf_weight_(),
    lq_configuration_(),
    lqq_configuration_(),
    phiqq_configuration_(),
    configuration_Jacobian_() {
}


void JointSpaceCost::set_q_ref(const Eigen::VectorXd& q_ref) {
  assert(q_ref.size() == dimq_);
  q_ref_ = q_ref;
}


void JointSpaceCost::set_v_ref(const Eigen::VectorXd& v_ref) {
  assert(v_ref.size() == dimv_);
  v_ref_ = v_ref;
}


void JointSpaceCost::set_a_ref(const Eigen::VectorXd& a_ref) {
  assert(a_ref.size() == dimv_);
  a_ref_ = a_ref;
}


void JointSpaceCost::set_u_ref(const Eigen::VectorXd& u_ref) {
  assert(u_ref.size() == dimv_);
  u_ref_ = u_ref;
}


void JointSpaceCost::set_q_weight(const Eigen::VectorXd& q_weight) {
  assert(q_weight.size() == dimq_);
  q_weight_ = q_weight;
  if (has_floating_base_) {
    lq_configuration_.resize(dimq_);
    lqq_configuration_.resize(dimq_, dimq_);
    for (int i=0; i<dimq_; ++i) {
      lqq_configuration_.coeffRef(i, i) = q_weight.coeff(i);
    }
  }
}


void JointSpaceCost::set_v_weight(const Eigen::VectorXd& v_weight) {
  assert(v_weight.size() == dimv_);
  v_weight_ = v_weight;
}


void JointSpaceCost::set_a_weight(const Eigen::VectorXd& a_weight) {
  assert(a_weight.size() == dimv_);
  a_weight_ = a_weight;
}


void JointSpaceCost::set_u_weight(const Eigen::VectorXd& u_weight) {
  assert(u_weight.size() == dimv_);
  u_weight_ = u_weight;
}


void JointSpaceCost::set_qf_weight(const Eigen::VectorXd& qf_weight) {
  assert(qf_weight.size() == dimq_);
  qf_weight_ = qf_weight;
  if (has_floating_base_) {
    lq_configuration_.resize(dimq_);
    phiqq_configuration_.resize(dimq_, dimq_);
    for (int i=0; i<dimq_; ++i) {
      phiqq_configuration_.coeffRef(i, i) = qf_weight.coeff(i);
    }
  }
}


void JointSpaceCost::set_vf_weight(const Eigen::VectorXd& vf_weight) {
  assert(vf_weight.size() == dimv_);
  vf_weight_ = vf_weight;
}


void JointSpaceCost::setConfigurationJacobian(const Robot& robot, 
                                              const Eigen::VectorXd& q) {
  assert(q.size() == dimq_);
  robot.configurationJacobian(q, configuration_Jacobian_);
}


double JointSpaceCost::l(const double dtau, const Eigen::VectorXd& q, 
                         const Eigen::VectorXd& v, const Eigen::VectorXd& a, 
                         const Eigen::VectorXd& u) const {
  assert(dtau > 0);
  assert(q.size() == dimq_);
  assert(v.size() == dimv_);
  assert(a.size() == dimv_);
  assert(u.size() == dimv_);
  double l = 0;
  l += (q_weight_.array()* (q-q_ref_).array()*(q-q_ref_).array()).sum();
  l += (v_weight_.array()* (v-v_ref_).array()*(v-v_ref_).array()).sum();
  l += (a_weight_.array()* (a-a_ref_).array()*(a-a_ref_).array()).sum();
  l += (u_weight_.array()* (u-u_ref_).array()*(u-u_ref_).array()).sum();
  return 0.5 * dtau * l;
}


void JointSpaceCost::lq(const double dtau, const Eigen::VectorXd& q, 
                        Eigen::VectorXd& lq) {
  assert(dtau > 0);
  assert(q.size() == dimq_);
  assert(lq.size() == dimv_);
  if (has_floating_base_) {
    lq_configuration_.array() = dtau * q_weight_.array() 
                                     * (q.array()-q_ref_.array());
    lq = configuration_Jacobian_.transpose() * lq_configuration_;
  }
  else {
    lq.array() = dtau * q_weight_.array() * (q.array()-q_ref_.array());
  }
}


void JointSpaceCost::lv(const double dtau, const Eigen::VectorXd& v, 
                        Eigen::VectorXd& lv) const {
  assert(dtau > 0);
  assert(v.size() == dimv_);
  assert(lv.size() == dimv_);
  lv.array() = dtau * v_weight_.array() * (v.array()-v_ref_.array());
}


void JointSpaceCost::la(const double dtau, const Eigen::VectorXd& a, 
                        Eigen::VectorXd& la) const {
  assert(dtau > 0);
  assert(a.size() == dimv_);
  assert(la.size() == dimv_);
  la.array() = dtau * a_weight_.array() * (a.array()-a_ref_.array());
}


void JointSpaceCost::lu(const double dtau, const Eigen::VectorXd& u, 
                        Eigen::VectorXd& lu) const {
  assert(dtau > 0);
  assert(u.size() == dimv_);
  assert(lu.size() == dimv_);
  lu.array() = dtau * u_weight_.array() * (u.array()-u_ref_.array());
}


void JointSpaceCost::lqq(const double dtau, Eigen::MatrixXd& lqq) const {
  assert(dtau > 0);
  assert(lqq.rows() == dimv_);
  assert(lqq.cols() == dimv_);
  if (has_floating_base_) {
    lqq = dtau * configuration_Jacobian_.transpose() * lqq_configuration_ 
                                                     * configuration_Jacobian_;
  }
  else {
    for (int i=0; i<dimq_; ++i) {
      lqq.coeffRef(i, i) = dtau * q_weight_.coeff(i);
    }
  }
}


void JointSpaceCost::lvv(const double dtau, Eigen::MatrixXd& lvv) const {
  assert(dtau > 0);
  assert(lvv.rows() == dimv_);
  assert(lvv.cols() == dimv_);
  for (int i=0; i<dimv_; ++i) {
    lvv.coeffRef(i, i) = dtau * v_weight_.coeff(i);
  }
}


void JointSpaceCost::laa(const double dtau, Eigen::MatrixXd& laa) const {
  assert(dtau > 0);
  assert(laa.rows() == dimv_);
  assert(laa.cols() == dimv_);
  for (int i=0; i<dimv_; ++i) {
    laa.coeffRef(i, i) = dtau * a_weight_.coeff(i);
  }
}


void JointSpaceCost::luu(const double dtau, Eigen::MatrixXd& luu) const {
  assert(dtau > 0);
  assert(luu.rows() == dimv_);
  assert(luu.cols() == dimv_);
  for (int i=0; i<dimv_; ++i) {
    luu.coeffRef(i, i) = dtau * u_weight_.coeff(i);
  }
}


void JointSpaceCost::augment_lqq(const double dtau, 
                                 Eigen::MatrixXd& lqq) const {
  assert(dtau > 0);
  assert(lqq.rows() == dimv_);
  assert(lqq.cols() == dimv_);
  if (has_floating_base_) {
    lqq.noalias() 
        += dtau * configuration_Jacobian_.transpose() * lqq_configuration_ 
                                                      * configuration_Jacobian_;
  }
  else {
    for (int i=0; i<dimq_; ++i) {
      lqq.coeffRef(i, i) += dtau * q_weight_.coeff(i);
    }
  }
}


void JointSpaceCost::augment_lvv(const double dtau, 
                                 Eigen::MatrixXd& lvv) const {
  assert(dtau > 0);
  assert(lvv.rows() == dimv_);
  assert(lvv.cols() == dimv_);
  for (int i=0; i<dimv_; ++i) {
    lvv.coeffRef(i, i) += dtau * v_weight_.coeff(i);
  }
}


void JointSpaceCost::augment_laa(const double dtau, 
                                 Eigen::MatrixXd& laa) const {
  assert(dtau > 0);
  assert(laa.rows() == dimv_);
  assert(laa.cols() == dimv_);
  for (int i=0; i<dimv_; ++i) {
    laa.coeffRef(i, i) += dtau * a_weight_.coeff(i);
  }
}


void JointSpaceCost::augment_luu(const double dtau, 
                                 Eigen::MatrixXd& luu) const {
  assert(dtau > 0);
  assert(luu.rows() == dimv_);
  assert(luu.cols() == dimv_);
  for (int i=0; i<dimv_; ++i) {
    luu.coeffRef(i, i) += dtau * u_weight_.coeff(i);
  }
}


double JointSpaceCost::phi(const Eigen::VectorXd& q, 
                           const Eigen::VectorXd& v) const {
  assert(q.size() == dimq_);
  assert(v.size() == dimv_);
  double phi = 0;
  phi += (qf_weight_.array()* (q-q_ref_).array()*(q-q_ref_).array()).sum();
  phi += (vf_weight_.array()* (v-v_ref_).array()*(v-v_ref_).array()).sum();
  return 0.5 * phi;
}


void JointSpaceCost::phiq(const Eigen::VectorXd& q, Eigen::VectorXd& phiq) {
  assert(q.size() == dimq_);
  assert(phiq.size() == dimv_);
  if (has_floating_base_) {
    lq_configuration_.array() = qf_weight_.array() * (q.array()-q_ref_.array());
    phiq = configuration_Jacobian_.transpose() * lq_configuration_;
  }
  else {
    phiq.array() = qf_weight_.array() * (q.array()-q_ref_.array());
  }
}


void JointSpaceCost::phiv(const Eigen::VectorXd& v, 
                          Eigen::VectorXd& phiv) const {
  assert(v.size() == dimv_);
  assert(phiv.size() == dimv_);
  phiv.array() = vf_weight_.array() * (v.array()-v_ref_.array());
}


void JointSpaceCost::phiqq(Eigen::MatrixXd& phiqq) const {
  assert(phiqq.rows() == dimv_);
  assert(phiqq.cols() == dimv_);
  if (has_floating_base_) {
    phiqq = configuration_Jacobian_.transpose() * phiqq_configuration_ 
                                                * configuration_Jacobian_;
  }
  else {
    for (int i=0; i<dimq_; ++i) {
      phiqq.coeffRef(i, i) = qf_weight_.coeff(i);
    }
  }
}


void JointSpaceCost::phivv(Eigen::MatrixXd& phivv) const {
  assert(phivv.rows() == dimv_);
  assert(phivv.cols() == dimv_);
  for (int i=0; i<dimv_; ++i) {
    phivv.coeffRef(i, i) = vf_weight_.coeff(i);
  }
}

} // namespace idocp