#include <string>
#include <random>
#include <utility>
#include <vector>
#include <memory>

#include <gtest/gtest.h>
#include "Eigen/Core"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"

#include "idocp/robot/point_contact.hpp"


namespace idocp {

class FloatingBasePointContactTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    srand((unsigned int) time(0));
    std::random_device rnd;
    urdf_ = "../../urdf/anymal/anymal.urdf";
    pinocchio::urdf::buildModel(urdf_, model_);
    data_ = pinocchio::Data(model_);
    dimq_ = model_.nq;
    dimv_ = model_.nv;
    njoints_ = model_.njoints;
    contact_frame_id_ = 2 + (int)((rnd()%(model_.nframes-2)) / 2) * 2;
    q_ = pinocchio::randomConfiguration(model_, 
                                        -Eigen::VectorXd::Ones(dimq_), 
                                        Eigen::VectorXd::Ones(dimq_));
    v_ = Eigen::VectorXd::Random(dimv_);
    a_ = Eigen::VectorXd::Random(dimv_);
    baumgarte_weight_on_velocity_ = std::abs(Eigen::VectorXd::Random(2)[1]);
    baumgarte_weight_on_position_ = std::abs(Eigen::VectorXd::Random(2)[1]);
  }

  virtual void TearDown() {
  }

  std::string urdf_;
  pinocchio::Model model_;
  pinocchio::Data data_;
  int dimq_, dimv_, njoints_, contact_frame_id_;
  double baumgarte_weight_on_velocity_, baumgarte_weight_on_position_;
  Eigen::VectorXd q_, v_, a_;
};


TEST_F(FloatingBasePointContactTest, defaultConstructor) {
  PointContact contact;
  // Check the constructor works well.
  EXPECT_FALSE(contact.isActive());
  EXPECT_EQ(contact.contact_frame_id(), 0);
  EXPECT_EQ(contact.parent_joint_id(), 0);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_velocity(), 0);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_position(), 0);
  EXPECT_TRUE(contact.contact_point().isApprox(Eigen::Vector3d::Zero()));
}


TEST_F(FloatingBasePointContactTest, constructor) {
  PointContact contact(model_, contact_frame_id_, baumgarte_weight_on_velocity_, 
                       baumgarte_weight_on_position_);
  // Check the constructor works well.
  EXPECT_FALSE(contact.isActive());
  EXPECT_EQ(contact.contact_frame_id(), contact_frame_id_);
  EXPECT_EQ(contact.parent_joint_id(), model_.frames[contact_frame_id_].parent);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_velocity(), 
                   baumgarte_weight_on_velocity_);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_position(), 
                   baumgarte_weight_on_position_);
  EXPECT_TRUE(contact.contact_point().isApprox(Eigen::Vector3d::Zero()));
  // Check reset parameters set appropriately.
  double baumgarte_weight_on_velocity_tmp 
      = std::abs(Eigen::VectorXd::Random(2)[1]);
  double baumgarte_weight_on_position_tmp 
      = std::abs(Eigen::VectorXd::Random(2)[1]);
  Eigen::Vector3d contact_point = Eigen::Vector3d::Random();
  contact.resetBaugrarteParameters(baumgarte_weight_on_velocity_tmp, 
                                   baumgarte_weight_on_position_tmp);
  EXPECT_DOUBLE_EQ(baumgarte_weight_on_velocity_tmp, 
                   contact.baumgarte_weight_on_velocity());
  EXPECT_DOUBLE_EQ(baumgarte_weight_on_position_tmp, 
                   contact.baumgarte_weight_on_position());
  // Check activate works appropriately.
  contact.activate();
  EXPECT_TRUE(contact.isActive());
  contact.deactivate();
  EXPECT_FALSE(contact.isActive());
  // Check the contact point assignment by kinematics.
  pinocchio::forwardKinematics(model_, data_, q_);
  pinocchio::updateFramePlacement(model_, data_, contact_frame_id_);
  contact.resetContactPointByCurrentKinematics(data_);
  EXPECT_TRUE(
      contact.contact_point().isApprox(
          data_.oMf[contact_frame_id_].translation()));
}


TEST_F(FloatingBasePointContactTest, copyConstructor) {
  PointContact contact_ref(model_, contact_frame_id_, 
                           baumgarte_weight_on_velocity_, 
                           baumgarte_weight_on_position_);
  contact_ref.activate();
  pinocchio::forwardKinematics(model_, data_, q_);
  pinocchio::updateFramePlacement(model_, data_, contact_frame_id_);
  contact_ref.resetContactPointByCurrentKinematics(data_);
  Eigen::Vector3d contact_point_ref = contact_ref.contact_point();
  PointContact contact(contact_ref);
  // Check the constructor works well.
  EXPECT_TRUE(contact.isActive());
  EXPECT_EQ(contact.contact_frame_id(), contact_frame_id_);
  EXPECT_EQ(contact.parent_joint_id(), model_.frames[contact_frame_id_].parent);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_velocity(),  
                   baumgarte_weight_on_velocity_);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_position(), 
                   baumgarte_weight_on_position_);
  EXPECT_TRUE(contact.contact_point().isApprox(contact_point_ref));
  EXPECT_TRUE(
      contact.contact_point().isApprox(
          data_.oMf[contact_frame_id_].translation()));
}


TEST_F(FloatingBasePointContactTest, assign) {
  PointContact contact_ref(model_, contact_frame_id_, 
                           baumgarte_weight_on_velocity_, 
                           baumgarte_weight_on_position_);
  contact_ref.activate();
  pinocchio::forwardKinematics(model_, data_, q_);
  pinocchio::updateFramePlacement(model_, data_, contact_frame_id_);
  contact_ref.resetContactPointByCurrentKinematics(data_);
  PointContact contact;
  contact = contact_ref;
  // Check the constructor works well.
  EXPECT_TRUE(contact.isActive());
  EXPECT_EQ(contact.contact_frame_id(), contact_frame_id_);
  EXPECT_EQ(contact.parent_joint_id(), model_.frames[contact_frame_id_].parent);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_velocity(), 
                   baumgarte_weight_on_velocity_);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_position(), 
                   baumgarte_weight_on_position_);
  EXPECT_TRUE(contact.contact_point().isApprox(contact_ref.contact_point()));
  EXPECT_TRUE(
      contact.contact_point().isApprox(
          data_.oMf[contact_frame_id_].translation()));
}


TEST_F(FloatingBasePointContactTest, moveAssign) {
  PointContact contact_ref(model_, contact_frame_id_, 
                           baumgarte_weight_on_velocity_, 
                           baumgarte_weight_on_position_);
  contact_ref.activate();
  pinocchio::forwardKinematics(model_, data_, q_);
  pinocchio::updateFramePlacement(model_, data_, contact_frame_id_);
  contact_ref.resetContactPointByCurrentKinematics(data_);
  PointContact contact_empty = contact_ref;
  PointContact contact = std::move(contact_empty);
  // Check the constructor works well.
  EXPECT_TRUE(contact.isActive());
  EXPECT_EQ(contact.contact_frame_id(), contact_frame_id_);
  EXPECT_EQ(contact.parent_joint_id(), model_.frames[contact_frame_id_].parent);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_velocity(), 
                   baumgarte_weight_on_velocity_);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_position(), 
                   baumgarte_weight_on_position_);
  EXPECT_TRUE(contact.contact_point().isApprox(contact_ref.contact_point()));
  EXPECT_TRUE(
      contact.contact_point().isApprox(
          data_.oMf[contact_frame_id_].translation()));
}


TEST_F(FloatingBasePointContactTest, moveConstructor) {
  PointContact contact_ref(model_, contact_frame_id_, 
                           baumgarte_weight_on_velocity_, 
                           baumgarte_weight_on_position_);
  contact_ref.activate();
  pinocchio::forwardKinematics(model_, data_, q_);
  pinocchio::updateFramePlacement(model_, data_, contact_frame_id_);
  contact_ref.resetContactPointByCurrentKinematics(data_);
  PointContact contact_empty = contact_ref;
  PointContact contact(std::move(contact_empty));
  // Check the constructor works well.
  EXPECT_TRUE(contact.isActive());
  EXPECT_EQ(contact.contact_frame_id(), contact_frame_id_);
  EXPECT_EQ(contact.parent_joint_id(), model_.frames[contact_frame_id_].parent);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_velocity(), 
                   baumgarte_weight_on_velocity_);
  EXPECT_DOUBLE_EQ(contact.baumgarte_weight_on_position(), 
                   baumgarte_weight_on_position_);
  EXPECT_TRUE(contact.contact_point().isApprox(contact_ref.contact_point()));
  EXPECT_TRUE(
      contact.contact_point().isApprox(
          data_.oMf[contact_frame_id_].translation()));
}


TEST_F(FloatingBasePointContactTest, stdvector) {
  std::vector<PointContact> contacts;
  // Check the size at default.
  EXPECT_TRUE(contacts.empty());
  EXPECT_EQ(contacts.size(), 0);
  contacts.push_back(PointContact(model_, contact_frame_id_, 
                                  baumgarte_weight_on_velocity_, 
                                  baumgarte_weight_on_position_));
  // Check the size when an element is appended.
  EXPECT_FALSE(contacts.empty());
  EXPECT_EQ(contacts.size(), 1);
  contacts.push_back(PointContact(model_, (int)(contact_frame_id_/2), 
                                  0.5*baumgarte_weight_on_velocity_, 
                                  0.5*baumgarte_weight_on_position_));
  // Check each contact's parameters.
  EXPECT_EQ(contacts.size(), 2);
  EXPECT_EQ(contacts[0].contact_frame_id(), contact_frame_id_);
  EXPECT_EQ(contacts[0].baumgarte_weight_on_velocity(), 
            baumgarte_weight_on_velocity_);
  EXPECT_EQ(contacts[0].baumgarte_weight_on_position(), 
            baumgarte_weight_on_position_);
  EXPECT_EQ(contacts[1].contact_frame_id(), (int)(contact_frame_id_/2));
  EXPECT_EQ(contacts[1].baumgarte_weight_on_velocity(), 
            0.5*baumgarte_weight_on_velocity_);
  EXPECT_EQ(contacts[1].baumgarte_weight_on_position(), 
            0.5*baumgarte_weight_on_position_);
  int remove_idx = 0;
  std::swap<PointContact>(contacts[remove_idx], contacts.back());
  contacts.pop_back();
  // Check whether swap-and-pop_back works correctly.
  EXPECT_EQ(contacts.size(), 1);
  EXPECT_EQ(contacts[0].contact_frame_id(), (int)(contact_frame_id_/2));
  EXPECT_EQ(contacts[0].baumgarte_weight_on_velocity(), 
            0.5*baumgarte_weight_on_velocity_);
  EXPECT_EQ(contacts[0].baumgarte_weight_on_position(), 
            0.5*baumgarte_weight_on_position_);
  contacts.push_back(PointContact(model_, contact_frame_id_, 
                                  baumgarte_weight_on_velocity_, 
                                  baumgarte_weight_on_position_));
  EXPECT_EQ(contacts.size(), 2);
  // Check each contact's parameters.
  EXPECT_EQ(contacts[0].contact_frame_id(), (int)(contact_frame_id_/2));
  EXPECT_EQ(contacts[0].baumgarte_weight_on_velocity(), 
            0.5*baumgarte_weight_on_velocity_);
  EXPECT_EQ(contacts[0].baumgarte_weight_on_position(), 
            0.5*baumgarte_weight_on_position_);
  EXPECT_EQ(contacts[1].contact_frame_id(), contact_frame_id_);
  EXPECT_EQ(contacts[1].baumgarte_weight_on_velocity(), 
            baumgarte_weight_on_velocity_);
  EXPECT_EQ(contacts[1].baumgarte_weight_on_position(), 
            baumgarte_weight_on_position_);
  remove_idx = 1;
  contacts.erase(contacts.begin()+remove_idx);
  // Check whether erase works correctly.
  EXPECT_EQ(contacts.size(), 1);
  EXPECT_EQ(contacts[0].contact_frame_id(), (int)(contact_frame_id_/2));
  EXPECT_EQ(contacts[0].baumgarte_weight_on_velocity(), 
            0.5*baumgarte_weight_on_velocity_);
  EXPECT_EQ(contacts[0].baumgarte_weight_on_position(), 
            0.5*baumgarte_weight_on_position_);
  contacts.push_back(PointContact(model_, contact_frame_id_, 
                                  baumgarte_weight_on_velocity_, 
                                  baumgarte_weight_on_position_));
  contacts.push_back(PointContact(model_, contact_frame_id_, 
                                  baumgarte_weight_on_velocity_, 
                                  baumgarte_weight_on_position_));
  // Check whether erase works correctly.
  EXPECT_EQ(contacts.size(), 3);
  contacts.erase(contacts.begin()+remove_idx, contacts.end());
  EXPECT_EQ(contacts.size(), 1);
  contacts.pop_back();
  EXPECT_TRUE(contacts.empty());
}


TEST_F(FloatingBasePointContactTest, computeJointForceFromContactForce) {
  PointContact contact(model_, contact_frame_id_, baumgarte_weight_on_velocity_, 
                       baumgarte_weight_on_position_);
  pinocchio::container::aligned_vector<pinocchio::Force> fjoint
      = pinocchio::container::aligned_vector<pinocchio::Force>(
            model_.joints.size(), pinocchio::Force::Zero());
  pinocchio::container::aligned_vector<pinocchio::Force> fjoint_ref = fjoint;
  Eigen::Vector3d fext = Eigen::Vector3d::Random();
  contact.computeJointForceFromContactForce(fext, fjoint);
  const int parent_joint_id = contact.parent_joint_id();
  fjoint_ref[parent_joint_id] 
      = model_.frames[contact_frame_id_].placement.act(
          pinocchio::Force(fext, Eigen::Vector3d::Zero()));
  for (int i=0; i<fjoint.size(); ++i) {
    EXPECT_TRUE(fjoint[i].isApprox(fjoint_ref[i]));
  }
}


TEST_F(FloatingBasePointContactTest, getContactJacobian) {
  PointContact contact(model_, contact_frame_id_, baumgarte_weight_on_velocity_, 
                       baumgarte_weight_on_position_);
  pinocchio::forwardKinematics(model_, data_, q_);
  pinocchio::computeJointJacobians(model_, data_, q_);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, dimv_);
  Eigen::MatrixXd J_trans = Eigen::MatrixXd::Zero(dimv_, 3);
  Eigen::MatrixXd J_ref = Eigen::MatrixXd::Zero(6, dimv_);
  contact.getContactJacobian(model_, data_, J);
  pinocchio::getFrameJacobian(model_, data_, contact_frame_id_, 
                              pinocchio::LOCAL, J_ref);
  EXPECT_TRUE(J.isApprox(J_ref.topRows(3)));
  const bool transpose = true;
  contact.getContactJacobian(model_, data_, J_trans, transpose);
  EXPECT_TRUE(J_trans.isApprox(J_ref.topRows(3).transpose()));
  const int block_rows_begin = rand() % 5;
  const int block_cols_begin = rand() % 5;
  Eigen::MatrixXd J_block 
      = Eigen::MatrixXd::Zero(3+2*block_rows_begin, dimv_+2*block_cols_begin);
  contact.getContactJacobian(model_, data_, J_block.block(block_rows_begin, 
                                                          block_cols_begin,
                                                          3, dimv_));
  EXPECT_TRUE(J_block.block(block_rows_begin, block_cols_begin, 3, dimv_).
              isApprox(J));
  Eigen::MatrixXd J_trans_block 
      = Eigen::MatrixXd::Zero(dimv_+2*block_cols_begin, 3+2*block_rows_begin);
  contact.getContactJacobian(model_, data_, J_trans_block.block(block_cols_begin, 
                                                                block_rows_begin,
                                                                dimv_, 3), true);
  EXPECT_TRUE(J_block.isApprox(J_trans_block.transpose()));
  J_block.block(block_rows_begin, block_cols_begin, 3, dimv_) -= J;
  EXPECT_TRUE(J_block.isZero());
}


TEST_F(FloatingBasePointContactTest, baumgarteResidual) {
  PointContact contact(model_, contact_frame_id_, baumgarte_weight_on_velocity_, 
                       baumgarte_weight_on_position_);
  pinocchio::forwardKinematics(model_, data_, q_, v_, a_);
  pinocchio::computeForwardKinematicsDerivatives(model_, data_, q_, v_, a_);
  pinocchio::updateFramePlacements(model_, data_);
  const int parent_joint_id = contact.parent_joint_id();
  Eigen::Vector3d residual, residual_ref;
  residual.setZero();
  residual_ref.setZero();
  contact.resetContactPointByCurrentKinematics(data_);
  contact.computeBaumgarteResidual(model_, data_, residual);
  residual_ref 
      = pinocchio::getFrameClassicalAcceleration(model_, data_, contact_frame_id_, 
                                                 pinocchio::LOCAL).linear()
          + baumgarte_weight_on_velocity_ 
              * pinocchio::getFrameVelocity(model_, data_, contact_frame_id_, 
                                            pinocchio::LOCAL).linear();
  const auto baumgarte_res_on_position = (data_.oMf[contact_frame_id_].translation() -contact.contact_point());
  residual_ref(2) += baumgarte_weight_on_position_  * baumgarte_res_on_position(2);
  EXPECT_TRUE(residual.isApprox(residual_ref));
  Eigen::VectorXd residuals = Eigen::VectorXd::Zero(10);
  contact.computeBaumgarteResidual(model_, data_, residuals.segment<3>(5));
  EXPECT_TRUE(residuals.head(5).isZero());
  EXPECT_TRUE(residuals.segment<3>(5).isApprox(residual_ref));
  EXPECT_TRUE(residuals.tail(2).isZero());
}


TEST_F(FloatingBasePointContactTest, baumgarteDerivatives) {
  PointContact contact(model_, contact_frame_id_, baumgarte_weight_on_velocity_, 
                       baumgarte_weight_on_position_);
  pinocchio::forwardKinematics(model_, data_, q_, v_, a_);
  pinocchio::computeForwardKinematicsDerivatives(model_, data_, q_, v_, a_);
  pinocchio::updateFramePlacement(model_, data_, contact_frame_id_);
  const int parent_joint_id = contact.parent_joint_id();
  Eigen::MatrixXd baum_partial_dq = Eigen::MatrixXd::Zero(3, dimv_);
  Eigen::MatrixXd baum_partial_dv = Eigen::MatrixXd::Zero(3, dimv_);
  Eigen::MatrixXd baum_partial_da = Eigen::MatrixXd::Zero(3, dimv_);
  contact.computeBaumgarteDerivatives(model_, data_, baum_partial_dq, 
                                      baum_partial_dv, baum_partial_da);
  Eigen::MatrixXd baum_partial_dq_ref = Eigen::MatrixXd::Zero(3, dimv_);
  Eigen::MatrixXd baum_partial_dv_ref = Eigen::MatrixXd::Zero(3, dimv_);
  Eigen::MatrixXd baum_partial_da_ref = Eigen::MatrixXd::Zero(3, dimv_);
  Eigen::MatrixXd frame_v_partial_dq = Eigen::MatrixXd::Zero(6, dimv_);
  Eigen::MatrixXd frame_a_partial_dq = Eigen::MatrixXd::Zero(6, dimv_);
  Eigen::MatrixXd frame_a_partial_dv = Eigen::MatrixXd::Zero(6, dimv_);
  Eigen::MatrixXd frame_a_partial_da = Eigen::MatrixXd::Zero(6, dimv_);
  pinocchio::getFrameAccelerationDerivatives(model_, data_, contact_frame_id_, 
                                             pinocchio::LOCAL,
                                             frame_v_partial_dq, 
                                             frame_a_partial_dq, 
                                             frame_a_partial_dv, 
                                             frame_a_partial_da);
  Eigen::MatrixXd J_frame = Eigen::MatrixXd::Zero(6, dimv_);
  pinocchio::getFrameJacobian(model_, data_, contact_frame_id_, 
                              pinocchio::LOCAL, J_frame);
  pinocchio::Motion v_frame = pinocchio::getFrameVelocity(model_, data_, 
                                                          contact_frame_id_, 
                                                          pinocchio::LOCAL);
  Eigen::Matrix3d v_linear_skew, v_angular_skew;
  v_linear_skew.setZero();
  v_angular_skew.setZero();
  pinocchio::skew(v_frame.linear(), v_linear_skew);
  pinocchio::skew(v_frame.angular(), v_angular_skew);
  baum_partial_dq_ref 
      = frame_a_partial_dq.template topRows<3>()
          + v_angular_skew * frame_v_partial_dq.template topRows<3>()
          + v_linear_skew * frame_v_partial_dq.template bottomRows<3>()
          + baumgarte_weight_on_velocity_ 
              * frame_v_partial_dq.template topRows<3>();
  const auto baumgarte_dq_dq = data_.oMf[contact_frame_id_].rotation() * J_frame.template topRows<3>();
  baum_partial_dq_ref.row(2) += baumgarte_weight_on_position_ * baumgarte_dq_dq.row(2);
  baum_partial_dv_ref 
      = frame_a_partial_dv.template topRows<3>()
          + v_angular_skew * J_frame.template topRows<3>()
          + v_linear_skew * J_frame.template bottomRows<3>()
          + baumgarte_weight_on_velocity_
              * frame_a_partial_da.template topRows<3>();
  baum_partial_da_ref 
      = frame_a_partial_da.template topRows<3>();
  EXPECT_TRUE(baum_partial_dq_ref.isApprox(baum_partial_dq));
  EXPECT_TRUE(baum_partial_dv_ref.isApprox(baum_partial_dv));
  EXPECT_TRUE(baum_partial_da_ref.isApprox(baum_partial_da));
  std::cout << "baum_partial_dq:" << std::endl;
  std::cout << baum_partial_dq << std::endl;
  std::cout << std::endl;
  std::cout << "baum_partial_dv:" << std::endl;
  std::cout << baum_partial_dv << std::endl;
  std::cout << std::endl;
  std::cout << "baum_partial_da:" << std::endl;
  std::cout << baum_partial_da << std::endl;
  std::cout << std::endl;
}


TEST_F(FloatingBasePointContactTest, baumgarteDerivativesBlock) {
  PointContact contact(model_, contact_frame_id_, baumgarte_weight_on_velocity_, 
                       baumgarte_weight_on_position_);
  Eigen::MatrixXd baum_partial_dq_ref = Eigen::MatrixXd::Zero(3, dimv_);
  Eigen::MatrixXd baum_partial_dv_ref = Eigen::MatrixXd::Zero(3, dimv_);
  Eigen::MatrixXd baum_partial_da_ref = Eigen::MatrixXd::Zero(3, dimv_);
  contact.computeBaumgarteDerivatives(model_, data_, baum_partial_dq_ref, 
                                      baum_partial_dv_ref, baum_partial_da_ref);
  const int block_rows_begin = rand() % 5;
  const int block_cols_begin = rand() % 5;
  Eigen::MatrixXd baum_partial_dq 
      = Eigen::MatrixXd::Zero(3+2*block_rows_begin, dimv_+2*block_cols_begin);
  Eigen::MatrixXd baum_partial_dv 
      = Eigen::MatrixXd::Zero(3+2*block_rows_begin, dimv_+2*block_cols_begin);
  Eigen::MatrixXd baum_partial_da 
      = Eigen::MatrixXd::Zero(3+2*block_rows_begin, dimv_+2*block_cols_begin);
  contact.computeBaumgarteDerivatives(
      model_, data_, 
      baum_partial_dq.block(block_rows_begin, block_cols_begin, 3, dimv_),
      baum_partial_dv.block(block_rows_begin, block_cols_begin, 3, dimv_),
      baum_partial_da.block(block_rows_begin, block_cols_begin, 3, dimv_));
  EXPECT_TRUE(
        baum_partial_dq.block(block_rows_begin, block_cols_begin, 3, dimv_)
        .isApprox(baum_partial_dq_ref));
  EXPECT_TRUE(
        baum_partial_dv.block(block_rows_begin, block_cols_begin, 3, dimv_)
        .isApprox(baum_partial_dv_ref));
  EXPECT_TRUE(
        baum_partial_da.block(block_rows_begin, block_cols_begin, 3, dimv_)
        .isApprox(baum_partial_da_ref));
  Eigen::MatrixXd baum_partial_dq_coeff
      = Eigen::MatrixXd::Zero(3+2*block_rows_begin, dimv_+2*block_cols_begin);
  Eigen::MatrixXd baum_partial_dv_coeff
      = Eigen::MatrixXd::Zero(3+2*block_rows_begin, dimv_+2*block_cols_begin);
  Eigen::MatrixXd baum_partial_da_coeff
      = Eigen::MatrixXd::Zero(3+2*block_rows_begin, dimv_+2*block_cols_begin);
  const double coeff = Eigen::VectorXd::Random(1)[0];
  contact.computeBaumgarteDerivatives(
      model_, data_, coeff,
      baum_partial_dq.block(block_rows_begin, block_cols_begin, 3, dimv_),
      baum_partial_dv.block(block_rows_begin, block_cols_begin, 3, dimv_),
      baum_partial_da.block(block_rows_begin, block_cols_begin, 3, dimv_));
  EXPECT_TRUE(baum_partial_dq_coeff.isApprox(coeff*baum_partial_dq));
  EXPECT_TRUE(baum_partial_dv_coeff.isApprox(coeff*baum_partial_dv));
  EXPECT_TRUE(baum_partial_da_coeff.isApprox(coeff*baum_partial_da));
  baum_partial_dq.block(block_rows_begin, block_cols_begin, 3, dimv_) 
      -= baum_partial_dq_ref;
  baum_partial_dv.block(block_rows_begin, block_cols_begin, 3, dimv_) 
      -= baum_partial_dv_ref;
  baum_partial_da.block(block_rows_begin, block_cols_begin, 3, dimv_) 
      -= baum_partial_da_ref;
  EXPECT_TRUE(baum_partial_dq.isZero());
  EXPECT_TRUE(baum_partial_dv.isZero());
  EXPECT_TRUE(baum_partial_da.isZero());
}

} // namespace idocp


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}