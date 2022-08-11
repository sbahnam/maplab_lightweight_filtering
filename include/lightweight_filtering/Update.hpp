/*
 * Update.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_UPDATEMODEL_HPP_
#define LWF_UPDATEMODEL_HPP_

#include <iostream>
#include <fstream>
// #define UPDATELOG
#define CHECK_UPDATE_MATRICES

#include "lightweight_filtering/common.hpp"
#include "lightweight_filtering/ModelBase.hpp"
#include "lightweight_filtering/PropertyHandler.hpp"
#include "lightweight_filtering/SigmaPoints.hpp"
#include "lightweight_filtering/OutlierDetection.hpp"
#include <list>
#include <Eigen/StdVector>

namespace LWF{

template<typename Innovation, typename FilterState, typename Meas, typename Noise, typename OutlierDetection = OutlierDetectionDefault, bool isCoupled = false>
class Update: public ModelBase<Update<Innovation,FilterState,Meas,Noise,OutlierDetection,isCoupled>,Innovation,typename FilterState::mtState,Noise>, public PropertyHandler{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static_assert(!isCoupled || Noise::D_ == FilterState::noiseExtensionDim_,"Noise Size for coupled Update must match noise extension of prediction!");
  typedef ModelBase<Update<Innovation,FilterState,Meas,Noise,OutlierDetection,isCoupled>,Innovation,typename FilterState::mtState,Noise> mtModelBase;
  typedef FilterState mtFilterState;
  typedef typename mtFilterState::mtState mtState;
  typedef typename mtModelBase::mtInputTuple mtInputTuple;
  typedef typename mtFilterState::mtPredictionMeas mtPredictionMeas;
  typedef typename mtFilterState::mtPredictionNoise mtPredictionNoise;
  typedef Innovation mtInnovation;
  typedef Meas mtMeas;
  typedef Noise mtNoise;
  typedef OutlierDetection mtOutlierDetection;
  // mtMeas meas_; // TODO change to pointer, or remove
  const mtMeas * meas_;
  static const bool coupledToPrediction_ = isCoupled;
  bool useSpecialLinearizationPoint_;
  bool useImprovedJacobian_;
  bool hasConverged_;
  bool successfulUpdate_;
  mutable bool cancelIteration_;
  mutable int candidateCounter_;
  mutable Eigen::MatrixXd H_;
  Eigen::MatrixXd Hlin_;
  Eigen::MatrixXd boxMinusJac_;
  Eigen::MatrixXd Hn_;
  Eigen::MatrixXd updnoiP_;
  Eigen::MatrixXd noiP_;
  Eigen::MatrixXd preupdnoiP_;
  Eigen::MatrixXd C_;
  mtInnovation y_;
  mutable Eigen::MatrixXd Py_;
  Eigen::MatrixXd Pyinv_;
  typename mtInnovation::mtDifVec innVector_;
  mtInnovation yIdentity_;
  typename mtState::mtDifVec updateVec_;
  mtState linState_;
  double updateVecNorm_;
  Eigen::MatrixXd K_;
  Eigen::MatrixXd Pyx_;
  mutable typename mtState::mtDifVec difVecLinInv_;

  SigmaPoints<mtState,2*mtState::D_+1,2*(mtState::D_+mtNoise::D_)+1,0> stateSigmaPoints_;
  SigmaPoints<mtNoise,2*mtNoise::D_+1,2*(mtState::D_+mtNoise::D_)+1,2*mtState::D_> stateSigmaPointsNoi_;
  SigmaPoints<mtInnovation,2*(mtState::D_+mtNoise::D_)+1,2*(mtState::D_+mtNoise::D_)+1,0> innSigmaPoints_;
  SigmaPoints<mtNoise,2*(mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_)> coupledStateSigmaPointsNoi_;
  SigmaPoints<mtInnovation,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,2*(mtState::D_+mtNoise::D_+mtPredictionNoise::D_)+1,0> coupledInnSigmaPoints_;
  SigmaPoints<LWF::VectorElement<mtState::D_>,2*mtState::D_+1,2*mtState::D_+1,0> updateVecSP_;
  SigmaPoints<mtState,2*mtState::D_+1,2*mtState::D_+1,0> posterior_;
  double alpha_;
  double beta_;
  double kappa_;
  double updateVecNormTermination_;
  int maxNumIteration_;
  int iterationNum_;
  mtOutlierDetection outlierDetection_;
  unsigned int numSequences;
  bool disablePreAndPostProcessingWarning_;
  Update(): H_((int)(mtInnovation::D_),(int)(mtState::D_)),
      Hlin_((int)(mtInnovation::D_),(int)(mtState::D_)),
      boxMinusJac_((int)(mtState::D_),(int)(mtState::D_)),
      Hn_((int)(mtInnovation::D_),(int)(mtNoise::D_)),
      updnoiP_((int)(mtNoise::D_),(int)(mtNoise::D_)),
      noiP_((int)(mtNoise::D_),(int)(mtNoise::D_)),
      preupdnoiP_((int)(mtPredictionNoise::D_),(int)(mtNoise::D_)),
      C_((int)(mtState::D_),(int)(mtInnovation::D_)),
      Py_((int)(mtInnovation::D_),(int)(mtInnovation::D_)),
      Pyinv_((int)(mtInnovation::D_),(int)(mtInnovation::D_)),
      K_((int)(mtState::D_),(int)(mtInnovation::D_)),
      Pyx_((int)(mtInnovation::D_),(int)(mtState::D_)){
    alpha_ = 1e-3;
    beta_ = 2.0;
    kappa_ = 0.0;
    updateVecNormTermination_ = 1e-6;
    maxNumIteration_  = 10;
    updnoiP_.setIdentity();
    updnoiP_ *= 0.0001;
    noiP_.setZero();
    preupdnoiP_.setZero();
    useSpecialLinearizationPoint_ = false;
    useImprovedJacobian_ = false;
    yIdentity_.setIdentity();
    updateVec_.setIdentity();
    refreshNoiseSigmaPoints();
    refreshUKFParameter();
    mtNoise n;
    n.setIdentity();
    n.registerCovarianceToPropertyHandler_(updnoiP_,this,"UpdateNoise.");
    doubleRegister_.registerScalar("alpha",alpha_);
    doubleRegister_.registerScalar("beta",beta_);
    doubleRegister_.registerScalar("kappa",kappa_);
    doubleRegister_.registerScalar("updateVecNormTermination",updateVecNormTermination_);
    intRegister_.registerScalar("maxNumIteration",maxNumIteration_);
    outlierDetection_.setEnabledAll(false);
    numSequences = 1;
    disablePreAndPostProcessingWarning_ = false;
  };
  virtual ~Update(){};
  void refreshNoiseSigmaPoints(){
    if(noiP_ != updnoiP_){
      noiP_ = updnoiP_;
      stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
    }
  }
  void refreshUKFParameter(){
    stateSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    innSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    coupledInnSigmaPoints_.computeParameter(alpha_,beta_,kappa_);
    updateVecSP_.computeParameter(alpha_,beta_,kappa_);
    posterior_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
    stateSigmaPointsNoi_.computeFromZeroMeanGaussian(noiP_);
    coupledStateSigmaPointsNoi_.computeParameter(alpha_,beta_,kappa_);
  }
  void refreshProperties(){
    refreshPropertiesCustom();
    refreshUKFParameter();
  }
  virtual void refreshPropertiesCustom(){}
  void eval_(mtInnovation& x, const mtInputTuple& inputs, double dt) const{
    evalInnovation(x,std::get<0>(inputs),std::get<1>(inputs));
  }
  template<int i,typename std::enable_if<i==0>::type* = nullptr>
  void jacInput_(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt) const{
    bool itered = false;
    jacState(F,std::get<0>(inputs), itered);
    std::cout<<F<<std::endl;
  }
  template<int i,typename std::enable_if<i==1>::type* = nullptr>
  void jacInput_(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt) const{
    jacNoise(F,std::get<0>(inputs));
  }
  virtual void evalInnovation(mtInnovation& y, const mtState& state, const mtNoise& noise) const = 0;
  virtual void evalInnovationShort(mtInnovation& y, const mtState& state) const{
    mtNoise n; // TODO get static for Identity()
    n.setIdentity();
    evalInnovation(y,state,n);
  }
  virtual void jacState(Eigen::MatrixXd& F, const mtState& state, bool& itered) const = 0;
  virtual void jacNoise(Eigen::MatrixXd& F, const mtState& state) const = 0;
  virtual void preProcess(mtFilterState& filterState, const mtMeas& meas, bool& isFinished){
    isFinished = false;
    if(!disablePreAndPostProcessingWarning_){
      std::cout << "Warning: update preProcessing is not implemented!" << std::endl;
    }
  }
  virtual bool extraOutlierCheck(const mtState& state) const{
    return hasConverged_;
  }
  virtual bool generateCandidates(const mtFilterState& filterState, mtState& candidate, int& zeros) const{
    candidate = filterState.state_;
    candidateCounter_++;
    if(candidateCounter_<=1)
      return true;
    else
      return false;
  }
  virtual void postProcess(mtFilterState& filterState, const mtMeas& meas, const mtOutlierDetection& outlierDetection, bool& isFinished){
    isFinished = true;
    if(!disablePreAndPostProcessingWarning_){
      std::cout << "Warning: update postProcessing is not implemented!" << std::endl;
    }
  }
  int performUpdate(mtFilterState& filterState, const mtMeas& meas){
    bool isFinished = true;
    int r = 0;
    // int features_count = 0;
    // double T_preprocess = 0;
    // double T_IEKF = 0;
    // double T_post = 0;
    // double T_fix = 0;
    // double T_sym = 0;
    meas_ =  &meas;
    do {
      // const double t1 = (double) cv::getTickCount();
      preProcess(filterState,meas,isFinished);
      // const double t2 = (double) cv::getTickCount();
      // T_preprocess += (t2-t1)/cv::getTickFrequency()*1000;
      // double t3;
      // double t4;
      if(!isFinished){
        // features_count++;
        switch(filterState.mode_){
          case ModeEKF:
            r = performUpdateEKF(filterState,meas);
            break;
          case ModeUKF:
            r = performUpdateUKF(filterState,meas);
            break;
          case ModeIEKF:
            // t3 = (double) cv::getTickCount();
            r = performUpdateIEKF(filterState,meas);
            // t4 = (double) cv::getTickCount();
            // T_IEKF += (t4-t3)/cv::getTickFrequency()*1000;
            break;
          default:
            r = performUpdateEKF(filterState,meas);
            break;
        }
      }
      // const double t5 = (double) cv::getTickCount();
      postProcess(filterState,meas,outlierDetection_,isFinished);
      // const double t6 = (double) cv::getTickCount();
      filterState.state_.fix();
      // const double t7 = (double) cv::getTickCount();
      enforceSymmetry(filterState.cov_);
      // const double t8 = (double) cv::getTickCount();
      // T_post += (t6-t5)/cv::getTickFrequency()*1000;
      // T_fix += (t7-t6)/cv::getTickFrequency()*1000;
      // T_sym += (t8-t7)/cv::getTickFrequency()*1000;
    } while (!isFinished);

    // double T_total= T_preprocess + T_IEKF + T_post + T_fix + T_sym;
    // std::ofstream IEKF_log;
    // IEKF_log.open ("/home/stavrow/fpv_dataset/results/IEKF_log.txt", std::ios::app);
    // IEKF_log << features_count << " " << T_preprocess << " " << T_IEKF << " " << T_post << " " << T_fix << " " << T_sym << " " << T_total  <<std::endl;
    // IEKF_log.close();
    return r;
  }
  int performUpdateEKF(mtFilterState& filterState, const mtMeas& meas){
    // meas_ = meas;
    if(!useSpecialLinearizationPoint_){
      // this->jacState(H_,filterState.state_);
      Hlin_ = H_;
      this->jacNoise(Hn_,filterState.state_);
      this->evalInnovationShort(y_,filterState.state_);
    } else {
      filterState.state_.boxPlus(filterState.difVecLin_,linState_);
      // this->jacState(H_,linState_);
      if(useImprovedJacobian_){
        filterState.state_.boxMinusJac(linState_,boxMinusJac_);
        Hlin_ = H_*boxMinusJac_;
      } else {
        Hlin_ = H_;
      }
      this->jacNoise(Hn_,linState_);
      this->evalInnovationShort(y_,linState_);
    }

    if(isCoupled){
      C_ = filterState.G_*preupdnoiP_*Hn_.transpose();
      Py_ = Hlin_*filterState.cov_*Hlin_.transpose() + Hn_*updnoiP_*Hn_.transpose() + Hlin_*C_ + C_.transpose()*Hlin_.transpose();
    } else {
      Py_ = Hlin_*filterState.cov_*Hlin_.transpose() + Hn_*updnoiP_*Hn_.transpose();
    }
    y_.boxMinus(yIdentity_,innVector_);

    // Outlier detection // TODO: adapt for special linearization point
    outlierDetection_.doOutlierDetection(innVector_,Py_,Hlin_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    if(isCoupled){
      K_ = (filterState.cov_*Hlin_.transpose()+C_)*Pyinv_;
    } else {
      K_ = filterState.cov_*Hlin_.transpose()*Pyinv_;
    }
    filterState.cov_ = filterState.cov_ - K_*Py_*K_.transpose();
    if(!useSpecialLinearizationPoint_){
      updateVec_ = -K_*innVector_;
    } else {
      filterState.state_.boxMinus(linState_,difVecLinInv_);
      updateVec_ = -K_*(innVector_+H_*difVecLinInv_); // includes correction for offseted linearization point, dif must be recomputed (a-b != (-(b-a)))
    }
    filterState.state_.boxPlus(updateVec_,filterState.state_);
    return 0;
  }
  int performUpdateIEKF(mtFilterState& filterState, const mtMeas& meas){
    int zeros;
    bool itered = false;
    #ifdef UPDATELOG
      int iters = 0;
      double t0;
      double t012;
      double t013;
      double t014;
      double t015;

      int candidates_count = 0;
      double t01;
      double t1;
      double t2;
      double t3;
      double t4;
      double t42;
      double t5;
      double t6;
      double t7;
      double t8;
      double t9;
      double t10;
      double t102;
      double t103;
      double t104;
      double t11;
      double t12;
      double t13;

      double T_jac = 0;
      double T_noise = 0;
      double T_eval = 0;
      double T_cov = 0;
      double T_boxmin = 0;
      double T_outlier = 0;
      double T_inv = 0;
      double T_kalman = 0;
      double T_update = 0;
      double T_outlier2 = 0;
      double T_score = 0;
      double T_best = 0;
      double T_init = 0;
      double T_init2 = 0;
      double T_init3 = 0;
      double T_init4 = 0;
      double T_init5 = 0;
      double T_final = 0;
      double T_total = 0;

      t0 = (double) cv::getTickCount();
    #endif

    // meas_ = meas;
    #ifdef UPDATELOG
      t012 = (double) cv::getTickCount();
    #endif
    successfulUpdate_ = false;
    candidateCounter_ = 0;

    std::vector<double> scores;
    #ifdef UPDATELOG
      t013 = (double) cv::getTickCount();
    #endif
    std::vector<mtState, Eigen::aligned_allocator<mtState>> states;
    #ifdef UPDATELOG
      t014 = (double) cv::getTickCount();
    #endif
    double bestScore = -1.0;
    mtState bestState;
    MXD bestCov;
    #ifdef UPDATELOG
      t015 = (double) cv::getTickCount();

      t01 = (double) cv::getTickCount();
      T_init = (t01-t0)/cv::getTickFrequency()*1000;
      T_init2 = (t012-t0)/cv::getTickFrequency()*1000;
      T_init3 = (t013-t012)/cv::getTickFrequency()*1000;
      T_init4 = (t014-t013)/cv::getTickFrequency()*1000;
      T_init5 = (t015-t014)/cv::getTickFrequency()*1000;
    #endif
    while(generateCandidates(filterState,linState_, zeros)){
      #ifdef UPDATELOG 
        candidates_count++;
      #endif
      cancelIteration_ = false;
      hasConverged_ = false;
      for(iterationNum_=0;iterationNum_<maxNumIteration_ && !hasConverged_ && !cancelIteration_;iterationNum_++){
        #ifdef UPDATELOG 
          iters++;
          t1 = (double) cv::getTickCount();
        #endif
        this->jacState(H_,linState_, itered);
        itered = true;
        #ifdef UPDATELOG 
          t2 = (double) cv::getTickCount();
        #endif
        #ifdef UPDATELOG 
          t3 = (double) cv::getTickCount();
        #endif
        this->evalInnovationShort(y_,linState_);
        #ifdef UPDATELOG 
          t4 = (double) cv::getTickCount();
        #endif

        #ifdef UPDATELOG
          t42 = (double) cv::getTickCount();
        #endif
        // if(isCoupled){
          // C_ = filterState.G_*preupdnoiP_*Hn_.transpose();
          // Py_ = H_*filterState.cov_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose() + H_*C_ + C_.transpose()*H_.transpose();
        // } else {
          // Py_ = H_*filterState.cov_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose();
        // Py_ = H_*filterState.cov_*H_.transpose() + updnoiP_;
        Py_ = H_.block(0,zeros,2,2)*filterState.cov_.block(zeros, zeros, 2, 2)*H_.block(0,zeros,2,2).transpose() + updnoiP_;
        #ifdef UPDATELOG
          t5 = (double) cv::getTickCount();
        #endif

      #ifdef CHECK_UPDATE_MATRICES;

        this->jacNoise(Hn_,linState_);
        bool Py_b = Py_.isApprox( H_*filterState.cov_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose(), 1e-12);
        if (!Py_b)
          {
            std::cout<<"Py_b is not the same!"<<std::endl;
            std::cout<<Py_ - (H_*filterState.cov_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose())<<std::endl;
            std::cout<<Py_<<std::endl<<std::endl;
            std::cout<<(H_*filterState.cov_*H_.transpose() + Hn_*updnoiP_*Hn_.transpose())<<std::endl<<std::endl;
          //   std::cout<<zeros<<std::endl<<std::endl;
            // std::cout<<filterState.cov_<<std::endl<<std::endl;
          //   std::cout<<Hn_<<std::endl<<std::endl;
          //   std::cout<<updnoiP_<<std::endl<<std::endl;
          }
      #endif

        // }
        y_.boxMinus(yIdentity_,innVector_);
        #ifdef UPDATELOG
          t6 = (double) cv::getTickCount();
        #endif

        // Outlier detection
        outlierDetection_.doOutlierDetection(innVector_,Py_,H_);
        #ifdef UPDATELOG
          t7 = (double) cv::getTickCount();
        #endif
        Pyinv_.setIdentity();
        Py_.llt().solveInPlace(Pyinv_);
        #ifdef UPDATELOG
          t8 = (double) cv::getTickCount();
        #endif

        // Kalman Update
        if(isCoupled){
          K_ = (filterState.cov_*H_.transpose()+C_)*Pyinv_;
        } else {
          // K_ = filterState.cov_*H_.transpose()*Pyinv_;
          //@todo use template block
          K_ = filterState.cov_.block(0, zeros, filterState.cov_.rows(), 2)*(H_.block(0,zeros,2,2).transpose()*Pyinv_); // first 2x2 * 2x2 then rowx2 * 2x2? No rounding error?
          // K_ = filterState.cov_.block(0, zeros, filterState.cov_.rows(), 2)*H_.block(0,zeros,2,2).transpose()*Pyinv_; // first 2x2 * 2x2 then rowx2 * 2x2? No rounding error?
          
        #ifdef CHECK_UPDATE_MATRICES
          bool K_b = K_.isApprox(filterState.cov_*H_.transpose()*Pyinv_, 1e-12);
          if (!K_b)
            std::cout<<"K_b is not the same!"<<std::endl;
        #endif

          #ifdef UPDATELOG
            t9 = (double) cv::getTickCount();
          #endif
        }
        filterState.state_.boxMinus(linState_,difVecLinInv_);


        // updateVec_ = -K_*(innVector_+H_*difVecLinInv_)+difVecLinInv_; // includes correction for offseted linearization point, dif must be recomputed (a-b != (-(b-a)))
        updateVec_ = -K_*(innVector_+H_.block(0,zeros,2,2)*difVecLinInv_.block(zeros, 0, 2, 1))+difVecLinInv_; // includes correction for offseted linearization point, dif must be recomputed (a-b != (-(b-a)))
        #ifdef CHECK_UPDATE_MATRICES
          bool updateVec_b = updateVec_.isApprox(-K_*(innVector_+H_*difVecLinInv_)+difVecLinInv_, 1e-12);
          if (!updateVec_b)
          {
            std::cout<<"updateVec_b is not the same!"<<std::endl;
            // std::cout<<H_.block(0,zeros,2,2)<<std::endl<<std::endl;
            // std::cout<<H_<<std::endl<<std::endl;
          }
        #endif

        linState_.boxPlus(updateVec_,linState_);
        updateVecNorm_ = updateVec_.norm();
        hasConverged_ = updateVecNorm_<=updateVecNormTermination_;
        #ifdef UPDATELOG
          t10 = (double) cv::getTickCount();

          T_jac += (t2-t1)/cv::getTickFrequency()*1000;
          T_noise += (t3-t2)/cv::getTickFrequency()*1000;
          T_eval += (t4-t3)/cv::getTickFrequency()*1000;
          T_cov += (t5-t42)/cv::getTickFrequency()*1000;
          T_boxmin += (t6-t5)/cv::getTickFrequency()*1000;
          T_outlier += (t7-t6)/cv::getTickFrequency()*1000;
          T_inv += (t8-t7)/cv::getTickFrequency()*1000;
          T_kalman += (t9-t8)/cv::getTickFrequency()*1000;
          T_update += (t10-t9)/cv::getTickFrequency()*1000;
        #endif
      }
      #ifdef UPDATELOG
        t102 = (double) cv::getTickCount();
        t103 = t102;
        t104 = t102;
      #endif

      // @todo: check why this takes so long
      if(extraOutlierCheck(linState_)){
        successfulUpdate_ = true;
        double score = (innVector_.transpose()*Pyinv_*innVector_)(0);
        scores.push_back(score);
        states.push_back(linState_);
        #ifdef UPDATELOG
          t103 = (double) cv::getTickCount();
          t104 = t103;
        #endif
        if(bestScore == -1.0 || score < bestScore){
          bestScore = score;
          bestState = linState_;
          //@todo: this is expensive, maybe just store K_ and Py_
          bestCov = filterState.cov_ - K_*Py_*K_.transpose();
        }
        #ifdef UPDATELOG
          t104 = (double) cv::getTickCount();
        #endif
      }
      #ifdef UPDATELOG
        t11 = (double) cv::getTickCount();
        T_outlier2 += (t11-t102)/cv::getTickFrequency()*1000;
        T_score += (t103-t102)/cv::getTickFrequency()*1000;
        T_best += (t104-t103)/cv::getTickFrequency()*1000;
      #endif
    }

    #ifdef UPDATELOG
      t12 = (double) cv::getTickCount();
    #endif
    if(successfulUpdate_){
      if(scores.size() == 1){
        filterState.state_ = bestState;
        filterState.cov_ = bestCov;
      } else {
        bool foundOtherMin = false;
        for(auto it = states.begin();it!=states.end();it++){
          bestState.boxMinus(*it,difVecLinInv_);
          if(difVecLinInv_.norm()>2*updateVecNormTermination_){
            foundOtherMin = true;
            break;
          }
        }
        if(!foundOtherMin){
          filterState.state_ = bestState;
          filterState.cov_ = bestCov;
        } else {
          successfulUpdate_ = false;
        }
      }
    }
    #ifdef UPDATELOG
      t13 = (double) cv::getTickCount();
      T_final = (t13-t12)/cv::getTickFrequency()*1000;
      T_total = (t13-t0)/cv::getTickFrequency()*1000;

      std::ofstream update_log;
      update_log.open ("/home/stavrow/fpv_dataset/results/update_log.txt", std::ios::app);
      update_log << iters << " " << candidates_count << " " << T_jac << " " << T_noise << " " << T_eval << " " << T_cov << " " <<
                  T_boxmin << " " << T_outlier << " " <<  T_inv << " " << T_kalman << " " << T_update << " " << T_outlier2 <<
                  " " << T_score << " " << T_best << " " <<  T_init << " " << T_init2 << " " << T_init3 << " " << T_init4 << " " << T_init5 <<  " " << T_final << " " << T_total << std::endl;
      update_log.close();
    #endif
    return 0;
  }
  int performUpdateUKF(mtFilterState& filterState, const mtMeas& meas){
    // meas_ = meas;
    handleUpdateSigmaPoints<isCoupled>(filterState);
    y_.boxMinus(yIdentity_,innVector_);

    outlierDetection_.doOutlierDetection(innVector_,Py_,Pyx_);
    Pyinv_.setIdentity();
    Py_.llt().solveInPlace(Pyinv_);

    // Kalman Update
    K_ = Pyx_.transpose()*Pyinv_;
    filterState.cov_ = filterState.cov_ - K_*Py_*K_.transpose();
    updateVec_ = -K_*innVector_;

    // Adapt for proper linearization point
    updateVecSP_.computeFromZeroMeanGaussian(filterState.cov_);
    for(unsigned int i=0;i<2*mtState::D_+1;i++){
      filterState.state_.boxPlus(updateVec_+updateVecSP_(i).v_,posterior_(i));
    }
    posterior_.getMean(filterState.state_);
    posterior_.getCovarianceMatrix(filterState.state_,filterState.cov_);
    return 0;
  }
  template<bool IC = isCoupled, typename std::enable_if<(IC)>::type* = nullptr>
  void handleUpdateSigmaPoints(mtFilterState& filterState){
    coupledStateSigmaPointsNoi_.extendZeroMeanGaussian(filterState.stateSigmaPointsNoi_,updnoiP_,preupdnoiP_);
    for(unsigned int i=0;i<coupledInnSigmaPoints_.L_;i++){
      this->evalInnovation(coupledInnSigmaPoints_(i),filterState.stateSigmaPointsPre_(i),coupledStateSigmaPointsNoi_(i));
    }
    coupledInnSigmaPoints_.getMean(y_);
    coupledInnSigmaPoints_.getCovarianceMatrix(y_,Py_);
    coupledInnSigmaPoints_.getCovarianceMatrix(filterState.stateSigmaPointsPre_,Pyx_);
  }
  template<bool IC = isCoupled, typename std::enable_if<(!IC)>::type* = nullptr>
  void handleUpdateSigmaPoints(mtFilterState& filterState){
    refreshNoiseSigmaPoints();
    stateSigmaPoints_.computeFromGaussian(filterState.state_,filterState.cov_);
    for(unsigned int i=0;i<innSigmaPoints_.L_;i++){
      this->evalInnovation(innSigmaPoints_(i),stateSigmaPoints_(i),stateSigmaPointsNoi_(i));
    }
    innSigmaPoints_.getMean(y_);
    innSigmaPoints_.getCovarianceMatrix(y_,Py_);
    innSigmaPoints_.getCovarianceMatrix(stateSigmaPoints_,Pyx_);
  }
  bool testUpdateJacs(double d = 1e-6,double th = 1e-6){
    mtState state;
    mtMeas meas;
    unsigned int s = 1;
    state.setRandom(s);
    meas.setRandom(s);
    return testUpdateJacs(state,meas,d,th);
  }
  bool testUpdateJacs(const mtState& state, const mtMeas& meas, double d = 1e-6,double th = 1e-6){
    mtInputTuple inputs;
    const double dt = 1.0;
    std::get<0>(inputs) = state;
    std::get<1>(inputs).setIdentity(); // Noise is always set to zero for Jacobians
    // meas_ = meas;
    return this->testJacs(inputs,d,th,dt);
  }
};

}

#endif /* LWF_UPDATEMODEL_HPP_ */
