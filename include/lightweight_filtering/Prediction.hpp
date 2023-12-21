/*
 * Prediction.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_PREDICTIONMODEL_HPP_
#define LWF_PREDICTIONMODEL_HPP_


#define CHECK_COV_PREDICTION_MATRICES


namespace LWF{

template<typename FilterState>
class Prediction: public ModelBase<Prediction<FilterState>,typename FilterState::mtState,typename FilterState::mtState,typename FilterState::mtPredictionNoise>, public PropertyHandler{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef ModelBase<Prediction<FilterState>,typename FilterState::mtState,typename FilterState::mtState,typename FilterState::mtPredictionNoise> mtModelBase;
  typedef FilterState mtFilterState;
  typedef typename mtFilterState::mtState mtState;
  typedef typename mtModelBase::mtInputTuple mtInputTuple;
  typedef typename mtFilterState::mtPredictionMeas mtMeas;
  typedef typename mtFilterState::mtPredictionNoise mtNoise;
  mtMeas meas_;
  Eigen::MatrixXd prenoiP_;
  Eigen::MatrixXd GprenoiP_G;
  Eigen::MatrixXd Fcov;
  Eigen::MatrixXd FcovF;
  Eigen::MatrixXd prenoiPinv_;
  bool disablePreAndPostProcessingWarning_;
  Prediction(): prenoiP_((int)(mtNoise::D_),(int)(mtNoise::D_)),
                GprenoiP_G((int)(mtNoise::D_),(int)(mtNoise::D_)),
                Fcov((int)(mtNoise::D_),(int)(mtNoise::D_)),
                FcovF((int)(mtNoise::D_),(int)(mtNoise::D_)),
                prenoiPinv_((int)(mtNoise::D_),(int)(mtNoise::D_)){
    GprenoiP_G.setZero();
    Fcov.setZero();
    FcovF.setZero();
    prenoiP_.setIdentity();
    prenoiP_ *= 0.0001;
    mtNoise n;
    n.setIdentity();
    n.registerCovarianceToPropertyHandler_(prenoiP_,this,"PredictionNoise.");
    disablePreAndPostProcessingWarning_ = false;
    refreshProperties();
  };
  virtual ~Prediction(){};
  void refreshProperties(){
    prenoiPinv_.setIdentity();
    prenoiP_.llt().solveInPlace(prenoiPinv_);
  }
  void eval_(mtState& x, const mtInputTuple& inputs, double dt) const{
    evalPrediction(x,std::get<0>(inputs),std::get<1>(inputs),dt);
  }
  template<int i,typename std::enable_if<i==0>::type* = nullptr>
  void jacInput_(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt) const{
    jacPreviousState(F,std::get<0>(inputs),dt);
  }
  template<int i,typename std::enable_if<i==1>::type* = nullptr>
  void jacInput_(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt) const{
    jacNoise(F,std::get<0>(inputs),dt);
  }
  virtual void evalPrediction(mtState& x, const mtState& previousState, const mtNoise& noise, double dt) const = 0;
  virtual void evalPredictionShort(mtState& x, const mtState& previousState, double dt) const{
    mtNoise n; // TODO get static for Identity()
    n.setIdentity();
    evalPrediction(x,previousState,n,dt);
  }
  virtual void jacPreviousState(Eigen::MatrixXd& F, const mtState& previousState, double dt) const = 0;
  virtual void jacNoise(Eigen::MatrixXd& F, const mtState& previousState, double dt) const = 0;
  virtual void noMeasCase(mtFilterState& filterState, mtMeas& meas, double dt){};
  virtual void preProcess(mtFilterState& filterState, const mtMeas& meas, double dt){
    if(!disablePreAndPostProcessingWarning_){
      std::cout << "Warning: prediction preProcessing is not implemented!" << std::endl;
    }
  };
  virtual void postProcess(mtFilterState& filterState, const mtMeas& meas, double dt){
    if(!disablePreAndPostProcessingWarning_){
      std::cout << "Warning: prediction postProcessing is not implemented!" << std::endl;
    }
  };

  // This is the prediction step that we do
  virtual int predictMergedEKF(mtFilterState& filterState, const double tTarget, const std::map<double,mtMeas>& measMap){
    const typename std::map<double,mtMeas>::const_iterator itMeasStart = measMap.upper_bound(filterState.t_);
    if(itMeasStart == measMap.end()) return 0;
    typename std::map<double,mtMeas>::const_iterator itMeasEnd = measMap.lower_bound(tTarget);
    if(itMeasEnd != measMap.end()) ++itMeasEnd;
    double dT = std::min(std::prev(itMeasEnd)->first,tTarget)-filterState.t_;
    if(dT <= 0) return 0;

    // Compute mean Measurement
    mtMeas meanMeas;
    typename mtMeas::mtDifVec vec;
    typename mtMeas::mtDifVec difVec;
    vec.setZero();
    double t = itMeasStart->first;
    for(typename std::map<double,mtMeas>::const_iterator itMeas=next(itMeasStart);itMeas!=itMeasEnd;itMeas++){
      itMeas->second.boxMinus(itMeasStart->second,difVec);
      vec = vec + difVec*(std::min(itMeas->first,tTarget)-t);
      t = std::min(itMeas->first,tTarget);
    }
    vec = vec/dT;
    itMeasStart->second.boxPlus(vec,meanMeas);

    preProcess(filterState,meanMeas,dT);
    meas_ = meanMeas;
    this->jacPreviousState(filterState.F_,filterState.state_,dT);
    this->jacNoise(filterState.G_,filterState.state_,dT); // Works for time continuous parametrization of noise
    for(typename std::map<double,mtMeas>::const_iterator itMeas=itMeasStart;itMeas!=itMeasEnd;itMeas++){
      meas_ = itMeas->second;
      this->evalPredictionShort(filterState.state_,filterState.state_,std::min(itMeas->first,tTarget)-filterState.t_);
      filterState.t_ = std::min(itMeas->first,tTarget);
    }

    const int G_rows = 21 + 3* ROVIO_NMAXFEATURE;
    GprenoiP_G = filterState.G_.template block<G_rows,3>(0,12)*prenoiP_.template block<3,3>(12,12)*(filterState.G_.template block<G_rows,3>(0,12)).transpose();
    GprenoiP_G.template block<12,12>(0,0).diagonal() += (filterState.G_.template block<12,12>(0,0).diagonal().cwiseProduct(prenoiP_.template block<12,12>(0,0).diagonal())).cwiseProduct(filterState.G_.template block<12,12>(0,0).diagonal());
    GprenoiP_G.template block<6,6>(15,15).diagonal() += (filterState.G_.template block<6,6>(15,15).diagonal().cwiseProduct(prenoiP_.template block<6,6>(15,15).diagonal())).cwiseProduct(filterState.G_.template block<6,6>(15,15).diagonal());
   
    for (int i=0; i<25; i++)
    {
      GprenoiP_G.template block<3,3>(21+3*i,21+3*i) += filterState.G_.template block<3,3>(21+3*i, 21+3*i) * prenoiP_.template block<3,3>(21+3*i, 21+3*i) * filterState.G_.template block<3,3>(21+3*i, 21+3*i).transpose();
    }


    Fcov.template block<6,G_rows>(0,0) = filterState.F_.template block<6,12>(0, 3) * filterState.cov_.template block(3, 0, 12, 96);
    Fcov.template block<3,G_rows>(0,0) += filterState.cov_.template block<3,96>(0, 0);
    Fcov.template block<15,G_rows>(6,0) = filterState.cov_.template block<15,96>(6, 0);
    Fcov.template block<3,G_rows>(12,0) += filterState.F_.template block<3,3>(12, 9) * filterState.cov_.template block<3,G_rows>(9,0);
    Fcov.template block<75,G_rows>(21,0) = filterState.F_.template block<75,3>(21, 3) * filterState.cov_.template block<3,G_rows>(3,0);
    Fcov.template block<75,G_rows>(21,0) += filterState.F_.template block<75,3>(21, 9) * filterState.cov_.template block<3,G_rows>(9,0);
    Fcov.template block<75,G_rows>(21,0) += filterState.F_.template block<75,6>(21, 15) * filterState.cov_.template block<6,G_rows>(15,0);

    for (int i=0; i<ROVIO_NMAXFEATURE; i++)
    {
      Fcov.template block<3,G_rows>(21+3*i,0) += filterState.F_.template block<3,3>(21+3*i, 21+3*i) * filterState.cov_.template block<3,G_rows>(21+3*i, 0);
    }

    // // calc Fcov * F
    FcovF.template block<96,6>(0,0) =  Fcov.template block<96,15>(0,0) * filterState.F_.template block<6,15>(0, 0).transpose();
    FcovF.template block<96,15>(0,6) = Fcov.template block<96,15>(0, 6);
    FcovF.template block<96,3>(0,12) +=  Fcov.template block<96,3>(0,9) * filterState.F_.template block<3,3>(12,9).transpose();
    FcovF.template block<96,75>(0,21) =  Fcov.template block<96,3>(0,3) * filterState.F_.template block<75,3>(21,3).transpose();
    FcovF.template block<96,75>(0,21) +=  Fcov.template block<96,3>(0,9) * filterState.F_.template block<75,3>(21,9).transpose();
    FcovF.template block<96,75>(0,21) +=  Fcov.template block<96,6>(0,15) * filterState.F_.template block<75,6>(21,15).transpose();


    for (int i=0; i<ROVIO_NMAXFEATURE; i++)
    {
      FcovF.template block<G_rows,3>(0, 21+3*i) += Fcov.template block<G_rows,3>(0, 21+3*i) * filterState.F_.template block<3,3>(21+3*i, 21+3*i).transpose();
    }


    #ifdef CHECK_COV_PREDICTION_MATRICES
      static int total_comp = 0;
      total_comp++;
      static int total_fail_G = 0;
      static int total_fail_F = 0;
      static int total_fail_Pred = 0;
      bool GpG = GprenoiP_G.isApprox(filterState.G_*prenoiP_*filterState.G_.transpose(), 1e-12);
      bool FcF = FcovF.isApprox(filterState.F_*filterState.cov_*filterState.F_.transpose(), 1e-12);
      Eigen::MatrixXd test = FcovF + GprenoiP_G;
      bool predcov =  test.isApprox(filterState.F_*filterState.cov_*filterState.F_.transpose() + filterState.G_*prenoiP_*filterState.G_.transpose(), 1e-12);
      if (!GpG)
      {
        total_fail_G++;
        std::cout<<"Gpg is not the same! || " <<total_fail_G<<"/"<<total_comp<<std::endl;
        std::cout<<GprenoiP_G<<std::endl<<std::endl;
        std::cout<<filterState.G_*prenoiP_*filterState.G_.transpose()<<std::endl<<std::endl;

      }
      if (!FcF)
      {
        total_fail_F++;
        std::cout<<"FcF is not the same! || " <<total_fail_F<<"/"<<total_comp<<std::endl;
        std::cout<<FcovF - filterState.F_*filterState.cov_*filterState.F_.transpose()<<std::endl<<std::endl;
        std::cout<<filterState.F_<<std::endl<<std::endl;
      } 

      if (!predcov)
      {
        total_fail_Pred++;
        std::cout<<"Prediction cov not the same! || " <<total_fail_Pred<<"/"<<total_comp<<std::endl;
      }

    #endif


    filterState.cov_ = FcovF +GprenoiP_G;


    filterState.state_.fix();
    enforceSymmetry(filterState.cov_);
    filterState.t_ = std::min(std::prev(itMeasEnd)->first,tTarget);
    postProcess(filterState,meanMeas,dT);
    return 0;
  }

};

}

#endif /* LWF_PREDICTIONMODEL_HPP_ */
