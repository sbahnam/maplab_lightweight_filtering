/*
 * ModelBase.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef LWF_ModelBase_HPP_
#define LWF_ModelBase_HPP_


namespace LWF{

template<typename DERIVED, typename Output, typename... Inputs>
class ModelBase{
 public:
  static const unsigned int nInputs_ = sizeof...(Inputs);
  typedef Output mtOutput;
  typedef std::tuple<Inputs...> mtInputTuple;
  ModelBase(){};
  virtual ~ModelBase(){};
  void eval(mtOutput& output, const mtInputTuple& inputs, double dt = 0.0) const{
    static_cast<const DERIVED&>(*this).eval_(output,inputs,dt);
  }
  template<int i>
  void jacInput(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt = 0.0) const{
    static_cast<const DERIVED&>(*this).template jacInput_<i>(F,inputs,dt);
  }
  template<int i,int s = 0, int n = std::tuple_element<i,mtInputTuple>::type::D_>
  void jacInputFD(Eigen::MatrixXd& F, const mtInputTuple& inputs, double dt, double d) const{
    static_assert(s + n <= (std::tuple_element<i,mtInputTuple>::type::D_), "Bad dimension for evaluating jacInputFD");
    mtInputTuple inputsDisturbed = inputs;
    typename std::tuple_element<i,mtInputTuple>::type::mtDifVec difVec;
    mtOutput outputReference;
    mtOutput outputDisturbed;
    eval(outputReference,inputs,dt);
    typename mtOutput::mtDifVec dif;
    for(unsigned int j=s;j<s+n;j++){
      difVec.setZero();
      difVec(j) = d;
      std::get<i>(inputs).boxPlus(difVec,std::get<i>(inputsDisturbed));
      eval(outputDisturbed,inputsDisturbed,dt);
      outputDisturbed.boxMinus(outputReference,dif);
      F.col(j) = dif/d;
    }
  }
};

}

#endif /* LWF_ModelBase_HPP_ */
