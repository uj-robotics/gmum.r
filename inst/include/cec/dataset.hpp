#include <RcppArmadillo.h>
#include <vector>
#include "boost/smart_ptr.hpp"

#include <boost/python.hpp>
namespace bp = boost::python;
#include <boost/numpy.hpp>
namespace bn = boost::numpy;

#ifndef DATASET_HPP
#define DATASET_HPP

namespace gmum {

  class Dataset {
  protected:
    Dataset() {}
  public:
    virtual arma::rowvec getPoint(int i) const=0;
    virtual int size() const=0;
    virtual int dim() const=0;
  };

  class DatasetArma : public Dataset {
    boost::shared_ptr<const arma::mat> data;
  public:
    DatasetArma(const arma::mat* data);

    arma::rowvec getPoint(int i) const;
    int size() const;
    int dim() const;

    boost::shared_ptr<const arma::mat> extractPoints() const;
  };

  class DatasetNumpy : public Dataset {

    //for some unknown reason data must not be reference
    const bn::matrix data;

    double get(int i, int j) const;
  public:
    DatasetNumpy(const bn::matrix& _data);

    arma::rowvec getPoint(int i) const;

    int size() const;
    int dim() const;
  };
}

#endif
