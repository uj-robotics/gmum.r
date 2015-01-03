#include "dataset.hpp"

namespace gmum {

  //pure virtual methods
  /*arma::rowvec Dataset::getPoint(int i) const {
    return arma::rowvec();
  }

  int Dataset::size() const {
    return 0;
  }

  int Dataset::dim() const {
    return 0;
  }*/

  DatasetArma::DatasetArma(const arma::mat* data)
    : data(boost::shared_ptr<const arma::mat>(data)) {}

  arma::rowvec DatasetArma::getPoint(int i) const {
    return data->row(i);
  }

  int DatasetArma::size() const {
    return data->n_rows;
  }
  int DatasetArma::dim() const {
    return data->n_cols;
  }

  boost::shared_ptr<const arma::mat>
  DatasetArma::extractPoints() const {
    return data;
  }

  double DatasetNumpy::get(int i, int j) const {
    return bp::extract<double>(data[bp::make_tuple(i,j)]);
  }

  DatasetNumpy::DatasetNumpy(const bn::matrix& _data)
    : data(_data) {}

  arma::rowvec DatasetNumpy::getPoint(int i) const {

    std::vector<double> vec(dim());
    for(int j=0; j<dim(); ++j)
      vec[j] = get(i,j);

    return arma::rowvec(vec);
  }

  int DatasetNumpy::size() const {
    return data.shape(0);
  }
  int DatasetNumpy::dim() const {
    return data.shape(1);
  }

}
