#ifndef CEC_MODULE_H
#define CEC_MODULE_H

#include <RcppCommon.h>
using namespace Rcpp;

#include "dataset.hpp"
#include "assignment.hpp"
#include "kmeanspp_assignment.hpp"
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/numeric.hpp>
#include <vector>

class CecConfiguration;
class CecModel;

RCPP_EXPOSED_CLASS (CecConfiguration)
RCPP_EXPOSED_CLASS (CecModel)

#include "cec_configuration.hpp"
#include "cec.hpp"
  using namespace gmum;

std::vector<unsigned int>& get_assignment(CecModel* cec) {
  return *cec->get_assignment();
}

//TODO: this is memory issue - fix it
const Dataset* get_points(CecModel* cec) {
  return &*cec->get_points();
}

RCPP_MODULE(cec) {

  class_<Dataset>("Dataset")
    .method("getPoint", &Dataset::getPoint)
    .method("size", &Dataset::size)
    .method("dim", &Dataset::dim)
    ;

  class_<CecConfiguration>("CecConfiguration")
    .constructor()
    .method("setDataSet", &CecConfiguration::set_data_set)
    .method("setEps", &CecConfiguration::set_eps)
    .method("setMix", &CecConfiguration::set_mix)
    .method("setNrOfClusters", &CecConfiguration::set_nclusters)
    .method("setLogEnergy", &CecConfiguration::set_log_energy)
    .method("setLogCluster", &CecConfiguration::set_log_cluster)
    .method("setNstart", &CecConfiguration::set_nstart)
    .method("setCentroids", &CecConfiguration::set_centroids)
    .method("setMethodInit", &CecConfiguration::set_method_init)
    .method("setMethodType", &CecConfiguration::set_method_type)
    .method("setCov", &CecConfiguration::set_cov)
    .method("setR", &CecConfiguration::set_r)
    .method("setFunction", &CecConfiguration::set_function)
    .method("setItmax", &CecConfiguration::set_it_max)
    .method("setIters", &CecConfiguration::set_iters);
  //.field("x", &cecConfiguration::data)

  std::list<double> (CecModel::*predict_1)(std::vector<double>,
					   bool) = &CecModel::predict;
  unsigned int (CecModel::*predict_2)(
				      std::vector<double>) const = &CecModel::predict;

  class_<CecModel>("CecModel")
    .constructor<CecConfiguration*>()
    .method("loop", &CecModel::loop)
    .method("singleLoop", &CecModel::single_loop)
    .method("entropy", &CecModel::entropy)
    .method("energy", &CecModel::get_energy)
    .method("y", get_assignment)
    .method("clustering", get_assignment)
    .method("centers", &CecModel::centers)
    .method("cov", &CecModel::cov)
    .method("predict", predict_1)
    .method("predict", predict_2)
    .method("log.ncluster", &CecModel::get_nclusters)
    .method("log.energy", &CecModel::get_energy)
    .method("log.iters", &CecModel::iters)
    .method("x", get_points);

}



class DatasetWrap : public Dataset, public bp::wrapper<Dataset> {
public:
  arma::rowvec getPoint(int i) const {

    std::cout << "getPoint" << std::endl;

    return this->get_override("getPoint")(i);
  }
  int size() const {
    return this->get_override("size")();
  }
  int dim() const {
    return this->get_override("dim")();
  }
};

std::vector<std::vector<double> > centers(CecModel* cec) {
  std::vector<arma::rowvec> centers;
  centers = cec->centers();

  std::vector<std::vector<double> > output;
  output.reserve(centers.size());
    
  BOOST_FOREACH(arma::rowvec vec, centers) {
    std::vector<double> out(vec.n_cols);
    for(int i=0; i<vec.n_cols; ++i) out[i] = vec[i];
    output.push_back(out);
  }

  return output;
}

bp::list cov(CecModel* cec) {
  std::vector<arma::mat> covs = cec->cov();
  bp::list output;

  BOOST_FOREACH(arma::mat cov, covs) {
    bn::matrix mat(bn::zeros(
			     bp::make_tuple(cov.n_rows, cov.n_cols), 
			     bn::dtype::get_builtin<double>()
			     ),
		   false);

    for(int i=0; i<cov.n_rows; ++i)
      for(int j=0; j<cov.n_cols; ++j)
	mat[bp::make_tuple(i,j)] = cov(i,j);

    output.append(mat);
  }

  return output;
}

template <class T>
std::vector<T> list_to_vec(const std::list<T>& list) {
  std::vector<T> output;
  output.reserve(list.size());

  for(typename std::list<T>::const_iterator it = list.begin();
      it != list.end(); ++it)
    output.push_back(*it);

  return output;
}

std::vector<double> bp_list_to_vec(bp::list point) {
  std::vector<double> vec;
  int dim = bp::len(point);
  vec.reserve(dim);
  for(int i=0; i<dim; ++i) vec.push_back(bp::extract<double>(point[i]));
  return vec;
}

std::vector<unsigned int> getNrOfClusters(CecModel* cec) {
  return list_to_vec(cec->get_nclusters());
}

std::vector<double> getEnergy(CecModel* cec) {
  return list_to_vec(cec->get_energy());
}

std::vector<double> predict_prob(CecModel* cec, bp::list point, bool) {
  std::vector<double> vec = bp_list_to_vec(point);
  return list_to_vec(cec->predict(vec, true));
}

unsigned int predict_cluster(CecModel* cec, bp::list point) {
  std::vector<double> vec = bp_list_to_vec(point);
  return cec->predict(vec);
}


BOOST_PYTHON_MODULE(cec) {

  bp::register_ptr_to_python<
    boost::shared_ptr<std::vector<unsigned int> >
    >();
bp::register_ptr_to_python< boost::shared_ptr<const Dataset> >();

bp::class_<std::vector<unsigned int> >("std::vector<unsigned int>")
.def(bp::vector_indexing_suite<std::vector<unsigned int> >());

bp::class_<std::vector<double> >("std::vector<double>")
.def(bp::vector_indexing_suite<std::vector<double> >());

bp::class_<std::vector<std::vector<double> > >("std::vector<std::vector<double> >")
.def(bp::vector_indexing_suite<std::vector<std::vector<double> > >());

bp::class_<CecModel>("CecModel", bp::init<CecConfiguration*>())
.def("loop", &CecModel::loop)
.def("singleLoop", &CecModel::single_loop)
.def("entropy", &CecModel::entropy)
.def("energy", &CecModel::get_energy)
.def("y", &CecModel::get_assignment)
.def("clustering",&CecModel::get_assignment)
.def("centers", centers)
.def("cov", cov)
.def("predict", predict_prob)
.def("predict", predict_cluster)
.def("log_ncluster", getNrOfClusters)
.def("log_energy", getEnergy)
.def("log_iters", &CecModel::iters)
.def("x", &CecModel::get_points)
;

bn::initialize();

bp::class_<DatasetWrap, boost::noncopyable>("DatasetWrap")
.def("getPoint", bp::pure_virtual(&Dataset::getPoint))
.def("size", bp::pure_virtual(&Dataset::size))
.def("dim", bp::pure_virtual(&Dataset::dim))
;

bp::class_<DatasetNumpy, bp::bases<Dataset> >
("DatasetNumpy", bp::init<const bn::matrix&>())
;

void (Assignment::*ptr)(bn::ndarray& output) = &Assignment::operator();

bp::class_<Assignment, boost::noncopyable>
("Assignment", bp::no_init)
.def("init", ptr)
;

bp::class_<KmeansppAssignment, bp::bases<Assignment> >
("Kmeanspp", bp::init<const Dataset&, const int>())
;

}


#endif
