#include "cluster_custom_function.hpp"

namespace gmum {

#ifdef RCPP_INTERFACE

ClusterCustomFunction::ClusterCustomFunction(unsigned int id,
		const std::vector<unsigned int> &assignment, const arma::mat &points,
		boost::shared_ptr<Rcpp::Function> function) :
ClusterUseCovMat(id, assignment, points), m_function(function) {
	m_entropy.second = calculate_entropy(m_n, *m_cov_mat.second);
}

ClusterCustomFunction::ClusterCustomFunction ( const ClusterCustomFunction& other ) 
    : ClusterUseCovMat(other), m_function(other.m_function)
{ }

double ClusterCustomFunction::calculate_entropy(int n,
		const arma::mat &cov_mat) {

	return Rcpp::as<double>(
			(*m_function)(Rcpp::Named("m", Rcpp::wrap(n)),
					Rcpp::Named("sigma", Rcpp::wrap(cov_mat))));
}

ClusterCustomFunction* ClusterCustomFunction::clone()
{
    return new ClusterCustomFunction(*this);
}

#endif

}
