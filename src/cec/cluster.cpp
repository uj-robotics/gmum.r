#include "cluster.hpp"
#include <algorithm>

namespace gmum {
    
Cluster::Cluster(unsigned int id, const std::vector<unsigned int> &assignment, const arma::mat &points)
    :   m_n(points.n_cols), m_entropy(std::pair<double, double>(0,0)), m_count(0)
{
	initialize_mean(id, assignment, points);
	if (m_count == 0)
    {
		throw(NoPointsInCluster());
    }
}

Cluster::Cluster(const Cluster& other)
    :   m_n(other.m_n), m_entropy(other.m_entropy)
{
    m_mean.first = new arma::rowvec(*other.m_mean.first);
    m_mean.second = new arma::rowvec(*other.m_mean.second);
}

Cluster::~Cluster()
{
    clear();
}

void Cluster::clear()
{
    if(m_mean.first != 0)
    {
        delete m_mean.first;
        m_mean.first = 0;
    }
    
    if(m_mean.second != 0)
    {
        delete m_mean.second;
        m_mean.second = 0;
    }
}

void Cluster::initialize_mean(unsigned int id, const std::vector<unsigned int>& assignment, const arma::mat &points) 
{
    clear();    
	m_mean.first = new arma::rowvec(m_n, arma::fill::zeros);
    m_mean.second = new arma::rowvec(m_n, arma::fill::zeros);
    
    for(unsigned int i = 0; i < points.n_rows; ++i)
    {
        if(assignment[i] == id)
        {
            (*m_mean.second) += points.row(i);
            ++m_count;
        }
    }
	(*m_mean.second) /= m_count; 
    (*m_mean.first) = (*m_mean.second);
}

void Cluster::swap_results()
{
    std::swap(m_entropy.first, m_entropy.second);
    std::swap(m_mean.first, m_mean.second);
}

int Cluster::size() const {
	return m_count; 
}

arma::rowvec Cluster::get_mean() {
	return (*m_mean.second);
}

double Cluster::entropy() const {
	return m_entropy.second;
}

void Cluster::add_last_point()
{
    ++m_count;
    swap_results();
}

void Cluster::add_point(const arma::rowvec& point)
{
    entropy_after_add_point(point);
    add_last_point();
}

void Cluster::remove_last_point()
{
    --m_count;
    swap_results();
}

void Cluster::remove_point(const arma::rowvec& point)
{
    entropy_after_remove_point(point);
    remove_last_point();
}

void ClusterUseCovMat::initialize_cov_mat(unsigned int id, const std::vector<unsigned int> &assignment, const arma::mat &points)
{
    clear();
    m_cov_mat.first = new arma::mat(m_n, m_n, arma::fill::zeros);
    m_cov_mat.second = new arma::mat(m_n, m_n, arma::fill::zeros);
    for(unsigned int i = 0; i < points.n_rows; ++i)
    {
        if(assignment[i] == id)
        {
            arma::rowvec point = points.row(i);
            arma::rowvec tmp = point - (*m_mean.second);
            (*m_cov_mat.second) += (tmp.t() * tmp) / (m_count);    
        }
    }
    (*m_cov_mat.first) = (*m_cov_mat.second);
}

ClusterUseCovMat::ClusterUseCovMat(unsigned int id, const std::vector<unsigned int> &assignment, const arma::mat &points) 
    :   Cluster(id, assignment, points), m_cov_mat(std::pair<arma::mat*, arma::mat*>(0, 0)) 
{
        initialize_cov_mat(id, assignment, points);
}

ClusterUseCovMat::ClusterUseCovMat(const ClusterUseCovMat& other)
    :   Cluster(other)
{ 
    m_cov_mat.first = new arma::mat(*other.m_cov_mat.first);
    m_cov_mat.second = new arma::mat(*other.m_cov_mat.second);
}

ClusterUseCovMat::~ClusterUseCovMat()
{
    clear();
}

void ClusterUseCovMat::clear()
{
    if(m_cov_mat.first != 0)
    {
        delete m_cov_mat.first;
        m_cov_mat.first = 0;
    }
    
    if(m_cov_mat.second != 0)
    {
        delete m_cov_mat.second;
        m_cov_mat.second = 0;
    }
}

void ClusterUseCovMat::swap_results()
{
    Cluster::swap_results();
    std::swap(m_cov_mat.first, m_cov_mat.second);
}

double ClusterUseCovMat::entropy_after_add_point(const arma::rowvec &point) {
    arma::rowvec& tmp_mean = (*m_mean.first);
    arma::rowvec& curr_mean = (*m_mean.second);
    arma::mat& tmp_cov_mat = (*m_cov_mat.first);
    arma::mat& curr_cov_mat = (*m_cov_mat.second);
    double& tmp_entropy = m_entropy.first;
    
    arma::rowvec r = curr_mean - point;
    tmp_mean = (m_count * curr_mean + point) / (m_count + 1);
    tmp_cov_mat = (static_cast<double>(m_count) / (m_count + 1)) * (curr_cov_mat + (r.t() * r) / (m_count + 1));
    tmp_entropy = calculate_entropy(m_n, tmp_cov_mat);
    return tmp_entropy;
}

double ClusterUseCovMat::entropy_after_remove_point(const arma::rowvec &point) 
{
    arma::rowvec& tmp_mean = (*m_mean.first);
    arma::rowvec& curr_mean = (*m_mean.second);
    arma::mat& tmp_cov_mat = (*m_cov_mat.first);
    arma::mat& curr_cov_mat = (*m_cov_mat.second);
    double& tmp_entropy = m_entropy.first;
    
    arma::rowvec r = curr_mean - point;
    tmp_mean = (m_count * curr_mean - point) / (m_count - 1);
    tmp_cov_mat = (static_cast<double>(m_count) / (m_count - 1)) * (curr_cov_mat - (r.t() * r) / (m_count - 1));
    tmp_entropy = calculate_entropy(m_n, tmp_cov_mat);
    return tmp_entropy; 
}

arma::mat ClusterUseCovMat::get_cov_mat(unsigned int id,
		const std::vector<unsigned int> &assignment, const arma::mat &points) {
	return (*m_cov_mat.second);
}

ClusterOnlyTrace::ClusterOnlyTrace(unsigned int id, const std::vector<unsigned int> & assignment, const arma::mat & points) 
    :   Cluster(id, assignment, points) {
	compute_cov_mat_trace(id, assignment, points);
}

ClusterOnlyTrace::ClusterOnlyTrace(const ClusterOnlyTrace& other)
    :   Cluster(other), m_cov_mat_trace(other.m_cov_mat_trace)
{ }

void ClusterOnlyTrace::compute_cov_mat_trace(unsigned int id,
		const std::vector<unsigned int> &assignment, const arma::mat &points) {
	m_cov_mat_trace = std::pair<double, double>(0, 0);
    for (unsigned int i = 0; i < points.n_rows; i++)
        if (assignment[i] == id) {
            arma::rowvec point = points.row(i);
            arma::rowvec tmp = point - (*m_mean.second);
            m_cov_mat_trace.second += dot(tmp, tmp);
        }
    m_cov_mat_trace.second /= m_count;
    m_cov_mat_trace.first = m_cov_mat_trace.second;
}

void ClusterOnlyTrace::swap_results()
{
    Cluster::swap_results();
    std::swap(m_cov_mat_trace.first, m_cov_mat_trace.second);
}

double ClusterOnlyTrace::entropy_after_remove_point(const arma::rowvec &point) {
    arma::rowvec& tmp_mean = (*m_mean.first);
    arma::rowvec& curr_mean = (*m_mean.second);
    double& tmp_cov_mat_trace = m_cov_mat_trace.first;
    double& curr_cov_mat_trace = m_cov_mat_trace.second;
    double& tmp_entropy = m_entropy.first;
    
    tmp_mean = (m_count * curr_mean - point) / (m_count - 1);
	arma::rowvec mean_diff = curr_mean - tmp_mean;
	arma::rowvec r = tmp_mean - point;
    tmp_cov_mat_trace = ((curr_cov_mat_trace + dot(mean_diff, mean_diff)) * m_count - dot(r, r)) / (m_count - 1);
    tmp_entropy = calculate_entropy(tmp_cov_mat_trace, m_n);
    return tmp_entropy;
}

double ClusterOnlyTrace::entropy_after_add_point(const arma::rowvec & point) 
{
    arma::rowvec& tmp_mean = (*m_mean.first);
    arma::rowvec& curr_mean = (*m_mean.second);
    double& tmp_cov_mat_trace = m_cov_mat_trace.first;
    double& curr_cov_mat_trace = m_cov_mat_trace.second;
    double& tmp_entropy = m_entropy.first;
    
    tmp_mean = (m_count * curr_mean + point) / (m_count + 1);
    arma::rowvec mean_diff = curr_mean - tmp_mean;
    arma::rowvec r = tmp_mean - point;
    tmp_cov_mat_trace = ((curr_cov_mat_trace + dot(mean_diff, mean_diff)) * m_count + dot(r, r)) / (m_count + 1);
    tmp_entropy = calculate_entropy(tmp_cov_mat_trace, m_n);
    return tmp_entropy;   
}

double ClusterOnlyTrace::get_cov_mat_trace() {
	return m_cov_mat_trace.second;
}

arma::mat ClusterOnlyTrace::get_cov_mat(unsigned int id, const std::vector<unsigned int> &assignment, const arma::mat &points) {
    arma::mat out(m_n, m_n, arma::fill::zeros);
    for (unsigned int i = 0; i < points.n_rows; i++) {
        if (assignment[i] == id) {
            arma::rowvec point = points.row(i);
            arma::rowvec tmp = point - (*m_mean.second);
            out += (tmp.t() * tmp) / (m_count);
        }
    }
    return out;
}

double ClusterStandard::calculate_entropy(int n, const arma::mat &cov_mat) {
	return n * log(2 * M_PI * M_E) / 2 + log(arma::det(cov_mat)) / 2;
}

ClusterStandard::ClusterStandard(unsigned int id,
		const std::vector<unsigned int> &assignment, const arma::mat &points) :
		ClusterUseCovMat(id, assignment, points) {
	m_entropy.second = calculate_entropy(m_n, *m_cov_mat.second);
}

ClusterStandard::ClusterStandard(const ClusterStandard& other)
    :   ClusterUseCovMat(other)
{ }

arma::mat ClusterStandard::get_cov_mat(unsigned int id, const std::vector<unsigned int> &assignment, const arma::mat &points) 
{
    return (*m_cov_mat.second);
}

ClusterCovMat::ClusterCovMat(const arma::mat & sigma, unsigned int id,
		const std::vector<unsigned int> &assignment, const arma::mat &points) :
		ClusterUseCovMat(id, assignment, points) {
	m_sigma_det = arma::det(sigma);
	m_inv_sigma = arma::inv(sigma);
	m_entropy.second = calculate_entropy(m_n, *m_cov_mat.second);
}

ClusterCovMat::ClusterCovMat(const ClusterCovMat& other)
    :   ClusterUseCovMat(other), m_inv_sigma(other.m_inv_sigma), m_sigma_det(other.m_sigma_det)
{ }

double ClusterCovMat::calculate_entropy(int n, const arma::mat &cov_mat) {
	return n * log(2 * M_PI) / 2 + arma::trace(m_inv_sigma * cov_mat) / 2 + log(m_sigma_det) / 2;
}

arma::mat ClusterCovMat::get_cov_mat(unsigned int id, const std::vector<unsigned int> &assignment, const arma::mat &points) 
{
    return arma::inv(m_inv_sigma);
}

ClusterConstRadius::ClusterConstRadius(double r, unsigned int id,
		const std::vector<unsigned int> &assignment, const arma::mat &points) :
		ClusterOnlyTrace(id, assignment, points), m_r(r) {
	m_entropy.second = calculate_entropy(m_cov_mat_trace.second, m_n);
}

ClusterConstRadius::ClusterConstRadius(const ClusterConstRadius& other)
    :   ClusterOnlyTrace(other), m_r(other.m_r)
{ }

double ClusterConstRadius::calculate_entropy(double cov_mat_trace, int n) {
	return n * log(2 * M_PI) / 2 + cov_mat_trace / (2 * m_r) + n * log(m_r) / 2;
}

    arma::mat ClusterConstRadius::get_cov_mat(unsigned int id,
                                  const std::vector<unsigned int> &assignment,
                                  const arma::mat &points){
		arma::mat cov = ClusterOnlyTrace::get_cov_mat(id, assignment, points);
		cov = arma::eye(cov.n_cols, cov.n_cols) * arma::trace(cov) / cov.n_cols;
		return cov * m_r;
    }

ClusterSpherical::ClusterSpherical(unsigned int id,
		const std::vector<unsigned int> &assignment, const arma::mat &points) :
		ClusterOnlyTrace(id, assignment, points) {
	m_entropy.second = calculate_entropy(m_cov_mat_trace.second, m_n);
}

ClusterSpherical::ClusterSpherical(const ClusterSpherical& other)
    :   ClusterOnlyTrace(other)
{ }

double ClusterSpherical::calculate_entropy(double cov_mat_trace, int n) {
	return n * log(2 * M_PI * M_E / n) / 2 + n * log(cov_mat_trace) / 2;
}

arma::mat ClusterSpherical::get_cov_mat(unsigned int id,
                                        const std::vector<unsigned int> &assignment,
                                        const arma::mat &points)
{
    arma::mat cov = ClusterOnlyTrace::get_cov_mat(id, assignment, points);
    cov = arma::eye(cov.n_cols, cov.n_cols) * arma::trace(cov) / cov.n_cols;
    return cov;
}

ClusterDiagonal::ClusterDiagonal(unsigned int id,
		const std::vector<unsigned int> &assignment, const arma::mat &points) :
		ClusterUseCovMat(id, assignment, points)
{
	m_entropy.second = calculate_entropy(m_n, *m_cov_mat.second);
}

ClusterDiagonal::ClusterDiagonal(const ClusterDiagonal& other)
    :   ClusterUseCovMat(other)
{ }

double ClusterDiagonal::calculate_entropy(int n, const arma::mat &cov_mat) {
	return n * log(2 * M_PI * M_E) / 2
			+ log(arma::det(arma::diagmat(cov_mat))) / 2;
}

arma::mat ClusterDiagonal::get_cov_mat(unsigned int id,
                                       const std::vector<unsigned int> &assignment,
                                       const arma::mat &points)
{
    (*m_cov_mat.second) = arma::diagmat(*m_cov_mat.second);
    return *m_cov_mat.second;
}

ClusterCovMat* ClusterCovMat::clone() {
    return new ClusterCovMat(*this);
}

ClusterConstRadius* ClusterConstRadius::clone() {
    return new ClusterConstRadius(*this);
}

ClusterSpherical* ClusterSpherical::clone() {
    return new ClusterSpherical(*this);
}

ClusterStandard* ClusterStandard::clone() {
    return new ClusterStandard(*this);
}

ClusterDiagonal* ClusterDiagonal::clone() {
    return new ClusterDiagonal(*this);
}

}