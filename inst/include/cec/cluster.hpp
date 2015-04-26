#ifndef CLUSTER_HPP
#define CLUSTER_HPP

//add_point and remove_point are updating the cluster. nice! madry stan

#include <armadillo>
#include <cmath>
#include <vector>
#include <list>
#include <utility>
#include "boost/smart_ptr.hpp"
#include "exceptions.hpp"

namespace gmum {

/**
 * Cluster stores its entropy and knows how to update it
 */
class Cluster {
public:
    virtual ~Cluster();
    virtual void swap_results();
    virtual void add_last_point();
    virtual void add_point(const arma::rowvec& point);
    virtual void remove_last_point();
    virtual void remove_point(const arma::rowvec& point);
    
	virtual double entropy_after_add_point(const arma::rowvec &point) = 0;
	virtual double entropy_after_remove_point(const arma::rowvec &point) = 0;
	virtual Cluster* clone() = 0;
    virtual void clear();

	double entropy() const;
	int size() const;
	arma::rowvec get_mean();
	virtual arma::mat get_cov_mat(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points) = 0;
protected:
    unsigned int m_n;
    std::pair<arma::rowvec*, arma::rowvec*> m_mean;
    std::pair<double, double> m_entropy;
    unsigned int m_count;

    Cluster(unsigned int id, const std::vector<unsigned int> &assignment, const arma::mat &points);
    Cluster(const Cluster& other);
    
    void initialize_mean(unsigned int id, const std::vector<unsigned int>& assignment, const arma::mat& points);
};

//abstract, never created
class ClusterUseCovMat: public Cluster {
protected:
	std::pair<arma::mat*, arma::mat*> m_cov_mat;

	void initialize_cov_mat(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    
	ClusterUseCovMat(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    
    ClusterUseCovMat(const ClusterUseCovMat& other);

	virtual double calculate_entropy(int n, const arma::mat &cov_mat) = 0;
public:
    virtual void swap_results();
    virtual double entropy_after_add_point ( const arma::rowvec& point );
    virtual double entropy_after_remove_point ( const arma::rowvec& point );

	virtual arma::mat get_cov_mat(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    virtual void clear();
    
    virtual ~ClusterUseCovMat(); 
};

//abstract, never created
class ClusterOnlyTrace: public Cluster {
protected:
	std::pair<double, double> m_cov_mat_trace;
	ClusterOnlyTrace(unsigned int id,
			const std::vector<unsigned int> & assignment,
			const arma::mat & points);
    ClusterOnlyTrace ( const ClusterOnlyTrace& other );
    
	void compute_cov_mat_trace(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
	virtual double calculate_entropy(double, int) =0;
public:
	virtual arma::mat get_cov_mat(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    
  virtual void swap_results();
  virtual double entropy_after_add_point ( const arma::rowvec& point );
  virtual double entropy_after_remove_point ( const arma::rowvec& point );

  double get_cov_mat_trace();

  virtual ~ClusterOnlyTrace() { }
};

class ClusterStandard: public ClusterUseCovMat {
private:
	double calculate_entropy(int n, const arma::mat &cov_mat);

public:
	ClusterStandard(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    ClusterStandard ( const ClusterStandard& other );
	virtual ClusterStandard* clone();

    virtual ~ClusterStandard() { }

    virtual arma::mat get_cov_mat(unsigned int id,
                                  const std::vector<unsigned int> &assignment,
                                  const arma::mat &points);
};

class ClusterCovMat: public ClusterUseCovMat {
private:
	arma::mat m_inv_sigma;
	double m_sigma_det;

	double calculate_entropy(int n, const arma::mat &cov_mat);
public:
	ClusterCovMat(const arma::mat & sigma, unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    ClusterCovMat(const ClusterCovMat& other); 
	virtual ClusterCovMat* clone();
    virtual ~ClusterCovMat() { }

    virtual arma::mat get_cov_mat(unsigned int id,
                                  const std::vector<unsigned int> &assignment,
                                  const arma::mat &points);
};

class ClusterConstRadius: public ClusterOnlyTrace {
private:
	double calculate_entropy(double, int);
	double m_r;
public:
	ClusterConstRadius(double r, unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    ClusterConstRadius ( const ClusterConstRadius& other );
	virtual ClusterConstRadius* clone();

    virtual ~ClusterConstRadius() { }

    virtual arma::mat get_cov_mat(unsigned int id,
                                  const std::vector<unsigned int> &assignment,
                                  const arma::mat &points);
};

class ClusterSpherical: public ClusterOnlyTrace {
private:
	double calculate_entropy(double, int);
public:
	ClusterSpherical(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    ClusterSpherical ( const ClusterSpherical& other );
	virtual ClusterSpherical* clone();

    virtual ~ClusterSpherical() { }

    virtual arma::mat get_cov_mat(unsigned int id,
                                  const std::vector<unsigned int> &assignment,
                                  const arma::mat &points);
};

class ClusterDiagonal: public ClusterUseCovMat {
private:
	double calculate_entropy(int n, const arma::mat &cov_mat);
public:
	ClusterDiagonal(unsigned int id,
			const std::vector<unsigned int> &assignment,
			const arma::mat &points);
    ClusterDiagonal ( const ClusterDiagonal& other );
	virtual ClusterDiagonal* clone();

    virtual ~ClusterDiagonal() { }

    virtual arma::mat get_cov_mat(unsigned int id,
                                  const std::vector<unsigned int> &assignment,
                                  const arma::mat &points);
};

}

#endif
