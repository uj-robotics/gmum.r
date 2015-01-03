#ifndef ASSIGNMENT_HPP
#define ASSIGNMENT_HPP

#include "boost/foreach.hpp"
#include <RcppArmadillo.h>
#include <list>
#include "dataset.hpp"

namespace gmum {

/**
 * Different methods of initiating assignment inherit from this class.
 * You use operator() for getting new assignment.
 */
class Assignment {
protected:
    const Dataset &m_points;
    const unsigned int m_nclusters;
public:
    Assignment(const Dataset &points, const unsigned int nclusters) :
        m_points(points), m_nclusters(nclusters) {
    }

    virtual void operator()(std::vector<unsigned int> &assignment) = 0;
    void operator()(bn::ndarray& output);
    virtual ~Assignment() {
    }
};

/**
 * @centers are ids of rows in points
 */
unsigned int find_nearest(unsigned int i,
                          const std::vector<unsigned int> &centers,
			  const Dataset &points);

unsigned int find_nearest(unsigned int i,
                          const std::list<std::vector<double> > &centers,
                          const Dataset &points);

/**
 * @centers are ids of rows in points
 */

template <class T>
void assignPoints(std::vector<unsigned int> &assignment,
		  const T &centers,
		  const Dataset &points) {

    for(unsigned int i=0; i<assignment.size(); ++i)
      assignment[i] = findNearest(i, centers, points);
}

#endif
