
SOURCES = $(wildcard cec/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

PYTHON_VERSION = 2.7

R_HOME := 	$(shell R RHOME)
RCPPFLAGS := 	$(shell $(R_HOME)/bin/R CMD config --cppflags)
RLDFLAGS := 	$(shell $(R_HOME)/bin/R CMD config --ldflags)

NUMPY_INC = /home/lorddidger/git/Boost.NumPy
PYTHON_INC = /usr/include/python$(PYTHON_VERSION)

RCPPINCL := 	$(shell echo 'Rcpp:::CxxFlags()' | $(R_HOME)/bin/R --vanilla --slave)
RCPPARMAINCL := 	$(shell echo 'RcppArmadillo:::CxxFlags()' | $(R_HOME)/bin/R --vanilla --slave)

RCPPLIB := -L$(subst include,libs,$(shell echo $(RCPPINCL) | cut -c 3-))
RCPPARMALIB := -L$(subst include,libs,$(shell echo $(RCPPARMAINCL) | cut -c 3-))

LAPACK_LIBS := 	$(shell ${R_HOME}/bin/R CMD config LAPACK_LIBS)
BLAS_LIBS := 	$(shell ${R_HOME}/bin/R CMD config BLAS_LIBS)
NUMPY_LIB = -L/usr/lib -L/usr/local/lib -L/home/lorddidger/git/Boost.NumPy/build/lib

PKG_CPPFLAGS := -std=c++11 -I../inst/include/cec -I. $(shell $(R_HOME)/bin/R CMD config CPPFLAGS)
PKG_CXXFLAGS := $(RCPPFLAGS) $(BOOSTINCL) $(RCPPINCL) $(RCPPARMAINCL) $(RINSIDEINCL) $(shell $(R_HOME)/bin/R CMD config CXXFLAGS) -g -pthread -fPIC
PKG_LIBS := $(RLDFLAGS) $(RRPATH) $(RBLAS) $(RLAPACK) $(RCPPLIB) $(RCPPARMALIB) $(RINSIDELIBS) -lrt -lpthread $(LAPACK_LIBS) $(BLAS_LIBS) $(NUMPY_LIB) -lboost_python -lboost_numpy -larmadillo -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION)

%.o : %.cpp
	g++ $(PKG_CPPFLAGS) $(PKG_CXXFLAGS) -I$(PYTHON_INC) -I$(NUMPY_INC) $^ -c -o $@

cec.so: $(OBJECTS)
	export LD_LIBRARY_PATH=~/git/Boost.NumPy/build/lib/
	g++ -shared -Wl,--export-dynamic $^ $(PKG_LIBS) -o cec.so #gmum.r.so

all: cec.so

clean:
	#-rm cec/*.o
	#-rm gmum.r.so
	#-rm cec.so

.PHONY: all clean