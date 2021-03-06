% Generated by roxygen2 (4.1.1): do not edit by hand
% Please edit documentation in R/svm.R
\name{SVM}
\alias{SVM}
\title{SVM}
\usage{
SVM(x, ...)
}
\arguments{
\item{x}{Training data without labels in one of the following formats:
\code{data.frame}, \code{data.matrix}, \code{SparseM::matrix.csr}, \code{Matrix::Matrix},
\code{slam::simple_triplet_matrix}}

\item{y}{Labels in one of the followinf formts: \code{factor}, \code{vector}.
Recommended type is \code{factor}}

\item{data}{Can be passed instead of \code{x,} \code{y} pair with \code{formula} to mark the labels
column, supported formats are:
\code{data.frame}, \code{data.matrix}}

\item{formula}{Can be passed with \code{data}  instead of \code{x}, \code{y} pair,
formula needs to point to lables column, for example: \code{target~.}}

\item{core}{Support Vector Machine library to use in traning, available are:
\code{'libsvm'}, \code{'svmlight'}; default: \code{'libsvm'}}

\item{kernel}{Kernel type as string, available are: \code{'linear'}, \code{'poly'},
\code{'rbf'}, \code{'sigmoid'};
default: \code{'linear'}
\itemize{
\item \code{linear}: \eqn{x'*w}
\item \code{poly}: \eqn{(gamma*x'*w + coef0)^{degree}}
\item \code{rbf}: \eqn{exp(-gamma*|x-w|^2)}
\item \code{sigmoid}: \eqn{tanh(gamma*x'*w + coef0)}
}}

\item{prep}{Preprocess method as string, available are: \code{'none'}, \code{'2e'};
default: \code{'none'}. For more information on \code{2eSVM} see:
\url{http://www.sciencedirect.com/science/article/pii/S0957417414004138}}

\item{C}{Cost/complexity parameter, default: \code{1}}

\item{gamma}{Parameter for \code{poly}, \code{rbf} and \code{sigmoid} kernels,
default: \code{1/n_features}}

\item{coef0}{For \code{poly} and \code{sigmoid} kernels, default: \code{0}}

\item{degree}{For \code{poly} kernel, default: \code{3}}

\item{cache_size}{Cache memory size in MB, default: \code{100}}

\item{tol}{Tolerance of termination criterion, default: \code{1e-3}}

\item{max.iter}{Depending on library:
\itemize{
 \item libsvm: number of iterations after which the training porcess is killed
 (it can end earlier is desired tolerance is met), default: \code{1e6}
 \item svmlight: number of iterations after which if there is no progress traning is killed,
 default: \code{-1} (no limit)
 }}

\item{transductive.learning}{Option got SVM model to deduce missing labels from the dataset,
default: \code{FALSE}
NOTE: this feature is only available with svmlight library, missing labels are marked as
\code{'TR'}, if none are found and transductive to \code{TRUE}, label \code{0} will be
interpreted as missing}

\item{transductive.posratio}{Fraction of unlabeled examples to be classified into the positive class
as float from \eqn{[0,1]}, default: the ratio of positive and negative examples in the training data}

\item{class.weights}{Named vector with weight fir each class, default: \code{NULL}}

\item{example.weights}{Vector of the same length as training data with weights for each traning example,
default: \code{NULL} NOTE: this feature is only supported with svmlight library}

\item{class.type}{Multiclass algorithm type as string,
available are: \code{'one.versus.all', 'one.versus.one'}; default: \code{'one.versus.one'}}

\item{verbosity}{How verbose should the process be, as integer from \eqn{[1,6]}, default: \code{4}}
}
\value{
SVM model object
}
\description{
Create and train SVM model object.
}
\examples{
# train SVM from data in x and labels in y
svm <- SVM(x, y, core="libsvm", kernel="linear", C=1)

# train SVM using a dataset with both data and lables and a formula pointing to labels
formula <- target ~ .
svm <- SVM(formula, data, core="svmlight", kernel="rbf", gamma=1e3)

# train a model with 2eSVM algorithm
data(svm_breast_cancer_dataset)
ds <- svm.breastcancer.dataset
svm.2e <- SVM(x=ds[,-1], y=ds[,1], core="libsvm", kernel="linear", prep = "2e", C=10);
# more at <link to the 2e sample>

# train SVM on a multiclass data set
data(iris)
# with "one vs rest" strategy
svm.ova <- SVM(Species ~ ., data=iris, class.type="one.versus.all", verbosity=0)
# or with "one vs one" strategy
svm.ovo <- SVM(x=iris[,1:4], y=iris[,5], class.type="one.versus.one", verbosity=0)

# we can use svmlights sample weighting feature, suppose we have weights vector
# with a weight for every sample in the traning data
weighted.svm <- SVM(formula=y~., data=df, core="svmlight", kernel="rbf", C=1.0,
                    gamma=0.5, example.weights=weights)

# svmlight alows us to determine missing labels from a dataset
# suppose we have a labels y with missing labels marked as zeros
svm.transduction <- SVM(x, y, transductive.learning=TRUE, core="svmlight")

# for more in-depth examples visit <link to samples on the website>
}

