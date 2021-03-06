library(ggplot2)

#' @title SVM
#' @export
#' 
#' @description Create and train SVM model object. 
#' 
#' @param x Training data without labels in one of the following formats:
#'  \code{data.frame}, \code{data.matrix}, \code{SparseM::matrix.csr}, \code{Matrix::Matrix},
#'  \code{slam::simple_triplet_matrix}
#' @param y Labels in one of the followinf formts: \code{factor}, \code{vector}. 
#'  Recommended type is \code{factor}
#' @param data Can be passed instead of \code{x,} \code{y} pair with \code{formula} to mark the labels 
#'  column, supported formats are:
#'  \code{data.frame}, \code{data.matrix}
#' @param formula Can be passed with \code{data}  instead of \code{x}, \code{y} pair, 
#'  formula needs to point to lables column, for example: \code{target~.}
#' @param core Support Vector Machine library to use in traning, available are: 
#'  \code{'libsvm'}, \code{'svmlight'}; default: \code{'libsvm'} 
#' @param kernel Kernel type as string, available are: \code{'linear'}, \code{'poly'},
#' \code{'rbf'}, \code{'sigmoid'}; 
#' default: \code{'linear'}
#' \itemize{
#' \item \code{linear}: \eqn{x'*w}
#' \item \code{poly}: \eqn{(gamma*x'*w + coef0)^{degree}}
#' \item \code{rbf}: \eqn{exp(-gamma*|x-w|^2)}
#' \item \code{sigmoid}: \eqn{tanh(gamma*x'*w + coef0)}
#' }
#' @param prep Preprocess method as string, available are: \code{'none'}, \code{'2e'}; 
#' default: \code{'none'}. For more information on \code{2eSVM} see:
#' \url{http://www.sciencedirect.com/science/article/pii/S0957417414004138}
#' @param C Cost/complexity parameter, default: \code{1}
#' @param gamma Parameter for \code{poly}, \code{rbf} and \code{sigmoid} kernels, 
#'  default: \code{1/n_features}
#' @param coef0 For \code{poly} and \code{sigmoid} kernels, default: \code{0}
#' @param degree For \code{poly} kernel, default: \code{3}
#' @param cache_size Cache memory size in MB, default: \code{100}
#' @param tol Tolerance of termination criterion, default: \code{1e-3}
#' @param max.iter Depending on library:
#' \itemize{
#'  \item libsvm: number of iterations after which the training porcess is killed 
#'  (it can end earlier is desired tolerance is met), default: \code{1e6}
#'  \item svmlight: number of iterations after which if there is no progress traning is killed, 
#'  default: \code{-1} (no limit)
#'  }  
#' @param transductive.learning Option got SVM model to deduce missing labels from the dataset, 
#'  default: \code{FALSE}
#'  NOTE: this feature is only available with svmlight library, missing labels are marked as 
#'  \code{'TR'}, if none are found and transductive to \code{TRUE}, label \code{0} will be 
#'  interpreted as missing
#' @param transductive.posratio Fraction of unlabeled examples to be classified into the positive class 
#'  as float from \eqn{[0,1]}, default: the ratio of positive and negative examples in the training data
#' @param class.weights Named vector with weight fir each class, default: \code{NULL}
#' @param example.weights Vector of the same length as training data with weights for each traning example,
#'  default: \code{NULL} NOTE: this feature is only supported with svmlight library
#' @param class.type Multiclass algorithm type as string, 
#' available are: \code{'one.versus.all', 'one.versus.one'}; default: \code{'one.versus.one'}
#' @param verbosity How verbose should the process be, as integer from \eqn{[1,6]}, default: \code{4}
#'  
#' @return SVM model object
#' @examples 
#' # train SVM from data in x and labels in y 
#' svm <- SVM(x, y, core="libsvm", kernel="linear", C=1)
#' 
#' # train SVM using a dataset with both data and lables and a formula pointing to labels
#' formula <- target ~ . 
#' svm <- SVM(formula, data, core="svmlight", kernel="rbf", gamma=1e3)
#' 
#' # train a model with 2eSVM algorithm
#' data(svm_breast_cancer_dataset)
#' ds <- svm.breastcancer.dataset
#' svm.2e <- SVM(x=ds[,-1], y=ds[,1], core="libsvm", kernel="linear", prep = "2e", C=10);
#' # more at <link to the 2e sample>
#' 
#' # train SVM on a multiclass data set
#' data(iris)
#' # with "one vs rest" strategy
#' svm.ova <- SVM(Species ~ ., data=iris, class.type="one.versus.all", verbosity=0)
#' # or with "one vs one" strategy
#' svm.ovo <- SVM(x=iris[,1:4], y=iris[,5], class.type="one.versus.one", verbosity=0)
#' 
#' # we can use svmlights sample weighting feature, suppose we have weights vector 
#' # with a weight for every sample in the traning data
#' weighted.svm <- SVM(formula=y~., data=df, core="svmlight", kernel="rbf", C=1.0, 
#'                     gamma=0.5, example.weights=weights)
#'                     
#' # svmlight alows us to determine missing labels from a dataset   
#' # suppose we have a labels y with missing labels marked as zeros
#' svm.transduction <- SVM(x, y, transductive.learning=TRUE, core="svmlight")
#' 
#' # for more in-depth examples visit <link to samples on the website>                       
SVM <- NULL

.createMultiClassSVM <- NULL
summary.MultiClassSVM <- NULL
show.MultiClassSVM <- NULL
plot.MulticlassSVM <- NULL
predict.MultiClassSVM <- NULL

#' @title Predict
#' 
#' @description Returns predicted classes or distance to discriminative for provided test examples.
#' 
#' @export
#' 
#' @usage predict(svm, x)
#' 
#' @param object Trained SVM object.
#' @param x Unlabeled data, in one of the following formats:
#'  \code{data.frame}, \code{data.matrix}, \code{SparseM::matrix.csr}, \code{Matrix::Matrix},
#'  \code{slam::simple_triplet_matrix}
#' @param decision.function Uf \code{TRUE} returns SVMs decision function 
#' (distance of a point from discriminant) instead of predicted labels, default: \code{FALSE}
#' 
#' @docType methods
#' 
predict.svm.gmum <- NULL

#' @title print
#' 
#' @description Prints short summary of the SVM object and its parameters.
#' 
#' @export
#' 
#' @usage print(svm)
#' 
#' @param object SVM object
#' 
#' @docType methods
print.svm <- NULL

#' @title plot
#' 
#' @description Plots trained svm data and models disciriminative
#' 
#' @export
#' 
#' @param x Trained svm object
#' @param X Optional new data points to be predicted and plotted in one of the following formats:
#'  \code{data.frame}, \code{data.matrix}; default: \code{NULL}
#' @param mode Which plotting mode to use as string, available are: 
#'  \itemize{
#'  \item \code{'normal'} - default mode, plots data in cols argument and a linear decision 
#'    boundry in available
#'  \item \code{'pca'} - preforms PCA decomposition and draws data in a subspace of first 2 dimensions 
#'  from the PCA 
#'  \item \code{'contour'} - countour plot for non-linear kernels
#'  }
#' @param cols Data dimensions to be plotted as vector of length 2, default: \code{c(1,2)}
#' @param radius Radius of the plotted data points as float, default: \code{3}
#' @param radius.max Maximum radius of data points can be plotted, when model is trained 
#'  with example weights as float, default: \code{10}
#' 
#' @usage plot(svm)
#' @usage plot(svm, X=x, cols=c(1,3))
#' @usage plot(svm, mode="pca", radius=5)
#' 
plot.svm <- NULL

#' @title summary
#' 
#' @description Prints short summary of a trained model.
#' 
#' @export
#' 
#' @usage summary(svm)
#' 
#' @param svm SVM object 
#' 
summary.svm <- NULL

# Support for caret
caret.gmumSvmRadial <- NULL
caret.gmumSvmLinear <- NULL
caret.gmumSvmPoly <- NULL


SVM.formula <- NULL
SVM.default <- NULL

loadModule('svm_wrapper', TRUE)

evalqOnLoad({

  SVM <<- function(x, ...)
    UseMethod("SVM")
  
  SVM.formula <<- function(formula, data, ...) {
    
    call <- match.call(expand.dots = TRUE)
    
    if (!inherits(formula, "formula")) stop("Please provide valid formula for this method.")
    if(inherits(data, "Matrix") || inherits(x, "simple_triplet_matrix") || inherits(x, "matrix.csr")) 
      stop("Please provide dense data for this method")
  
    labels <- all.vars(update(formula, .~0))
    y <- data[, labels]
    
    # better way?
    if (formula[3] == ".()") {
      x <- data[, colnames(data) != labels]
    }
    else {
      columns <- all.vars(update(formula, 0~.))
      x <- data[, columns]
    } 
    
    if (is.data.frame(x)) x <- data.matrix(x)

    ret <- SVM.default(x, y, ...)
    
    call[[1]] <- as.name("SVM")
    assign("call", call, ret)
    
    return(ret)
  }
  
  .createMultiClassSVM <<- function(x, y, class.type, ...){
    call <- match.call(expand.dots=TRUE)
    ys <- as.factor(y)
    tys <- table(ys)
    lev <- levels(ys)
    pick <- rbind(c(1),c(2))
    if (class.type == 'one.versus.all'){
      ymat <- matrix(-1, nrow = nrow(x), ncol = length(tys))
      ymat[cbind(seq(along = ys), sapply(ys, function(x) which(x == lev)))] <- 1
      # Result: ymat - dummy matrix where ymat[, i] is matrix for problem i
    } else if(class.type == 'one.versus.one') { 
      ## Classification: one against one
      nclass <- length(tys)
      m <- (nclass - 1)
      minus <- nclass + 1 - sequence(m:1)
      plus <- rep(1:m, m:1)
      pick <- rbind(plus, minus)
      xsplit <- split(data.frame(x), ys)
      ymat <- list()
      xlist <- list()
      for(k in 1:ncol(pick)){
        ymat[[k]] <- c(rep(1, nrow(xsplit[[ pick[1, k] ]])), rep(-1, nrow(xsplit[[ pick[2, k] ]])))
        xlist[[k]] <- rbind(xsplit[[ pick[1, k] ]], xsplit[[ pick[2, k] ]])
      }
      
      
      # Result: ymat[[i]] - classes for problem i
      # Result: xlist[[i]] - dataset for problem i
    }else{
      stop("Incorrect class.type")
    }
    # Get number of subproblems
    if(is.matrix(ymat)){ 
      J <- 1:ncol(ymat)               
    }else if(is.list(ymat)){
      J <- 1:length(ymat)
    }
    
    models <- list()
    
    # Fit one model after another
    for (j in J){
      x.model <- NULL
      y.model <- NULL
      
      if (class.type == "one.versus.all"){
        x.model <- x
        y.model <- ymat[,j]
      }else if(class.type == "one.versus.one"){
        # Note: it could be improved, but not so easily in R (all is copy)
        x.model <- xlist[[j]]
        y.model <- ymat[[j]]
      }  
      #TODO: uncomment
      p <- as.list(match.call(expand.dots=TRUE))
      p$x <- x.model
      p$y <- as.factor(y.model)
      models[[j]] <- do.call(SVM, p[2:length(p)])
    }
    call[[1]] <- as.name("SVM")
    core <- as.list(call)$core
    kernel <- as.list(call)$kernel
    if (is.null(core)) core <- "libsvm"
    if (is.null(kernel)) kernel <- "linear"

    obj <- list(models=models, 
                class.type=class.type, 
                X=x,
                Y=y,
                pick=pick, 
                levels=lev, 
                call=call, 
                core=core, 
                kernel=kernel)

    class(obj) <- "MultiClassSVM"
    obj
  }
  
  SVM.default <<- 
  function(x,
           y,
           core         = "libsvm",             
           kernel      = "linear",
           prep        = "none",
           transductive.learning = FALSE,
           transductive.posratio = -1.,
           C           = 1,
           gamma       = if (is.vector(x)) 1 else 1 / ncol(x),
           coef0       = 0,
           degree      = 3,
           shrinking   = TRUE,
           probability = FALSE,
           class.weights    = NULL,
           example.weights    = NULL,
           cache_size  = 100,
           tol         = 1e-3,
           max.iter    = -1,
           verbosity   = 4,
           class.type = 'one.versus.all',
           seed = NULL) {
    # First check if we have binary or multiclass case
    if (!is.vector(y) && !is.factor(y)) {
      stop("y is of a wrong class, please provide vector or factor")
    }
    
    levels <- NULL
    if (is.factor(y)){
      levels <- levels(y)
    }else{
      # Standarizing, easier for library
      y <- as.factor(y)
      levels <- levels(y)
      warning("It is recommended to pass y as factor")
    }
  
    # We don't support transductive multiclass, because it is bazinga
    if((length(levels) > 2 && !transductive.learning)){
      params <- as.list(match.call(expand.dots=TRUE))
      #skipping first param which is function itself
      return(do.call(.createMultiClassSVM, as.list(params[2:length(params)])))
    }
    
    
    call <- match.call(expand.dots = TRUE)
    call[[1]] <- as.name("SVM")

    # check for errors
    if ( core != "libsvm" && core != "svmlight") 
      stop(paste(GMUM_WRONG_LIBRARY, ": bad library, available are: libsvm, svmlight" ))  
    if ( core != "svmlight" && transductive.learning) 
      stop(paste(GMUM_WRONG_LIBRARY, ": bad library, transductive learning is supported only by svmlight" ))  
    if (kernel != "linear" && kernel != "poly" && kernel != "rbf" && kernel != "sigmoid") 
      stop(paste(GMUM_WRONG_KERNEL, ": bad kernel" ))
    if (prep != "2e" && prep != "none") stop(paste(GMUM_BAD_PREPROCESS, ": bad preprocess" ))
    if (verbosity < 0 || verbosity > 6) stop("Wrong verbosity level, should be from 0 to 6")
    if (C < 0 || gamma < 0 || degree < 1) stop(paste(GMUM_WRONG_PARAMS, ": bad SVM parameters" ))
    if (verbosity < 0 || verbosity > 6) stop("Wrong verbosity level, should be from 0 to 6")
    if ((transductive.posratio < 0 && transductive.posratio != -1) || transductive.posratio > 1)
      stop("Please pass transductive.posratio in range [0,1]")  
    
    # check data
    if(nrow(x) != length(y)) stop("x and y have different lengths")
    if(inherits(x, "Matrix")) {
      library("SparseM")
      library("Matrix")
      x <- as(x, "matrix.csr")
    }
    else if(inherits(x, "simple_triplet_matrix")) {
      library("SparseM")
      ind <- order(data$i, data$j)
      x <- new("matrix.csr",
               ra = x$v[ind],
               ja = x$j[ind],
               ia = as.integer(cumsum(c(1, tabulate(x$i[ind])))),
               dimension = c(x$nrow, data$ncol))
    }
    else if(inherits(x, "matrix.csr")) {
      library("SparseM")
    }
    else if(is.data.frame(x)) {
      x <- data.matrix(x)
    }
    else if (!is.matrix(x)){
      stop("data is of a wrong class, please provide supported format: 
           matrix or data.frame for dense; 
           Matrix, simple_triplet_matrix or matrix.csr for sparse")
    }

    sparse <- inherits(x, "matrix.csr")
    
    if (sparse) {
      library("SparseM")
      if (is.null(y)) {
        stop("Please provide label vector y for sparse matrix classification")
      }
    }
    
    

    
    # Binary classification or 2 classes + unlabeled (for transductive learning)
    if( (length(levels) != 2 && !transductive.learning) || 
         (length(levels) != 3 && transductive.learning)){
      stop("Please pass correct (binary) number of classes or 3 for transductive")
    }

    # Decide what label is used for unlabeled examples
    unlabeled.level = "TR"
    unlabeled.level = "TR"
    if(transductive.learning){
      if(! ("TR" %in% levels || "0" %in% levels ) ){
        stop("Please include TR or 0 factor in levels for transductive learning")
      }
      if("TR" %in% levels && "0" %in% levels ){
        stop("Couldn't deduce which label to use for transductive learning")
      }
      
      if("TR" %in% levels){
        unlabeled.level <- "TR"
      }else{
        unlabeled.level <- "0"
      }
    }
    # This ugly block of code ensures -1, 1 and 0 classes.
    # Contribution to simplifying this are welcome :)
    if(transductive.learning){
      # Erasing TR from levels. We will never return it
      levels = levels[levels != unlabeled.level] 
      indexes.unlabeled <- y == unlabeled.level  
      z <- y[!indexes.unlabeled]
      z <- as.integer(factor(z, levels=levels))
      z[z==1] = -1
      z[z==2] = 1
      
      y <- as.integer(y)
      y[indexes.unlabeled] <- 0
      y[!indexes.unlabeled] <- z
    }else{
      y <- as.integer(y) # Standarization, omits 0!
      y[y==1] = -1 # Standarize it further!
      y[y==2] = 1
    }
    
    config <- new(SVMConfiguration)
    config$y <- data.matrix(y)
    
    config$use_transductive_learning = transductive.learning
    config$transductive_posratio = transductive.posratio
    
    # sparse 
    if (sparse) {
      config$sparse <- 1
      
      #x@ia - rowptr 
      #x@ja - colind
      #x@ra - values
      config$set_sparse_data(x@ia, x@ja, x@ra, nrow(x), ncol(x), TRUE)
    }
    else {
      config$sparse <- 0
      config$x <- x
    }
    
    if(!is.null(seed)){
      config$setSeed(seed)
    }
    config$setLibrary(core)
    config$setKernel(kernel)
    config$setPreprocess(prep)
    config$set_verbosity(verbosity)
    
    config$C <- C
    config$gamma <- gamma
    config$coef0 <- coef0
    config$degree <- degree
    config$eps <- tol
    config$cache_size <- cache_size
    config$max_iter <- max.iter
    
    if (!is.null(class.weights) && !is.logical(class.weights)) {
      
      if(is.null(names(class.weights)) && class.weights != 'auto'){
          stop("Please provide class.weights as named (by classes) list or vector or 'auto'")
      }
      
      if (is.character(class.weights) && class.weights == "auto") {
        # sklearns heuristic automatic class weighting
        counts <- hist(y, breaks=2, plot=FALSE)$counts
        inv_freq <- 1 / counts
        weights <- inv_freq / mean(inv_freq)
        config$setClassWeights(weights)
      }
      else {
        # Maps name -> index that is feed into SVM
        # Note: factor is transformed such that class -> index in levels of factor
        class.labels.indexes <- sapply(names(class.weights), function(cls){ which(levels== cls)[1] })
        # Standarize for all libraries (so if passed list("2"=1, "1"=3) it is reversed)
        class.weights <- class.weights[order(class.labels.indexes)] 
        # We always pass numeric, so it will work if it is the case
        if(!is.numeric(y)){
          stop("[DEV] breaking change, please fix")
        }
        config$setClassWeights(as.numeric(class.weights))
      }
    }
    
    if (!is.null(example.weights) && !is.logical(example.weights)) {
      config$use_example_weights <- 1
      config$example_weights <- example.weights
    }
    
    if (shrinking) {
      config$shrinking <- 1
    } else {
      config$shrinking <- 0
    }
    
    if (probability) {
      config$probability <- 1
    } else {
      config$probability <- 0
    }
    
    client <- new(SVMClient, config)
    client$train()

    assign("call", call, client)
    assign("levels", levels, client)
    client 
  } 

  print.svm <- function(x) {
    summary(x)
  }
  
  summary.MultiClassSVM <<- function(object) {
    print(sprintf("Support Vector Machine, multiclass.type: %s, core: %s, kernel: %s",
                  object$class.type, 
                  object$core, 
                  object$kernel))
    print(sprintf("%d classes", 
                  length(object$levels)))
  }
  
  print.MultiClassSVM <<- function(object) {
    summary.MultiClassSVM(object)
  }
  
  summary.svm <<- function(object) {
    print(sprintf("Support Vector Machine, core: %s, kernel: %s, preprocess: %s",
                  object$getLibrary(), 
                  object$getKernel(), 
                  object$getPreprocess()))
    print(sprintf("%d classes with %d support vectors", 
                  object$getNumberClass(), 
                  object$getNumberSV()))
  }
  
  plot.MultiClassSVM <<- function(x, X=NULL, cols=c(1,2), radius=3, radius.max=10) {
    plot.svm(x, X=X, cols=cols, radius=radius, radius.max=radius.max)
  }
  
  plot.svm <<- function(x, X=NULL, mode="normal", cols=c(1,2), radius=3, radius.max=10) {
    #0. Some initial preparing
    if (mode != "pca" && mode != "normal" && mode != "contour" ) {
      stop("Wrong mode!") 
    }
    if(class(x) == "MultiClassSVM"){
      obj <- x$models[[1]]
    }else{
      obj <- x
    }
    if(obj$isSparse()){
      library(SparseM)
      library(Matrix)
      library(e1071)
    }
    
    #1. Get X and Y
    if(is.null(X)){
      if(class(x) == "MultiClassSVM"){
         X <- x$X
         true_target <- x$Y
      }else{
        true_target <- as.factor(x$.getY())
        if(obj$isSparse()){
          X <- Matrix::t(obj$.getSparseX())
        }else{
          X <- obj$.getX()
        }
      }
      
      t <- predict(x, X)
      
    }else{
      t <- predict(x, X)
      true_target <- NULL
    }
    labels <- levels(as.factor(t))
    
    #2. Do some checking

    if (ncol(X) > 2){
      warning("Only 2 dimension plotting is supported for multiclass. Plotting using cols parameter")
    }   
    if (ncol(X) > 2 && mode == "contour"){
      stop("Contour mode is supported only for 2 dimensional data")
    }
    if (ncol(X) == 1){
      stop("Plotting is not supported for 1 dimensional data")
    }
    
    #3. Prepare df. This is ugly copy so that we can do whatever we want
    if(obj$isSparse()){
      df <- data.frame(SparseM::as.matrix(X[,cols]))
    }else{
      df <- data.frame(X[,cols])
    }
    colnames(df) <- c("X1", "X2") # This is even worse
    df['prediction'] <- as.factor(t)
    
    if(!is.null(true_target)){
      levels(true_target) <- x$levels
      df['label'] <- true_target
    }
  
    #4. Prepare data for plotting
    if (obj$areExamplesWeighted()) {
      df['sizes'] <- obj$getExampleWeights()
      scale_size <- scale_size_continuous(range = c(radius,radius.max))
    }else {
      df['sizes'] <- radius
      scale_size <- scale_size_identity()
    }
    
    #5. Support parameters
    kernel <- obj$getKernel()
    
    
    if(mode == "pca"){
      mx <- colMeans(X)
      pca_data <- prcomp(X, scale=FALSE)
      # Transform data
      df$X1 <- pca_data$x[,1]
      df$X2 <- pca_data$x[,2]
    }
    
    w <- NULL
    if (kernel == "linear" && class(x) != "MultiClassSVM") {
      # W will be used only for binary model
      if (mode == "pca") {
        w <- c(obj$getW())
        w <- (w - mx) %*% pca_data$rotation
      }else if(ncol(X)==2){
        w <- c(obj$getW())
      }
    }
    
    
    
    #6. PLOT
    if(ncol(X) == 2 && mode == "contour"){
      x_col <- df$X1
      y_col <- df$X2
      
      x_max <- max(x_col)
      x_min <- min(x_col) 
      y_max <- max(y_col)
      y_min <- min(y_col)
      
      x_axis <- seq(from=x_min, to=x_max, length.out=300)
      y_axis <- seq(from=y_min, to=y_max, length.out=300)
      grid <- data.frame(x_axis,y_axis)
      grid <- expand.grid(x=x_axis,y=y_axis)
      
      prediction <- predict(x, grid)
      grid['prediction'] <- prediction
      
      
      if(!is.null(true_target)){
        pl <- ggplot()+ 
          geom_tile(data=grid, aes(x=x,y=y, fill=prediction, alpha=.5)) + 
          theme(legend.position="none") + 
          scale_fill_brewer(palette="Set1") + 
          scale_alpha_identity() + 
          geom_point(data=df, aes(X1, X2, size=sizes, colour=prediction, shape=label)) + 
          scale_colour_brewer(palette="Set1") + 
          scale_size
      }
      else{
        pl <- ggplot()+ 
          geom_tile(data=grid, aes(x=x,y=y, fill=prediction, alpha=.5)) + 
          theme(legend.position="none") + 
          scale_fill_brewer(palette="Set1") + 
          scale_alpha_identity() + 
          geom_point(data=df, aes(X1, X2, size=sizes, colour=prediction)) + 
          scale_colour_brewer(palette="Set1") + 
          scale_size
      }  
    }else{
      warning("Only limited plotting is currently supported for multidimensional data")
      if(!is.null(true_target)){
        pl <- ggplot()+ 
          geom_point(data=df, aes(X1, X2, size=sizes, colour=prediction, shape=label)) + 
          scale_colour_brewer(palette="Set1") + 
          scale_size
      }else{
        pl <- ggplot()+ 
          geom_point(data=df, aes(X1, X2, size=sizes, colour=prediction)) + 
          scale_colour_brewer(palette="Set1") + 
          scale_size
      }
    }
    
    # Add line
    if(!is.null(w) && ncol(X) && mode != "pca"){
      s <- -w[1]/w[2]
      int <- -obj$getBias()/w[2]
      pl <- pl + geom_abline(slope=s, intercept=int)
    }
    
    plot(pl)
  }
  
  predict.MultiClassSVM <<- function(object, x){
    # Sums votes
    prediction.row.oao <- function(r){
      object$levels[which.max(sapply(1:length(object$levels), function(cl){ sum(r==cl)}))]
    }
    # Argmax of decision function
    prediction.row.oaa <- function(r){
      object$levels[which.max(r)]
    }
    ymat <- c()
    for(i in 1:length(object$models)){
      model <- object$models[[i]]

      if(object$class.type == "one.versus.one"){
        pick <- as.integer(object$pick[,i])
        pick = pick[c(2,1)] # Reverse order
        # Predict
        prediction <- predict(model, x)
        
        # Replace labels
        votes <- pick[as.integer(prediction)]
        ymat <- cbind(ymat, votes)
      }else{
        # Predict
        prediction <- predict(model, x, decision.function=TRUE)
        ymat <- cbind(ymat, prediction)    
      }
    }
    if(object$class.type == "one.versus.one"){
      ymat.preds <- apply(ymat, 1, prediction.row.oao)
    }else if(object$class.type == "one.versus.all"){
      ymat.preds <- apply(ymat, 1, prediction.row.oaa)
    }else{
      stop("Unrecognized class.type")
    }
    return(factor(ymat.preds, levels=object$levels))
  }
  
  predict.svm.gmum <<- function(object, x, decision.function=FALSE) {
    if ( !is(x, "data.frame") && !is(x, "matrix") && !is(x,"numeric") && !is(x,"matrix.csr") ) {
      stop("Wrong target class, please provide data.frame, matrix or numeric vector")
    }
    if (!object$isSparse()) {
      if (!is(x, "matrix") && !is(x, "data.frame")) {
        stop("Please provide matrix or data.frame")
      }
      if (!is(x, "matrix")) {
        x <- data.matrix(x)
      }
      object$predict(x)
    }
    else {
      if (!is(x, "matrix.csr")) {
        stop("Please provide sparse matrix")
      }
      object$sparse_predict(x@ia, x@ja, x@ra, nrow(x), ncol(x))
    }
    
    if(decision.function){
      return(object$getDecisionFunction())
    }else{
      prediction <- object$getPrediction()
      
      if(any(prediction == 0) || length(unique(prediction)) > length(object$levels)){
        stop("Failed prediction, returned too many unique labels from library.")
      }
      
      
      if(!is.null(object$levels)){
        # This line works because we do as.numeric() which transforms into 1 and 2
        # And we expect SVM to return same labels as passed
        if(length(object$levels) == 2){
          # Binary case
          prediction <- factor(object$levels[(prediction+1)/2 + 1], levels = object$levels)
        }else{
          prediction <- factor(object$levels[prediction], levels = object$levels)
        }
        
      }
      
      prediction
    }
  }

  setMethod("print", "Rcpp_SVMClient", print.svm)
  setMethod("predict", "Rcpp_SVMClient", predict.svm.gmum)
  setMethod("plot", "Rcpp_SVMClient",  plot.svm)
  setMethod("summary", "Rcpp_SVMClient", summary.svm)
  setMethod("show", "Rcpp_SVMClient", summary.svm)


# Add (very basic) support for caret
  
  copy <- function(x) x
  
  gmum.r.svm.radial.params=c("C", "gamma")
  gmum.r.svm.radial.params.classes=c("double", "double")
  
  gmum.r.svm.linear.params=c("C")
  gmum.r.svm.linear.params.classes=c("double")
  
  gmum.r.svm.poly.params=c("C", "gamma", "degree", "coef0")
  gmum.r.svm.poly.params.classes=c("double", "double", "double", "double")
  
  caret.gmumSvmRadial <<- list(label = "gmum.r.svmRadial",
                               library = c("gmum.r"),
                               type = "Classification",
                               parameters = data.frame(parameter = gmum.r.svm.radial.params,
                                                       class = gmum.r.svm.radial.params.classes,
                                                       label = gmum.r.svm.radial.params),
                               grid = function(x, y, len = NULL) {
                                 # We pass tuning grid manually. 
                                 expand.grid(C=10^(-7:11),
                                             gamma=10^(-10:10))
                               },
                               fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
                                 ## First fti the pls model, generate the training set scores,
                                 ## then attach what is needed to the random forest object to 
                                 ## be used late
                                 x.df = as.data.frame(x)
                                 x.df$y = as.numeric(y)
                                 param$kernel = 'linear'
                                 
                                 if(is.null(param$gamma)){
                                   param$gamma = 1
                                 }else{
                                   param$kernel = 'rbf'
                                 }
                                 if(is.null(param$degree)){
                                   param$degree = 3
                                 }else{
                                   param$kernel = 'poly'
                                 }
                                 if(is.null(param$coef0)){
                                   param$coef0 = 0
                                 }
                                 
                                 
                                 sv <- gmum.r::SVM(
                                   x=x,
                                   y=y,
                                   C = param$C,
                                   gamma = param$gamma,
                                   degree = param$degree,
                                   coef0 = param$coef0,
                                   probability = classProbs,
                                   kernel = param$kernel,
                                   ...
                                 )
                                 
                                 return(sv)
                               },
                               predict = function(modelFit, newdata, submodels = NULL) {  
                                 as.factor(predict(modelFit, newdata))
                               },
                               prob = function(modelFit, newdata, submodels = NULL) {  
                                 predict(modelFit, newdata)
                               },
                               varImp = NULL,
                               levels = function(x) {levels(x$.getY())},
                               sort = function(x) x[order(x[,1]),]
  )
  
  caret.gmumSvmLinear.loc <- copy(caret.gmumSvmRadial)
  caret.gmumSvmPoly.loc <- copy(caret.gmumSvmRadial)
  
  
  caret.gmumSvmLinear.loc$parameters <- data.frame(parameter = gmum.r.svm.linear.params,
                                                class = gmum.r.svm.linear.params.classes,
                                                label = gmum.r.svm.linear.params)
  
  caret.gmumSvmLinear.loc$grid <- function(x, y, len = NULL) {
    expand.grid(C=10^(-7:11))
  }
  
  caret.gmumSvmPoly.loc$grid <- function(x, y, len = NULL) {
    expand.grid(C=10^(-7:11), gamma= 10^(-10:10), coef0=c(0,1,10), degree=c(2,3,4) )
  }
  
  
  caret.gmumSvmPoly.loc$parameters <- data.frame(parameter = gmum.r.svm.poly.params,
                                             class = gmum.r.svm.poly.params.classes,
                                             label = gmum.r.svm.poly.params)
  
  caret.gmumSvmPoly <<- caret.gmumSvmPoly.loc
  caret.gmumSvmLinear <<- caret.gmumSvmLinear.loc
  
})


