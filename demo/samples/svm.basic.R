#' ---
#' title: "Basic classification on breast cancer dataset"
#' author: ""
#' date: ""
#' output:
#'  html_document:
#'    self_contained: false
#' ---

library(gmum.r)

# We will perform basic classification on breast cancer dataset 
# using LIBSVM with linear kernel
data(svm_breast_cancer_dataset)

# We can pass either formula or explicitly X and Y
svm <- SVM(X1 ~ ., svm.breastcancer.dataset, core="libsvm", kernel="linear", C=10)

pred <- predict(svm, svm.breastcancer.dataset[,-1])

plot(svm, mode="pca")


