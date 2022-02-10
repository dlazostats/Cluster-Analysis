## Cluster Analysis
##-------------------
library(mlr3cluster)
library(mlr3)
library(mlr3viz)

### list of measures and learners
mlr_measures$keys("clust")
mlr_learners$keys("clust")

# Data
task = mlr_tasks$get("usarrests")

# Train/predict
train_set<-sample(task$nrow,0.8*task$nrow)
test_set<-setdiff(seq_len(task$nrow),train_set)

# Learner
learner <- mlr_learners$get("clust.kmeans")
learner$train(task,row_ids = train_set)

# Prediction
preds<-learner$predict(task,row_ids = test_set)
preds

## Benchmark and evaluation (Evaluate three different algorithms)
design = benchmark_grid(tasks = tsk("usarrests"),
         learners = list(lrn("clust.kmeans", centers = 3L),
                         lrn("clust.pam", k = 3L),
                         lrn("clust.cmeans", centers = 3L)),
          resamplings = rsmp("holdout"))
print(design)

# execute benchmark
bmr = benchmark(design)

# define measure
measures = list(msr("clust.silhouette"))
bmr$aggregate(measures)

## Visualization
learner = mlr_learners$get("clust.kmeans")
learner$param_set$values = list(centers = 3L)
learner$train(task)
preds = learner$predict(task)

# Task visualization
autoplot(task)

# Pairs plot with cluster assignments
autoplot(preds, task)

# Silhouette plot with mean silhouette value as reference line
autoplot(preds, task, type = "sil")

# Performing PCA on task data and showing cluster assignments
autoplot(preds, task, type = "pca")
autoplot(preds, task, type = "pca",frame=T)

## hierarchical clustering
##---------------------------
task = mlr_tasks$get("usarrests")
learner = mlr_learners$get("clust.agnes")
learner$train(task)

# Simple dendrogram
autoplot(learner)

autoplot(learner,
         k = learner$param_set$values$k, rect_fill = TRUE,
         rect = TRUE, rect_border = c("red", "cyan"))
