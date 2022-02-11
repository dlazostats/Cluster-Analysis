## Cluster Analysis 2
##-------------------
library(mlr3cluster)
library(mlr3)
library(mlr3viz)
library(dplyr)
library(cluster)
library(useful)

### list of measures and learners
mlr_measures$keys("clust")
mlr_learners$keys("clust")

## working directory
script_name <- 'cluster_analysis_2.R'
ruta <- gsub(rstudioapi::getActiveDocumentContext()$path,pattern = script_name,replacement = '')
setwd(ruta)

## data 
#df_mgm<-readRDS("oes.rds")
#task <- mlr_tasks$get("df_mgm")
wineUrl <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine <- read.table(wineUrl, header=FALSE, sep=',',
                   stringsAsFactors=FALSE,
                   col.names=c('Cultivar', 'Alcohol', 'Malic.acid','Ash', 'Alcalinity.of.ash',
                                'Magnesium', 'Total.phenols','Flavanoids', 'Nonflavanoid.phenols',
                                'Proanthocyanin', 'Color.intensity','Hue', 'OD280.OD315.of.diluted.wines',
                                'Proline'))
head(wine)
winec<-wine %>% select(-Cultivar)

## data as task
task <- TaskClust$new(id = "winec", backend = winec)

# Train/predict
train_set<-sample(task$nrow,0.8*task$nrow)
test_set<-setdiff(seq_len(task$nrow),train_set)

## Compare algorithms
design = benchmark_grid(tasks = task,
                        learners = list(lrn("clust.kmeans", centers = 3L),
                                        lrn("clust.pam", k = 3L),
                                        lrn("clust.cmeans", centers = 3L)),
                        resamplings = rsmp("holdout"))
bmr = benchmark(design)
measures = list(msr("clust.silhouette"))
bmr$aggregate(measures)

## kmeans 
learner <- mlr_learners$get("clust.kmeans")
learner$param_set$values = list(centers = 3L)
learner$train(task,row_ids = train_set)

## Prediction
preds = learner$predict(task)

# Pairs plot with cluster assignments
autoplot(preds, task, type = "pca",frame=T)
autoplot(preds, task, type = "sil")


## Using other packages
##----------------------
wineTrain <- wine[, which(names(wine) != "Cultivar")]
set.seed(278613)

## kmeans
wineK3 <- kmeans(x=wineTrain, centers=3)
plot(wineK3, data=wineTrain)
plot(wineK3, data=wine, class="Cultivar")

## Using the useful package
wineBest <- FitKMeans(wineTrain, max.clusters=20, 
                     nstart=25, seed=278613) 
wineBest
PlotHartigan(wineBest)

wineK3N25<-kmeans(x=wineTrain, centers=3)
table(wine$Cultivar, wineK3N25$cluster)
plot(table(wine$Cultivar, wineK3N25$cluster))

## gap statistics
theGap <- clusGap(wineTrain, FUNcluster=pam, K.max=20)
gapDF <- as.data.frame(theGap$Tab)
head(gapDF)

ggplot(gapDF, aes(x=1:nrow(gapDF))) +
       geom_line(aes(y=logW), color="blue") +
       geom_point(aes(y=logW), color="blue") +
       geom_line(aes(y=E.logW), color="green") +
       geom_point(aes(y=E.logW), color="green") +
       labs(x="Number of Clusters")

# gap curve
ggplot(gapDF, aes(x=1:nrow(gapDF))) +
       geom_line(aes(y=gap), color="red") +
       geom_point(aes(y=gap), color="red") +
       geom_errorbar(aes(ymin=gap-SE.sim, ymax=gap+SE.sim), color="red") +
      labs(x="Number of Clusters", y="Gap")

