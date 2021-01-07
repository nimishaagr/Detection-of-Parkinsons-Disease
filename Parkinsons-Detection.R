#------------------------------------------------------------------------------#
#INSTALLING MISSING PACKAGES FROM CRAN
#PACKAGES NEEDED FOR THE REPORT
if(!require(rmarkdown)) install.packages("rmarkdown", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")
#for installing TinyTeX (LaTeX distribution) using the 
#tinytex R package
tinytex::install_tinytex() 

#PACKAGES NEEEDED FOR THE CODE
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomcoloR)) install.packages("randomcoloR", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(MLmetrics)) install.packages("MLmetrics", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org", dependencies=TRUE)
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

#------------------------------------------------------------------------------#
#LOADING THE REQUIRED PACKAGES
library(tidyverse)
library(caret)
library(randomcoloR)
library(GGally)
library(ggcorrplot)
library(reshape2)
library(MLmetrics)
library(caTools)
library(e1071)
library(nnet)
library(rpart)
library(gbm)
library(randomForest)

#------------------------------------------------------------------------------#
#IMPORTING DATA INTO R
url <- 
"https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

#creating a temporary file to download data into
temp <- tempfile()

#downloading from the url
download.file(url,temp)

#looking at the first few lines of the file
read_lines(temp, n_max = 3)
#separator is comma, file contains header

#reading the file into an R object
parkinsons <- read_csv(temp)

#unlinking the temporary file
unlink(temp)

#------------------------------------------------------------------------------#
#PRE-PROCESSING
#removing the column containing names since I want to make a general predictor,
#which can be extended to all 
parkinsons <- parkinsons %>% select(-name)

#changing the column names since they contain characters that may throw up errors
colnames(parkinsons) <- make.names(colnames(parkinsons))

#checking zero variance
nearZeroVar(parkinsons)

#identifying correlated predictors 
corr <- parkinsons %>% select(-status) %>% cor() %>% round(1)
#flagging predictors for removal with a cutoff of 0.75
highlyCorr <- findCorrelation(corr, cutoff=0.75)
#removing the columns from parkinsons
parkinsons <- parkinsons %>% 
  select(-status) %>% 
  select(-all_of(highlyCorr)) %>% 
  cbind(status=parkinsons$status)

#------------------------------------------------------------------------------#
#SPITTING THE DATA SET INTO TRAIN AND TEST SET
set.seed(730, sample.kind = "Rounding")
test_index <- createDataPartition(parkinsons$status, times=1, p=0.2, list=FALSE)
test_set <- parkinsons[test_index,]
train_set <- parkinsons[-test_index,]

#------------------------------------------------------------------------------#
#EXPLORATORY DATA ANALYSIS
head(train_set)
str(train_set)
summary(train_set)

#checking for NAs
sum(is.na(train_set))

#setting the theme
theme_set(theme_minimal())

#checking the distribution of the outcome 'status'
train_set %>% mutate(status=factor(status, labels=c("Healthy","Parkinson's"))) %>% 
  ggplot(aes(status, fill=status)) +
  geom_bar() + 
  scale_fill_manual(values=c("mediumseagreen", "palevioletred")) +
  theme(legend.position = "none") +
  labs(x="Outcome (status)", y="Count",
       title="Distribution of Outcome")
#imbalanced classification therefore use kappa instead of accuracy as the metric

#checking the distribution of all predictors
train_set %>% select(-status) %>% gather() %>%
  group_by(key) %>% mutate(mean=mean(value)) %>% ungroup() %>%
  ggplot(aes(value, y=..count.., fill=key))+
  geom_density() +
  geom_vline(aes(xintercept=mean)) + 
  facet_wrap(.~key, scales="free", ncol=4) +
  scale_fill_manual(values=distinctColorPalette(10)) +
  theme(legend.position = "none") +
  labs(x="Predictor", y="Count",
       title="Distribution of Predictors")

#parallel coordinates chart
train_set %>% mutate(status=factor(status, labels=c("Healthy","Parkinson's"))) %>%
  ggparcoord(columns=c(1:10), groupColumn = 11, scale="std", alphaLines=0.4) +
  scale_color_manual(values=c("mediumseagreen", "palevioletred")) + 
  geom_line(size=1, alpha=0.3) +
  theme(axis.text.x = element_text(angle=40, hjust=1))+
  labs(x="Predictor", y="Standardized Value",
       title="Parallel Coordinates Chart of Outcomes for all Predictors",
       col="Outcome\n(status)")

#boxplots with jitter
train_set %>% mutate(status=factor(status, labels=c("Healthy","Parkinson's"))) %>%
  melt(id.vars="status") %>%
  ggplot(aes(status, value, fill=status)) + 
  facet_wrap(.~variable, scales="free", ncol=4) +
  geom_boxplot() + geom_jitter(color="grey", alpha=0.5) +
  theme(legend.position = "none") +
  scale_fill_manual(values=c("mediumseagreen", "palevioletred")) +
  labs(x="Outcome (status)", y="Value",
       title="Boxplots of Predictors vs. Status")

#checking correlation of predictors with the outcome
train_corr <- round(cor(train_set),1)
train_p <- cor_pmat(train_set)
train_corrp <- data.frame(
  "feature"=colnames(train_set),
  "r"=train_corr[,11],
  "p"=train_p[,11],
  row.names = NULL
) %>% filter(feature!="status")
train_corrp %>% mutate(feature=reorder(feature,r)) %>%
  ggplot(aes(feature, r, shape=(p<0.05), col=fct_rev(as.factor(r)))) + 
  geom_point(size=5) +
  geom_hline(yintercept=0, col="black") +
  ylim(-0.6,0.6) +
  coord_flip() +
  scale_color_brewer(palette="RdBu") +
  guides(shape = FALSE) +
  labs(y="Pearson Correlation", x="Predictor",
       title="Correlation of Predictors with the Outcome",
       col="Pearson\nCorrelation\n(rounded)")
#all values are significant

#------------------------------------------------------------------------------#
#MACHINE LEARNING
#outcome is 'status', 10 features available to use
#will use the confusion matrix as a metric along with accuracy
#false negatives can give false assurances, and false positives can send them to get needless, expensive medical tests
#positive class for confusionmatrix should be "1" since in medical science
#will use kappa as the metric in train functions since the classes of our outcome are imbalanced

#creating a copy of the train_set with status as a factor variable for use in ML algorithms
train_fct <- train_set %>% mutate(status=as.factor(status))

#setting the standard for 10-fold cross validation will be done with each algorithm
#because seeds need to be set according to tuning parameters for reproducible values

#LOGISTIC REGRESSION
#setting all the seeds for cross validation to get reproducible numbers
set.seed(730, sample.kind = "Rounding")
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 1)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(730, sample.kind = "Rounding")
train_logireg <- train(status ~ ., method = "LogitBoost", 
                       data = train_fct,
                       trControl = control,
                       metric="Kappa",
                       preProcess = c("center", "scale"),
                       tuneGrid = data.frame(nIter = seq(1, 400, 20)))
train_logireg
#plotting the parameters that were tuned
ggplot(train_logireg, highlight = TRUE) + labs(title="Tuning the Logistic Regression")
#storing the predicted values
predicted_logireg <- predict(train_logireg, test_set)
#computing metrics to assess efficacy of the algorithm
cm_logireg <- confusionMatrix(predicted_logireg, as.factor(test_set$status), positive="1")
kappa_logireg <- cm_logireg$overall["Kappa"]
accu_logireg <- cm_logireg$overall["Accuracy"]
f1_logireg <- F1_Score(predicted_logireg, as.factor(test_set$status), positive="1")


#K-NEAREST NEIGHBOURS
#setting all the seeds for cross validation to get reproducible numbers
set.seed(730, sample.kind = "Rounding")
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 20)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(730, sample.kind = "Rounding")
train_knn <- train(status ~ ., method = "knn", 
                   data = train_fct,
                   trControl = control,
                   metric="Kappa",
                   tuneGrid = data.frame(k = seq(1, 20, 1)),
                   preProcess = c("center", "scale"))
train_knn
#plotting the parameters that were tuned
ggplot(train_knn, highlight = TRUE) + labs(title="Tuning the K-Nearest Neighbours")
#storing the predicted values
predicted_knn <- predict(train_knn, test_set)
#computing metrics to assess efficacy of the algorithm
cm_knn <- confusionMatrix(predicted_knn, as.factor(test_set$status), positive="1")
kappa_knn <- cm_knn$overall["Kappa"]
accu_knn <- cm_knn$overall["Accuracy"]
f1_knn <- F1_Score(predicted_knn, as.factor(test_set$status), positive="1")

#SUPPORT VECTOR MACHINE
#setting all the seeds for cross validation to get reproducible numbers
set.seed(730, sample.kind = "Rounding")
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 15)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(730, sample.kind = "Rounding")
train_svm <- train(status ~ ., method = "svmLinearWeights", 
                   data = train_fct,
                   trControl = control,
                   metric="Kappa",
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(cost = c(.25, .5, 1), weight = c(1:5)))
train_svm
#plotting the parameters that were tuned
ggplot(train_svm, highlight = TRUE) + labs(title="Tuning the Support Vector Machine")
#storing the predicted values
predicted_svm <- predict(train_svm, test_set)
#computing metrics to assess efficacy of the algorithm
cm_svm <- confusionMatrix(predicted_svm, as.factor(test_set$status), positive="1")
kappa_svm <- cm_svm$overall["Kappa"]
accu_svm <- cm_svm$overall["Accuracy"]
f1_svm <- F1_Score(predicted_svm, as.factor(test_set$status), positive="1")

#NEURAL NETWORK
#setting all the seeds for cross validation to get reproducible numbers
set.seed(730, sample.kind = "Rounding")
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 40)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(730, sample.kind = "Rounding")
train_nn <- train(status ~ ., method = "nnet", 
                  data = train_fct,
                  trControl = control,
                  metric="Kappa",
                  preProcess = c("center", "scale"),
                  tuneGrid=expand.grid(size=c(0.1,0.01,0.001,0.0001), decay=1:10))
train_nn
#plotting the parameters that were tuned
ggplot(train_nn, highlight = TRUE) + labs(title="Tuning the Neural Network")
#storing the predicted values
predicted_nn <- predict(train_nn, test_set)
#computing metrics to assess efficacy of the algorithm
cm_nn <- confusionMatrix(predicted_nn, as.factor(test_set$status), positive="1")
kappa_nn <- cm_nn$overall["Kappa"]
accu_nn <- cm_nn$overall["Accuracy"]
f1_nn <- F1_Score(predicted_nn, as.factor(test_set$status), positive="1")

#DECISION TREE
#setting all the seeds for cross validation to get reproducible numbers
set.seed(730, sample.kind = "Rounding")
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 1)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(730, sample.kind = "Rounding")
train_rpart <- train(status ~ ., method = "rpart", 
                   data = train_fct,
                   metric="Kappa",
                   trControl = control,
                   tuneGrid = data.frame(cp = seq(0, 1, len = 25)))
train_rpart
#plotting the parameters that were tuned
ggplot(train_rpart, highlight = TRUE) + labs(title="Tuning the Decision Tree")
#storing the predicted values
predicted_rpart <- predict(train_rpart, test_set)
#computing metrics to assess efficacy of the algorithm
cm_rpart <- confusionMatrix(predicted_rpart, as.factor(test_set$status), positive="1")
kappa_rpart <- cm_rpart$overall["Kappa"]
accu_rpart <- cm_rpart$overall["Accuracy"]
f1_rpart <- F1_Score(predicted_rpart, as.factor(test_set$status), positive="1")

#GRADIENT BOOSTING MACHINE
#setting all the seeds for cross validation to get reproducible numbers
set.seed(730, sample.kind = "Rounding")
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 90)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(730, sample.kind = "Rounding")
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
train_gbm <- train(status ~ ., method = "gbm", 
                   data = train_fct,
                   trControl = control,
                   metric="Kappa",
                   tuneGrid=gbmGrid, 
                   verbose=FALSE)
train_gbm
#plotting the parameters that were tuned
ggplot(train_gbm, highlight = TRUE) + labs(title="Tuning the Gradient Boosting Machine")
#storing the predicted values
predicted_gbm <- predict(train_gbm, test_set) 
#computing metrics to assess efficacy of the algorithm
cm_gbm <- confusionMatrix(predicted_gbm, as.factor(test_set$status), positive="1")
kappa_gbm <- cm_gbm$overall["Kappa"]
accu_gbm <- cm_gbm$overall["Accuracy"]
f1_gbm <- F1_Score(predicted_gbm, as.factor(test_set$status), positive="1")

#RANDOM FOREST
#setting all the seeds for cross validation to get reproducible numbers
set.seed(730, sample.kind = "Rounding")
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, 10)
seeds[[101]] <- sample.int(1000, 1)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, seeds=seeds)

#training the model
set.seed(730, sample.kind = "Rounding")
#tuning for the parameter 'mtry' and storing the best value of 'mtry'
mtry_rf <- train(status ~ ., method = "rf", 
      data = train_fct,
      trControl = control,
      metric="Kappa",
      tuneGrid=data.frame(mtry=1:10))$bestTune
#tuning for nodesize using the best value of 'mtry' computed above
rf_nodesize <- seq(1, 20, 1)
kappa_all_rf <- sapply(rf_nodesize, function(ns){
  set.seed(730, sample.kind = "Rounding")
  train(status ~ ., method = "rf", 
        data = train_fct,
        trControl = control,
        metric="Kappa",
        tuneGrid=data.frame(mtry=mtry_rf),
        nodesize = ns)$results$Kappa
})
#plotting the kappa values against the node sizes
qplot(rf_nodesize, kappa_all_rf) + 
  labs(x="Nodesize", y="Kappa", 
       title="Tuning the Random Forest for 'nodesize'")
#training the final model using the ebst mtry and the best nodesize
set.seed(730, sample.kind = "Rounding")
train_rf <- train(status ~ ., method = "rf", 
                  data = train_fct,
                  trControl = control,
                  metric="Kappa",
                  tuneGrid=data.frame(mtry=mtry_rf),
                  modesize=rf_nodesize[which.max(kappa_all_rf)])
train_rf
#storing the predicted values
predicted_rf <- predict(train_rf, test_set)
#computing metrics to assess efficacy of the algorithm
cm_rf <- confusionMatrix(predicted_rf, as.factor(test_set$status), positive="1")
kappa_rf <- cm_rf$overall["Kappa"]
accu_rf <- cm_rf$overall["Accuracy"]
f1_rf <- F1_Score(predicted_rf, as.factor(test_set$status), positive="1")


#ENSEMBLE (USING OUR TUNED PREDICTIONS)
pred_all <- data.frame(predicted_logireg, predicted_knn, predicted_svm,
                       predicted_nn, predicted_rpart, predicted_gbm, predicted_rf)
predicted_esmb <- as.factor(ifelse(rowMeans(pred_all=="0")>0.5, 0, 1))
cm_esmb <- confusionMatrix(predicted_esmb, as.factor(test_set$status), positive="1")
kappa_esmb <- cm_esmb$overall["Kappa"]
accu_esmb <- cm_esmb$overall["Accuracy"]
f1_esmb <- F1_Score(predicted_esmb, as.factor(test_set$status), positive="1")

#making a data frame of all models and metrics
results <- data.frame(model=c("Logistic Regression", "K Nearest Neighbours",
                              "Support Vector Machine", "Neural Network",
                              "Decision Tree", "Gradient Boosting Machine",
                              "Random Forest", "Ensemble"),
                      kappa=c(kappa_logireg, kappa_knn, kappa_svm, kappa_nn,
                              kappa_rpart, kappa_gbm, kappa_rf, kappa_esmb),
                      accuracy=c(accu_logireg, accu_knn, accu_svm, accu_nn,
                                 accu_rpart, accu_gbm, accu_rf, accu_esmb),
                      F1_Score=c(f1_logireg, f1_knn, f1_svm, f1_nn, 
                                 f1_rpart, f1_gbm, f1_rf, f1_esmb))
results
results %>% mutate(model=reorder(model,kappa)) %>%
  ggplot(aes(model, kappa)) + geom_col(width=0.5, fill="steelblue") + 
  coord_flip() +
  labs(x="Kappa", y="Model",
       title="Models' Performance Summary: Kappa")
results %>% mutate(model=reorder(model,accuracy)) %>%
  ggplot(aes(model, accuracy)) + geom_col(width=0.5, fill="goldenrod3") + 
  coord_flip() +
  labs(x="Accuracy", y="Model",
       title="Models' Performance Summary: Accuracy")
results %>% mutate(model=reorder(model,F1_Score)) %>%
  ggplot(aes(model, F1_Score)) + geom_col(width=0.5, fill="deeppink4") + 
  coord_flip() +
  labs(x="F1 Score", y="Model",
       title="Models' Performance Summary: F1 Score")
