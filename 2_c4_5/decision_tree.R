# Removing Variables from the Local Environment
rm(list=ls())
# Sys.setenv(LANG = "en")
library(tidyverse)
library(caret)
library(RWeka)
library(xtable)
library(MLmetrics) # metrics like Acc, Rec, Sens, FSC
# library(C50)
library(modeldata)
library(ROCR)
library(e1071)
# Clear console
cat("\014")

# -----------------------------------------------------------------------
# Load Iris dataset
dataset = read.csv("./data/iris.data", header=TRUE, sep=",")
# Shuffle dataset
rows = sample(nrow(dataset))
dataset = dataset[rows,]
# Set class as factor
dataset$Class = as.factor(dataset$Class)
# Set dataset name
dataset_name = "iris"

# Load Glass dataset
dataset = read.csv("./data/glass.data", header=TRUE, sep=",")
dataset = dataset[-1]
# Shuffle dataset
rows = sample(nrow(dataset))
dataset = dataset[rows,]
# Set class as factor
dataset$Class = as.factor(dataset$Class)
# Set dataset name
dataset_name = "glass"

# Load Wine dataset
dataset = read.csv("./data/wine.data", header=TRUE, sep=",")
# Shuffle dataset
rows = sample(nrow(dataset))
dataset = dataset[rows,]
# Set class as factor
dataset$Class = as.factor(dataset$Class)
# Set dataset name
dataset_name = "wine"

# Load Seeds dataset
dataset = read.csv("./data/seeds_dataset.txt", header=TRUE, sep="\t")
# Shuffle dataset
rows = sample(nrow(dataset))
dataset = dataset[rows,]
# Set class as factor
dataset$Class = as.factor(dataset$Class)
# Set dataset name
dataset_name = "seeds"

# -----------------------------------------------------------------------
# View dataset
View(dataset)
# Show class statistics
table(dataset$Class)

# -----------------------------------------------------------------------
# Set number of folds (cross-validation)
no_folds = 2
no_folds = 5
no_folds = 10
no_folds = 15

# -----------------------------------------------------------------------
# Calculate metrics
metrics = function(data, lev = NULL, model = NULL) {
  acc = Accuracy(y_pred = data$pred, y_true = data$obs)
  prec = Precision(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  rec = Recall(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  #sens = Sensitivity(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  fsc = F1_Score(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  
  #c(ACC = acc, PREC = prec, SENS = sens, FSC = fsc)
  c(ACC = acc, PREC = prec, REC = rec, FSC = fsc)
}

# -----------------------------------------------------------------------
# K-Fold
tc = trainControl(method = "cv", number = no_folds, summaryFunction = metrics)

# Stratified-K-Fold 1
cvIndex = createFolds(factor(dataset$Class), no_folds, returnTrain = T)
tc = trainControl(index = cvIndex, method = "cv", number = no_folds, summaryFunction = metrics)
# Stratified-K-Fold 2
tc = trainControl(index = cvIndex, method = "repeatedcv", number = no_folds, repeats=no_folds, summaryFunction = metrics)

# -----------------------------------------------------------------------
# Train
parameters = expand.grid(C=c(0.1,0.25,0.5),  M=c(2,15))
model = train(Class~., data = dataset, method = "J48", tuneGrid = parameters, trControl = tc)

# Print metrics
model

# Print tree
model$finalModel

# -----------------------------------------------------------------------
# Plot tree (save as png)
png(filename=paste("out_", dataset_name, "/", dataset_name, "_tree_stratified_", toString(no_folds), ".png", sep=""))
#png(filename=paste("out_", dataset_name, "/", dataset_name, "_tree_", toString(no_folds), ".png", sep=""))
plot(model$finalModel)
dev.off()

# Save tests to .tex
print(xtable(model$results[,1:6], type = "latex"), file = paste("out_", dataset_name, "/", dataset_name, "_tree_stratified_", toString(no_folds), ".tex", sep=""))
#print(xtable(model$results[,1:6], type = "latex"), file = paste("out_", dataset_name, "/", dataset_name, "_tree_", toString(no_folds), ".tex", sep=""))
