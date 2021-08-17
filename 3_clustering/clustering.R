### to install package - use install.packages("pkg_name")
# Removing Variables from the Local Environment
rm(list=ls())
library(cluster) # pam() purity()
library(stats) # kmeans()
library(clusterCrit) # intCriteria()
library(ggplot2) # plotting
library(factoextra) # plotting
library(funtimes) # purity()
library(xtable) # xtable()
library(caret) # pre-processing
# library(NbClust) # Determining the Best Number of Clusters in a Data Set
# library(clusterSim) # Searching for Optimal Clustering Procedure for a Data Set - data.Normalization() - issue
# Clear console
cat("\014")

# -----------------------------------------------------------------------
# Load Iris dataset
dataset = read.csv("./data/iris.data", header=TRUE, sep=",")
# Shuffle dataset
rows = sample(nrow(dataset))
dataset = dataset[rows,]
# Set ds_data (without Class column) and ds_class (class only) as.matrix
ds_data = as.matrix(dataset[,-5])
ds_class = as.matrix(dataset[,5])
# Set dataset name
dataset_name = "iris"

# Load Glass dataset
dataset = read.csv("./data/glass.data", header=TRUE, sep=",")
dataset = dataset[-1]
# Shuffle dataset
rows = sample(nrow(dataset))
dataset = dataset[rows,]
# Set ds_data (without Class column) and ds_class (class only) as.matrix
ds_data = as.matrix(dataset[,-10])
ds_class = as.matrix(dataset[,10])
# Set dataset name
dataset_name = "glass"

# Load Wine dataset
dataset = read.csv("./data/wine.data", header=TRUE, sep=",")
# Shuffle dataset
rows = sample(nrow(dataset))
dataset = dataset[rows,]
# Set ds_data (without Class column) and ds_class (class only) as.matrix
ds_data = as.matrix(dataset[,-1])
ds_class = as.matrix(dataset[,1])
# Set dataset name
dataset_name = "wine"

# Load Seeds dataset
dataset = read.csv("./data/seeds_dataset.txt", header=TRUE, sep="\t")
# Shuffle dataset
rows = sample(nrow(dataset))
dataset = dataset[rows,]
# Set ds_data (without Class column) and ds_class (class only) as.matrix
ds_data = as.matrix(dataset[,-8])
ds_class = as.matrix(dataset[,8])
# Set dataset name
dataset_name = "seeds"

# -----------------------------------------------------------------------
# View dataset, data and class
View(dataset)
View(ds_data)
View(ds_class)
# Show class statistics
table(ds_class)

# -----------------------------------------------------------------------
# Data Pre-Processing - Standardize - mean = 0
head(ds_data)
summary(ds_data)
preproc1 = preProcess(ds_data, method=c("center", "scale"))
norm1 = predict(preproc1, ds_data)
head(norm1)
summary(norm1)
ds_data = norm1

# Data Pre-Processing - Normalize (Min-max Scaling) - min = 0; max = 1
head(ds_data)
summary(ds_data)
preproc2 <- preProcess(ds_data, method=c("range"))
norm2 <- predict(preproc2, ds_data)
head(norm2)
summary(norm2)
ds_data = norm2

# -----------------------------------------------------------------------
# Create metrics table
k_param_range = 1:10

metrics = matrix(nrow=k_param_range[length(k_param_range)], ncol=5)
dimnames(metrics) = list(c(k_param_range), c("K-param", "DBI", "Dunn", "Silhouette", "Putiry"))

# -----------------------------------------------------------------------
# K-Means algorithm for each clustering param (K) + processing metrics and adding them to metrics table
for(k_param in k_param_range)
{
  sprintf("Processing K-Means clustering and merics for K = %d ...", k_param)
  # k_param = 3
  km_res = kmeans(ds_data, k_param, nstart = 15)
  
  # Clusters for each data instance
  km_res$cluster
  # Plot clusters
  fviz_cluster(km_res, data = ds_data, geom = "point", stand = FALSE, ellipse.type = "norm")
  # Show cluster statistics
  table(km_res$cluster, ds_class)
  
  criteria = intCriteria(ds_data, km_res$cluster, "all")
  purity_val = purity(ds_class, km_res$cluster)
  metrics[k_param, 1] = k_param
  metrics[k_param, 2] = criteria$davies_bouldin # lepiej mniej
  metrics[k_param, 3] = criteria$dunn # więcej lepiej
  metrics[k_param, 4] = criteria$silhouette # więcej lepiej
  metrics[k_param, 5] = purity_val$pur # więcej lepiej (% dopasowania)
}

# -----------------------------------------------------------------------
# PAM algorithm for each clustering param (K) + processing metrics and adding them to metrics table
for(k_param in k_param_range)
{
  sprintf("Processing PAM clustering and merics for K = %d ...", k_param)
  # k_param = 3
  pam_res = pam(ds_data, k_param, metric="manhattan")
  #pam_res = pam(ds_data, k_param, metric="euclidean")
  
  # Clusters for each data instance
  pam_res$cluster
  # Plot clusters
  fviz_cluster(pam_res, data = ds_data, geom = "point", stand = FALSE, ellipse.type = "norm")
  # Show cluster statistics
  table(pam_res$cluster, ds_class)
  
  criteria = intCriteria(ds_data, pam_res$cluster, "all")
  purity_val = purity(ds_class, pam_res$cluster)
  metrics[k_param, 1] = k_param
  metrics[k_param, 2] = criteria$davies_bouldin
  metrics[k_param, 3] = criteria$dunn
  metrics[k_param, 4] = criteria$silhouette
  metrics[k_param, 5] = purity_val$pur
}

metrics

# -----------------------------------------------------------------------
# Select file name
filename_part = "k-means"
filename_part = "pam-euclidean"
filename_part = "pam-manhattan"

png(filename=paste("out_", dataset_name, "/", dataset_name, "_metrics_", filename_part, ".png", sep=""))

# Plot results
plot(x=metrics[,1], metrics[,2], type="o", main=paste("Dataset", dataset_name, filename_part),  xlab="Number of clusters", 
     ylab="Metrics value", col="red", ylim=c(0,2), tck=1)
points(metrics[,1], metrics[,3], col="green", pch=1)
lines(metrics[,1], metrics[,3], col="green", lty=1)
points(metrics[,1], metrics[,4], col="blue", pch=1)
lines(metrics[,1], metrics[,4], col="blue", lty=1)
points(metrics[,1], metrics[,5], col="orange", pch=1)
lines(metrics[,1], metrics[,5], col="orange", lty=1)

legend("top", legend=c("DBI", "Dunn", "Silhouette", "Putiry"), 
       col=c("red", "green", "blue", "orange"), lty=1, pch=1 , ncol=2)

# Save file
dev.off()

# -----------------------------------------------------------------------
# Metrics to tex file
print(xtable(metrics[-1,1:5], type = "latex"), file = paste("out_", dataset_name, "/", dataset_name, "_metrics_", filename_part, ".tex", sep=""))

# -----------------------------------------------------------------------
# K-Means algorithm
filename_part = "k-means"

k_param = 3
km_res = kmeans(ds_data, k_param, nstart = 15)
png(filename=paste("out_", dataset_name, "/", dataset_name, "_metrics_", filename_part, toString(k_param), ".png", sep=""))
fviz_cluster(km_res, data = ds_data, geom = "point", stand = FALSE, ellipse.type = "norm")
dev.off()

# -----------------------------------------------------------------------
# PAM algorithm
filename_part = "pam-euclidean"
filename_part = "pam-manhattan"

k_param = 3
pam_res = pam(ds_data, k_param, metric="manhattan")
#pam_res = pam(ds_data, k_param, metric="euclidean")
png(filename=paste("out_", dataset_name, "/", dataset_name, "_metrics_", filename_part, toString(k_param), ".png", sep=""))
fviz_cluster(pam_res, data = ds_data, geom = "point", stand = FALSE, ellipse.type = "norm")
dev.off()
