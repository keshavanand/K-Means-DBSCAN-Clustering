# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:51:48 2024

@author: arshd
"""
# import necassary libraries
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# Set OMP_NUM_THREADS to 1 to avoid memory leak issues with KMeans on Windows
os.environ["OMP_NUM_THREADS"] = "2"


#load the olivetti faces dataset
X , y = fetch_olivetti_faces(return_X_y=True)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


# Initialize the SVM classifier
classifier = SVC(kernel='linear', random_state=42)

# Initialize Stratified K-Fold Cross-validation
skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)

# List to store accuracy for each fold
accuracies = []

# Cross-validation loop
for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Train the classifier on the current fold
    classifier.fit(X_train_fold, y_train_fold)
    
    # Predict on the validation fold
    y_pred = classifier.predict(X_val_fold)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val_fold, y_pred)
    accuracies.append(accuracy)

    print(f"Fold accuracy: {accuracy:.4f}")

# Average accuracy across the folds
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy across 5 folds: {average_accuracy:.4f}")


# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
cluster_range = range(2, 41)  # Test for 2 to 10 clusters

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train)
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(cluster_range)
plt.grid()
plt.show()

# Choose the optimal number of clusters (with the highest silhouette score)
optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"The optimal number of clusters is: {optimal_n_clusters}")

# Refit K-Means with the optimal number of clusters
kmeans_final = KMeans(n_clusters=40, random_state=42)
X_train_reduced = kmeans_final.fit_transform(X_train)

# Train a classifier on the reduced data
classifier.fit(X_train_reduced, y_train)

# Validation set transformation
X_val_reduced = kmeans_final.transform(X_val)

# Evaluate on the validation set
y_val_pred_reduced = classifier.predict(X_val_reduced)
accuracy_reduced = accuracy_score(y_val, y_val_pred_reduced)
print(f"Validation accuracy on reduced set: {accuracy_reduced:.4f}")

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=100)  # Reduce to 100 components
X_pca = pca.fit_transform(X_scaled)

db = DBSCAN(eps=50, min_samples=6).fit(X_scaled)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

silhouette_avg = silhouette_score(X_pca, labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

