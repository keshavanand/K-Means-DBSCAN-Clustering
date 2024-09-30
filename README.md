# K-Means-DBSCAN-Clustering
**Report on Olivetti Faces Dataset Analysis**

**Step 1: Retrieve and Load the Olivetti Faces Dataset**

The Olivetti Faces dataset was successfully retrieved and loaded using sklearn.datasets. This dataset contains 400 grayscale images of faces, where each image is sized 64x64 pixels. The dataset includes 40 different subjects, with 10 images per subject. Each image is then flattened to create a feature vector of length 4096 for further processing.

**Step 2: Split the Dataset**

To ensure a balanced representation of each subject in the training, validation, and test sets, stratified sampling was employed. The dataset was divided into three parts:

- **Training Set**: 280 images (70%)
- **Validation Set**: 60 images (15%)
- **Test Set**: 60 images (15%)

This stratification ensures that there are 7 images per subject in the training set and 1 image per subject in both the validation and test sets. The rationale behind this split ratio is to provide a robust training dataset while allowing sufficient data for validation and testing to accurately assess model performance.

**Step 3: K-Fold Cross Validation**

Using K-fold cross-validation, the classifier was trained to predict which person is represented in each picture. A 5-fold cross-validation was implemented, yielding the following fold accuracies:

- Fold 1: 0.9643
- Fold 2: 0.9107
- Fold 3: 0.9643
- Fold 4: 1.0000
- Fold 5: 0.9286

The average accuracy across all 5 folds was calculated to be approximately **0.9536**, indicating strong performance of the classifier.

**Step 4: Dimensionality Reduction Using K-Means**

K-Means was applied to reduce the dimensionality of the dataset. The silhouette score approach was utilized to determine the optimal number of clusters, providing a measure of how similar the images in each cluster are compared to those in other clusters. After fitting K-Means, it was observed that the optimal number of clusters suggested was **2(0.16)** according to silhouette score but as we had 40 different classes the 40(0.145) clusters also show similar silhouette score.![A graph with blue lines and numbers

Description automatically generated]

**Step 5: Train Classifier on Reduced Set**

Using the transformed feature set from K-Means, a classifier was trained similarly to step 3. The validation accuracy on the reduced set was found to be **0.8000**. This indicates a decent performance, although it was lower than the average accuracy obtained in step 3.

**Step 6: DBSCAN Clustering**

DBSCAN was applied to the Olivetti Faces dataset to explore its clustering capabilities. After preprocessing and converting the images into feature vectors, DBSCAN estimated the number of clusters as **10** and identified **118** points as noise.

The silhouette score for this clustering was calculated to be **\-0.0334**, indicating poor clustering performance, likely due to the density of the data and the chosen parameters for eps and min_samples.

**Conclusion**

The analysis of the Olivetti Faces dataset demonstrated the effectiveness of both K-Means and DBSCAN for clustering, albeit with varying results. While K-Means provided a clearer cluster definition, DBSCAN's results showed challenges likely due to the nature of the data and the chosen parameters. Overall, this study highlights the importance of choosing appropriate clustering algorithms and parameters when working with complex datasets.
