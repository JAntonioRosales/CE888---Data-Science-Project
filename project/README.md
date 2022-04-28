# Imbalanced Datasets

## 'Should imbalanced datasets be "fixed"?'

An "imbalanced dataset" is one that contains many more samples for one class than it does for others. They are found in many real-world applications including medical diagnosis, fraud detection, text classification, to mention a few.

Even though they are pretty common, popular classification techniques and evaluation metrics perform poorly. Several alternatives have been proposed to "fix" or balance them, including oversampling the minority class, under-sampling the majority class, and modifying the cost of predicting classes in algorithms.

## Project description

In this project, we propose and analyze the performance of a method based on clustering an imbalanced dataset. What is expected is for these clusters to be composed exclusively of one class, or having a greater balance between classes, making it easier to train classifiers and predicting unseen samples correctly.

## Project structure

### Part 1

Three different, and initially balanced, classification datasets were found on Kaggle. These were explored, preprocessed and cleaned. Later on, each one was used to create three imbalanced surrogates, showing low (65%), medium (75%) and high (90%) imbalance. Finally, they were split into training and testing.

### Part 2

The second part of the project will see the actual training and analysis of the algorithm.

A baseline Random Forest will be trained. The features will them be scaled and selected using PCA. 10 stratified folds will be generated and 9 will be used to find the number of clusters of the dataset using the Elbow and Silhouette methods. K-Means will then be run to generate the clusters, hoping to have groups with either one class or a more balanced target distribution. In the second case, a Random Forest will be trained. New samples will be assigned to their nearest cluster, and will be labeled as the only class, or predicted by the trained Random Forest. This will be repeated for each permutation of the 10 folds. Finally, this method will be compared to the baseline model and basic re-sampling techniques using F-Measure as evaluation metric and permutation tests to determine if this approach is better for the classification of imbalanced datasets than trying to artificially balance them.

#### The datasets

The datasets used are in the "datasets" folder.

#### The code

The notebooks and utilities script used throughout the project are in the "code" folder.
