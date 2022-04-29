# Imbalanced Datasets

## 'Should imbalanced datasets be "fixed"?'

An "imbalanced dataset" is one that contains many more samples for one class than it does for others. They are found in many real-world applications including medical diagnosis, fraud detection, and text classification, to mention a few.

Even though they are pretty common, popular classification techniques and evaluation metrics perform poorly. Several alternatives have been proposed to "fix" or balance them, including oversampling the minority class, under-sampling the majority class, and modifying existing algorithms or proposing new ones.

## Project description

In this project we develop and analyze the performance of a clustering approach to classify imbalanced datasets. We expect this clustering to separate groups where their class distribution is either more balanced, or exclusively one class, making it easier to train classifiers. The F1 score is used throughout the project as the evaluating metric.

## Project structure

### Task 1 - 'Should imbalanced datasets be "fixed"?'

Three originally balanced classification datasets were found on Kaggle. Each one of them had different feature characteristics: one was fully numerical, a second was mostly categorical, and the third was a combination of both types of features. These were explored, preprocessed and cleaned. Later on, each one was used to create three imbalanced surrogates, showing low (65%), medium (75%) and high (90%) imbalances. Finally, they were split into training and testing.

### Task 2 - Evaluation of a clustering algorithm for imbalanced datasets

This part of the project saw the actual development and analysis of the algorithm.

Each dataset was scaled to standardize continuous numerical values and normalize discrete numerical ones. They were they cross-validated with 10 folds using a Random Forest (RF) classifier to establish a baseline score. Before performing the algorithm, each dataset was passed to a manual grid search to find a good number of components (for PCA) and clusters (for K-Means). To analyze the search's suggested parameters, an Elbow and Silhouette plot and a grid of Principal Components plotted against each other were generated.

To test the algorithm consisted of the following steps...

Using the suggested components and clusters, the training set was transformed with PCA and clustered with K-Means. The grouping prediction was used to isolate samples, store their label imbalance, and a trained classifier. This classifier was a numerical value, if all the samples in a cluster only had one label, or a trained RF, if they had mixed labels. The test set then followed the
same transformation and clustering, its samples were isolated and the corresponding classifiers retrieved to predict a label. All testing clusters were then joined to calculate the F1 score.

To test it, each training dataset was first divided into 10 stratified folds and fed to the model in 10 iterations of training and validation. Finally, the full training and testing sets were passed to the algorithm, plotting the imbalance in their clusters.

#### The datasets

The used datasets are located in the "datasets" folder.

#### The code

The notebooks and utility script used throughout the project are located in the "code" folder.
