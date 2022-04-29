# Code

These are the Google Colab Notebooks I used to explore and preprocess the datasets, as well as run them on the proposed algorithm.

Feel free to run them and analyze what they do. I used Google Colab to edit these files. To access each one simply open it and click the "Open in Colab" button, at the top left corner.

Each notebook contains two main sections:

### Task 1

#### 1. Description
   
This section presents the dataset in question. It explains its original purpose, the features it contains, and where the dataset was retrieved from (the Kaggle links are provided inside the notebook, but you can also find them in the datasets folder ;) ~/project/datasets).

#### 2. Data exploration and visualization

This part deals with the loading and preprocessing of the data; i.e., encoding the label and categorical values, dropping useless features and missing values, plotting feature histograms and target distributions, etc.

#### 3. Surrogates and splitting

This last part creates the imbalanced surrogates for each dataset and splits them into training and testing.

### Task 2

#### 1. Scaling and baseline

In this section the original sets and surrogates are scaled, and a Random Forest outputs a baseline score using 10-fold cross-validation for each set.

#### 2. Algorithm

This final section performs a manual gridsearch on each set to suggest a number of components (for PCA) and clusters (for K-Means). These parameters are visualized with a grid of Principal Components plotted against each other and an Elbow and Silhouette plot. Each dataset is then split into 10 stratified folds and the algorithm is run for each of these 10 iterations, and the full training and testing sets are passed one more time to output the generalization score. Finally, the inbalance within clusters is visualized.

### The utilities script

The `imb_utils.py` script contains several routines used multiple times throughout the notebooks, and it has to be loaded so the lines are executed. To do so...
1. Download the file to your computer.
2. Drop the file onto your Google Drive and store its path.
3. Pase the path in the 

> # Importing utility routines developed for the project
> !cp /content/drive/MyDrive/UNIVERSITY-OF-ESSEX/CE888-DataScience/Assignment/imb_utils.py /content
