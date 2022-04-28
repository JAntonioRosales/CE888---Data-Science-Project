
# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score as f1
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.preprocessing import StandardScaler



def imbalanced_surrogate(majority, minority, imb):
    '''
    This routine takes in two datasets and creates a third, imbalanced one,
    sampled from the minority dataset.

    The desired imbalance is the final percentage the majority dataset's label
    will represent (e.g., if imb == 0.9 and majority_label == 0, 90% of the new
    dataset's label will be 0, and the rest will be 1).

    majority: majority dataframe (pandas dataframe)
    minority: minority dataframe (pandas dataframe)
    imb: imbalance percentage (float from 0 to 1)
    '''

    # Subsampling
    no_min_samples = int(majority.shape[0] * ((1/imb) - 1))
    min_indices = np.random.choice(range(minority.shape[0]), size=no_min_samples, replace=False) # replace=False to not get the same sample more than once
    min_samples = minority.iloc[min_indices]

    # Surrogate dataset
    return pd.concat([majority, min_samples])



class CategoricalMaxScaler(BaseEstimator, TransformerMixin):
    '''
    This object creates a scaler to normalize a column of values similar to
    scikit-learn's MinMaxScaler. Instead of considering the minimum and maximum
    values of a column, this scaler only takes the maximum value and divides
    all of the column by this number.
    '''

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # For each column of interest
        for i in range(len(self.features)):

            # Find max_val
            col_name = self.features[i]
            col = X[col_name].copy()
            max_val = np.unique(col.values)[-1]

            # Divide entirely by max_val
            new_col = col/max_val

            # Replace the column
            X[col_name] = new_col

        return X



def scale_dataframe(df, cat_feat, num_feat, rem_feat):
    '''
    This routine scales a dataframe by columns using ColumnTransformer and
    returns the processed dataframe.

    df: dataframe (pandas dataframe)
    cat_feat: names of categorical columns (list of strings)
    num_feat: names of numerical columns (list of strings)
    rem_feat: names of remaining columns (list of strings)
    '''

    # Define the scaling for columns
    colTransformer = ColumnTransformer([
        ('num', StandardScaler(), num_feat),
        ('cat', CategoricalMaxScaler(cat_feat), cat_feat)
        ],
        remainder='passthrough'  # leave the rest of the columns untouched
        )

    # Scale
    processed_arr = colTransformer.fit_transform(df) # array

    # Create dataframe from array
    feat_names = num_feat + cat_feat + rem_feat
    return pd.DataFrame(processed_arr, columns=feat_names)



def separate_feats_lbl(set_name, df1, df2=None, df3=None, df4=None):
    '''
    This routine separates the features and labels of one, or four dataframes.
    When 4 sets are passed, it returns a list with four sets of features and
    another one with their corresponding four sets of labels. When 1 set is
    passed, it returns its features and its labels separately.

    set_name: name of the dataset we are working on (string)
    df1: dataframe 1 (pandas dataframe)
    df2: dataframe 2 (pandas dataframe)
    df3: dataframe 3 (pandas dataframe)
    df4: dataframe 4 (pandas dataframe)
    '''

    Xs = []
    ys = []
    dfs = [df1, df2, df3, df4]

    # Name of the label depending on the dataset
    if set_name == 'songs': lbl_name = 'target'
    elif set_name == 'wine': lbl_name = 'quality'
    elif set_name == 'flight': lbl_name = 'satisfaction'

    # Condition for 1 dataset only
    if df2 is None:
        dfs = [df1]

    # Separate features and label and store them
    for df in dfs:
        d = df.copy()
        y = d[lbl_name].copy()
        X = d.drop(columns=[lbl_name]).copy()
        Xs.append(X)
        ys.append(y)

    return Xs, ys



def brute_gridsearch(X, y):
    '''
    This routine looks for the best combination of components (for PCA) and
    clusters (for KMeans). It iterates through a list of components and
    clusters, adds the cluster prediction to the features dataset, and uses
    it to cross-validate a Random Forest Classifier to predict the label with
    F1 score as the perfomance metric. The best parameters are those with the
    highest F1 score.

    X: features (numpy array)
    y: labels (numpy array)
    '''

    # Parameters to try (n_components for PCA and n_clusters for KMeans)
    comp_list = [n_components for n_components in range(2, X.shape[1]+1)]
    k_list = [n_clusters for n_clusters in range(2, 21) if n_clusters % 2 == 0]

    # Lists to store parameters and their score
    params = []
    scores_mean = []
    scores_stdv = []

    # Counter for reference
    count = 0

    # Iterate over the PCA components
    for components in comp_list:
        pca = PCA(n_components=components)
        X_pca = pca.fit_transform(X)

        # Iterate over the KMeans clusters
        for k in k_list:

            # Show progress
            if k % 20 == 0:
                count += 1
                print(f"Computing... [{count}/{len(comp_list)}]")

            km = KMeans(n_clusters=k)
            y_pred = km.fit_predict(X_pca)

            # Add the cluster as an extra feature
            X_transf = np.hstack((X_pca, y_pred.reshape(-1, 1)))

            # Train a classifier to predict the original labels
            clf = RandomForestClassifier()
            scores_cv = cross_val_score(clf, X_transf, y, cv=10, scoring = make_scorer(f1))

            # Store parameters and scores
            params.append((components, k))
            scores_mean.append(scores_cv.mean())
            scores_stdv.append(scores_cv.std())

    # Find the best parameters
    max_score, score_idx = np.max(scores_mean), np.argmax(scores_mean)
    best_components, best_k, score_stdv = params[score_idx][0], params[score_idx][1], scores_stdv[score_idx]

    return best_components, best_k, max_score, score_stdv



def plot_identity_pca(X, y, set_abbrv=None):
    '''
    This routine creates a multiple-plot graph to visualize Principal
    Components (PC) plotted against each other, in a sort of identity matrix.

    The first row keeps PC1 constant in the x-axis, and plots it against PC1,
    PC2, PC3, ... PCn in the y-axis.

    The second row keeps PC2 constant in the x-axis, and plots it against PC1,
    PC2, PC3, ... PCn in the y-axis.

    X: features (numpy array)
    y: labels (numpy array)
    set_abbrv: abbreviation of set's balance (string)
    '''

    cols = X.shape[1]
    fig = plt.figure(figsize=(cols*5, cols*5))

    gs0 = gridspec.GridSpec(1,1)
    gs00 = gridspec.GridSpecFromSubplotSpec(cols, cols, subplot_spec=gs0[0])

    for i in range(cols):
        for j in range(cols):
            ax00 = fig.add_subplot(gs00[i, j])
            ax00.scatter(X[:, i], X[:, j], c=y)
            if i == 0: ax00.set_title('PC %d' % (j + 1), fontsize=30)
            if j == 0: plt.ylabel('PC %d' % (i + 1), fontsize=30)
            #plt.xlabel('PC %d' % (i + 1))
            #plt.ylabel('PC %d' % (j + 1))

    fig.suptitle('Principal Components against each other coloured by label', fontsize=50)

    # Save and show
    if set_abbrv is not None:
        plt.savefig('pca_plots_' + set_abbrv + '.pdf', dpi=1200, bbox_inches='tight')
    plt.show()



def elbow_silhouette(X,set_abbrv=None):
    '''
    This routine runs KMeans in a range of clusters similar to brute_gridsearch
    and calculates the inertia and silhouette scores for each iteration.
    The results are then displayed.

    X: features (numpy array)
    set_abbrv: abbreviation of set's balance (string)
    '''

    inertias, sil = [], []
    ran = [n_clusters for n_clusters in range(2,21) if n_clusters % 2 == 0]

    # Run KMeans
    for k in ran:
        kmeans = KMeans(n_clusters=k)
        y_pred = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        sil.append(silhouette_score(X, y_pred))

    # Plot
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(20,5))
    fig.suptitle('Elbow and Silhouette plots', fontsize=16)
    plt.setp(axs, xticks=np.arange(2, 21, 2))

    # Elbow (inertia vs number of clusters)
    axs[0].plot(ran, inertias, 'o-')
    axs[0].set_title('Elbow plot')
    axs[0].set_xlabel('Clusters (k)')
    axs[0].set_ylabel('Inertia')

    # Silhouette (silhouette score vs number of clusters)
    axs[1].plot(ran, sil, 'o-')
    axs[1].set_title('Silhouette plot')
    axs[1].set_xlabel('Clusters (k)')
    axs[1].set_ylabel('Silhouette score')

    # Save and show
    if set_abbrv is not None:
        plt.savefig('elb_sil_' + set_abbrv + '.pdf', dpi=1200, bbox_inches='tight')
    plt.show()



def cluster_and_balance(x_tr, y_tr, x_te, y_te, comps, k):
    '''
    This routine performs the proposed algorithm to classify imbalanced sets.

    The first part uses the TRAINING set.
    It begins by reducing dimensionality with PCA and clustering with KMeans.
    The predicted cluster is added to the feature set and is used to isolate
    the samples in each cluster. For each cluster, its imbalance and a
    classifier are stored. If the cluster's samples have mixed labels, a Random
    Forest Classifier is trained and saved; if they only have 1 label, that
    value is saved as the classifier.

    The second part uses the TEST set.
    The fit PCA and KMeans are used to transform the test set. The predicted
    cluster is also added to the feature set and used to isolate the samples in
    each cluster. For each cluster, the corresponding classifier is retrieved
    and used to predict a label. All the predicted clusters are then joined and
    the F1 score is calculated.

    x_tr: training features (numpy array)
    y_tr: training labels (numpy array)
    x_te: testing features (numpy array)
    y_te: testing labels (numpy array)
    comps: n_components for PCA (int)
    k: n_clusters for KMeans (int)
    '''

    # TRAINING SET -------------------------------------------------------------

    # Empty lists to save clusters, their imbalance and classifier
    clusters = []
    pct_ones = []
    clfs = []

    # PCA with recommended n_components
    pca = PCA(n_components=comps)
    x_tr_pca = pca.fit_transform(x_tr)

    # KMeans with recommended k
    km = KMeans(n_clusters=k)
    clust_pred_tr = km.fit_predict(x_tr_pca)

    # Add the predicted cluster as an extra feature and the ground truth
    x_tr_transf = np.hstack((x_tr_pca, clust_pred_tr.reshape(-1,1)))
    assert x_tr_transf.shape[0] == y_tr.shape[0] # sanity check
    x_tr_transf = np.hstack((x_tr_transf, y_tr.reshape(-1,1)))

    # ground truth is at x_tr_transf[:, -1]
    # cluster pred. is at x_tr_transf[:, -2]

    # Identify predicted clusters
    cluster_labels, samples_in_cluster = np.unique(x_tr_transf[:, -2], return_counts=True)

    for label in cluster_labels:

        # Separate samples by cluster
        l = int(label)
        c = x_tr_transf[np.where(x_tr_transf[:, -2] == l)]
        clusters.append(c)

        # Calculate the imbalance in the cluster (percentage of label = 1)
        vals, num = np.unique(c[:, -1], return_counts=True)
        if len(vals) > 1: pct = (vals[0]*num[0] + vals[1]*num[1]) / (num[0] + num[1])
        else: pct = (vals[0]*num[0]) / num[0]
        pct_ones.append(pct)

        # If samples in cluster have both labels (0 and 1), train a RF
        if len(vals) > 1:
            rf = RandomForestClassifier(n_estimators=15, max_depth=3)
            c_x = c[:, :-2] # features (without predicted cluster since it's constant)
            c_y = c[:, -1] # labels
            rf.fit(c_x, c_y)
            clfs.append(rf)

        # If all samples in cluster have 1 label (0 or 1), save the value
        else:
            lbl = vals[0]
            clfs.append(lbl)

    # TEST SET -----------------------------------------------------------------

    # Empty lists to save clusters
    te_clusters = []

    # PCA
    x_te_pca = pca.transform(x_te)

    # KMeans
    clust_pred_te = km.predict(x_te_pca)

    # Add the predicted cluster as an extra feature and the ground truth
    x_te_transf = np.hstack((x_te_pca, clust_pred_te.reshape(-1,1)))
    assert x_te_transf.shape[0] == y_te.shape[0]
    x_te_transf = np.hstack((x_te_transf, y_te.reshape(-1,1)))

    # ground truth is at x_te_transf[:, -1]
    # cluster pred. is at x_te_transf[:, -2]

    # Identify predicted clusters
    te_cluster_labels, te_samples_in_cluster = np.unique(x_te_transf[:, -2], return_counts=True)

    for te_label in te_cluster_labels:

        # Separate samples by cluster
        l = int(te_label)
        c = x_te_transf[np.where(x_te_transf[:,-2] == l)]

        # Retrieve classifier
        clf = clfs[l]

        # If classifier is a trained RF, predict sample labels
        if 'sklearn' in str(type(clf)):
            c_x = c[:, :-2] # features (without predicted cluster since it's constant)
            y_pred = clf.predict(c_x)

        # If classifier is a pure label, assign it to samples
        else:
            y_pred = np.full((c.shape[0],), clf) # array with the value of label

        # Append label prediction and save cluster
        c = np.hstack((c, y_pred.reshape(-1,1)))
        te_clusters.append(c)

    # Join all test clusters
    count = 1
    for cluster in te_clusters:
        if count > 1:
            full_set = np.vstack((full_set, cluster))
        else:
            full_set = cluster
        count += 1

    # Calculate F1 score
    f1_score = f1(full_set[:, -2], full_set[:, -1]) # (ground truth, y_pred)

    return f1_score, pct_ones



def algorithm_w_folds(X, y, comps, k):
    '''
    This routine creates 10 stratified folds and foe each iteration, the
    algorithm is called and the F1 score is stored. Finally, the average and
    standard deviation of the scores is displayed, and the array of 10 scores
    is returned.

    X: features (numpy array)
    y: labels (numpy array)
    comps: n_components for PCA (int)
    k: n_clusters for KMeans (int)
    '''

    # Empty list to store each fold's score
    f1_scores = []

    # Initialize
    skf = StratifiedKFold(n_splits=10)

    it = 1

    # In each of the 10 iterations
    for tr_ids, te_ids in skf.split(X, y):

        # Split dataset in training and testing
        x_tr, y_tr = X[tr_ids, :], y[tr_ids]
        x_te, y_te = X[te_ids, :], y[te_ids]

        # Display progress
        print(f"Computing... [{it}/10]")

        # Perform the algorithm
        score, blob = cluster_and_balance(x_tr, y_tr, x_te, y_te, comps, k)

        # Save the score
        f1_scores.append(score)

        it += 1

    # Sanity check
    assert len(f1_scores) == 10

    # Display results
    f1_scores = np.array(f1_scores)
    print("\nAverage F1 score: %0.4f +/- %0.4f" % (f1_scores.mean(), f1_scores.std()))

    return f1_scores



def plot_cluster_imbalance(pcts, set_abbrv=None):
    '''
    This routine generates a histogram to visualize the label imbalances
    within clusters.

    pcts: list of the imbalances for each cluster (list of floats)
    set_abbrv: abbreviation of set's balance (string)
    '''    

    # Defining the set full name
    if set_abbrv is not None:
        if set_abbrv == 'bal': set_name = 'balanced'
        elif set_abbrv == 'li': set_name = 'low imbalance'
        elif set_abbrv == 'mi': set_name = 'medium imbalance'
        elif set_abbrv == 'hi': set_name = 'high imbalance'

    # Creating plot
    sns.histplot(pcts, bins=10, binrange=(0.0, 1.0))
    plt.title(f'Imbalance in clusters for {set_name} dataset')
    plt.xlabel('Percentage of label = 1')
    plt.ylabel('Clusters')

    # Save and show
    if set_abbrv is not None:
        plt.savefig('cluster_imbalance_' + set_abbrv + '.pdf', dpi=1200, bbox_inches='tight')
    plt.show()