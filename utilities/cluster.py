from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.tree import export_text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

def calc_inertia(X,ai,labels):
    """
    Calculate the inertia of a clustering model.

    Parameters:
    X (numpy.ndarray): The input data matrix.
    ai (sklearn.cluster.KMeans): The trained clustering model.
    labels (numpy.ndarray): The predicted labels for each data point.

    Returns:
    float: The inertia value.

    """
    inertia = 0
    for i in range(len(labels)):
        # Calculate the squared Euclidean distance between each data point and its cluster center
        inertia_i = sum((X[i,j] - ai.cluster_centers_[labels[i]][j])**2 for j in range(X.shape[1]) )   
        # Sum up the inertia for all data points
        inertia += inertia_i

    # Return the total inertia value
    return inertia


def cluster(df_input, short_names,n_cl,figure_folder):
    """
    Clusters the input data using KMeans algorithm and returns a DataFrame with the initial cluster labels.

    Parameters:
    df_input (DataFrame): The input data to be clustered.
    short_names (list): A list of short names for the features in the input data.
    n_cl (int): The number of clusters to create.
    figure_folder (str): The folder path to save the generated plot.

    Returns:
    DataFrame: A copy of the input data DataFrame with an additional column for the initial cluster labels.
    """
        
    input_numpy = df_input.to_numpy()

    my_inertia = []
    score = []
    my_inertia_tree = []
    n_test = 25

    print("\t", end="")
    for x in short_names:
        print(x + "\t", end="")
    print()

    for i in range(2,n_test):
        #print(i)
        #ai = KMedoids(n_clusters=i)
        ai = KMeans(n_clusters=i, random_state=0, n_init=10)

        ai.fit(input_numpy)
        # wccs.append(ai.inertia_)

        my_inertia.append(calc_inertia(input_numpy,ai,labels=ai.labels_))

        target_tree = pd.DataFrame(data=ai.labels_)
        interpretation_tree = tree.DecisionTreeClassifier(max_leaf_nodes=i,criterion="entropy")
        interpretation_tree.fit(df_input,target_tree)
        score.append(interpretation_tree.score(df_input,target_tree))
        labels_tree = interpretation_tree.predict(df_input)
        my_inertia_tree.append(calc_inertia(input_numpy,ai,labels=labels_tree ))    


        tree_as_text = export_text(interpretation_tree,feature_names=short_names)
        print(str(i)+ "& \t", end="")
        for cat in short_names:
            print(str(int(tree_as_text.count(cat)/2))+ "& \t", end="")
        print("\\")


    plt.figure()
    # plt.plot(range(2,n_test),wccs)
    plt.plot(range(2,n_test),my_inertia,label="KMeans")
    plt.plot(range(2,n_test),my_inertia_tree,label="Re-ordered")
    plt.legend()
    plt.plot([n_cl,n_cl],[0,my_inertia_tree[0]],color="k")
    plt.xlabel("Number of clusters")
    plt.ylabel("Distance measure $d$")
    plt.savefig(figure_folder+"/elbow.svg")
    plt.savefig(figure_folder+"/elbow.pdf")

    ai = KMeans(n_clusters=n_cl, random_state=0, n_init=10)
    ai.fit(input_numpy)
    df_input_with_cluster = copy.deepcopy(df_input)
    df_input_with_cluster["initial cluster"] = [0]*input_numpy.shape[0]
    df_input_with_cluster["initial cluster"] = ai.labels_

    return df_input_with_cluster