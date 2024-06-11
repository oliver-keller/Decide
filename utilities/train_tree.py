#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text
import copy

from utilities.plot import plot_tree, plot_and_save_spyder_plots

#%%
def train_tree_and_reorder(df_input_with_cluster,short_names,figure_folder,colors,plot_all_spyders,tree_size=(15,10), absolute_values=False, print_info=True):
    """
    Trains a decision tree classifier using the input data and returns the reordered dataframe, nodes, and choices.
    Reordering means that the cluster labels are reordered based on the decision tree classifier.

    Parameters:
        df_input_with_cluster (DataFrame): The input dataframe with cluster information.
        short_names (list): The list of short names for the features.
        figure_folder (str): The folder path to save the generated figures.
        colors (list): The list of colors for plotting.
        plot_all_spyders (bool): Flag indicating whether to plot all spyder plots.
        tree_size (tuple, optional): The size of the decision tree plot. Defaults to (15, 10).
        absolute_values (bool, optional): Flag indicating whether to use absolute values for plotting. Defaults to False.
        print_info (bool, optional): Flag indicating whether to print additional information. Defaults to True.

    Returns:
        tuple: A tuple containing the reordered dataframe, nodes, and choices.
    """

    n_cl = df_input_with_cluster["initial cluster"].max() + 1
    categories = df_input_with_cluster.columns.to_list() 
    categories.remove("initial cluster")
    n_metrics = len(categories)

    target_tree = df_input_with_cluster["initial cluster"]

    if print_info:
        n_leafnodes = list(range(3,20))
        score = []

        for n_leafs in n_leafnodes:
            interpretation_tree = tree.DecisionTreeClassifier(max_leaf_nodes=n_leafs,criterion="entropy", random_state=0)
            interpretation_tree.fit(df_input_with_cluster[categories],target_tree)
            score.append(interpretation_tree.score(df_input_with_cluster[categories],target_tree))

        plt.figure()
        plt.plot(n_leafnodes,score)
        plt.plot([n_cl,n_cl],[0.92,1],"--",color="k")
        plt.xlabel("number of tree leaves")
        plt.ylabel("score")


    interpretation_tree = tree.DecisionTreeClassifier(max_leaf_nodes=n_cl,criterion="entropy", random_state=0)
    interpretation_tree.fit(df_input_with_cluster[categories],target_tree)
    plt.figure()
    tree.plot_tree(interpretation_tree,feature_names=short_names,fontsize=8)
    plt.tight_layout()



    plt.savefig(figure_folder+"/tree.svg")
    plt.savefig(figure_folder+"/tree.pdf")

    df_input_with_cluster["cluster_final"] = interpretation_tree.predict(df_input_with_cluster[categories])

    nodes, choices = plot_tree(interpretation_tree,categories,short_names,df_input_with_cluster,colors,size=tree_size, absolute_values=absolute_values)
    plt.savefig(figure_folder+"/nice_tree.svg")
    plt.savefig(figure_folder+"/nice_tree.pdf")

    if plot_all_spyders:
        plot_and_save_spyder_plots(interpretation_tree,categories, df_input_with_cluster,short_names,figure_folder,colors, absolute_values=absolute_values)
    


    return df_input_with_cluster, nodes, choices


