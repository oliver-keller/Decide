#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text
import copy

from plot import plot_tree, plot_and_save_spyder_plots

#%%
def train_tree_and_reorder(df_input_with_cluster,short_names,figure_folder,colors,plot_all_spyders,tree_size=(15,10)):
    n_cl = df_input_with_cluster["initial cluster"].max() + 1
    categories = df_input_with_cluster.columns.to_list() 
    categories.remove("initial cluster")
    n_metrics = len(categories)


    target_tree = df_input_with_cluster["initial cluster"]



    n_leafnodes = [i for i in range(3,20)]
    score = [0]*len(n_leafnodes)
    i_score = 0

    for n_leafs in n_leafnodes:
        interpretation_tree = tree.DecisionTreeClassifier(max_leaf_nodes=n_leafs,criterion="entropy")
        interpretation_tree.fit(df_input_with_cluster[categories],target_tree)
        score[i_score] = interpretation_tree.score(df_input_with_cluster[categories],target_tree)
        i_score += 1


    plt.figure()
    plt.plot(n_leafnodes,score)
    plt.plot([n_cl,n_cl],[0.92,1],"--",color="k")
    plt.xlabel("number of tree leaves")
    plt.ylabel("score")


    interpretation_tree = tree.DecisionTreeClassifier(max_leaf_nodes=n_cl,criterion="entropy")
    interpretation_tree.fit(df_input_with_cluster[categories],target_tree)
    plt.figure()
    tree.plot_tree(interpretation_tree,feature_names=short_names,fontsize=8)
    #plt.show()
    plt.tight_layout()



    plt.savefig(figure_folder+"/tree.svg")
    plt.savefig(figure_folder+"/tree.pdf")

    df_input_with_cluster["cluster_final"] = interpretation_tree.predict(df_input_with_cluster[categories])

    nodes, choices = plot_tree(interpretation_tree,categories,short_names,df_input_with_cluster,colors,size=tree_size)
    plt.savefig(figure_folder+"/nice_tree.svg")
    plt.savefig(figure_folder+"/nice_tree.pdf")

    if plot_all_spyders:
        plot_and_save_spyder_plots(interpretation_tree,categories, \
                                df_input_with_cluster,short_names,figure_folder,colors)
    


    return df_input_with_cluster, nodes, choices
#%%
# df_input_with_cluster.to_csv(folder+"df_input_with_cluster_tree.csv", sep = ";")
# %%
