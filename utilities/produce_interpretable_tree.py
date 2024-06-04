from utilities.cluster import cluster
from utilities.train_tree import train_tree_and_reorder
import copy
import numpy as np
import os
from numpy.linalg import eig


default_colors = ["b","c","k","g","m","r","y","tab:blue","tab:brown", "tab:orange", "tab:pink","tab:gree","tab:gray"]

def produce_interpretable_tree(df_input, short_names, n_cl, figure_folder=None, tree_size=(15,10), colors=default_colors,plot_all_spyders = True):

    if figure_folder is None:
        figure_folder = "figures"
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)

    df_input_with_initial_cluster = cluster(df_input, short_names, n_cl, figure_folder)

    df_input_with_final_cluster, nodes, choices = train_tree_and_reorder(df_input_with_initial_cluster, short_names, figure_folder, tree_size=tree_size, colors=colors, plot_all_spyders=plot_all_spyders)
    
    categories = df_input_with_final_cluster.columns.to_list() 
    categories.remove("initial cluster")
    categories.remove('cluster_final')
    n_metrics = len(categories)

    decision_space = {}
    decision_space["sum"] = copy.deepcopy(nodes)
    decision_space["volume"] = copy.deepcopy(nodes)
    decision_space["distance"] = copy.deepcopy(nodes)

    decision_space["effective_dimensionality"] = copy.deepcopy(nodes)
    #decision_space = 

    def multiplyList(myList):
    
        # Multiply elements one by one
        result = 1
        for x in myList:
            result = result * x
        return result
    

    ranges_initial = {}
    for cat in categories:
        ranges_initial[cat] = df_input_with_final_cluster[cat].max() - df_input_with_final_cluster[cat].min()

    for k in nodes.keys():
        for i in range(len(nodes[k])):
            if nodes[k][i] != []:
                # print(nodes[k][i])
                df_node = df_input_with_final_cluster.loc[df_input_with_final_cluster["cluster_final"].isin(nodes[k][i])]

                ranges_node = {}
                for cat in categories:
                    
                    ranges_node[cat] = df_node[cat].max() - df_node[cat].min()
                    
                #calculate effective dimensionality
                #lambdas ,v = eig(df_node[ categories].corr())
                lambdas ,v = eig(df_node[ categories].cov())
                # print(lambdas)
                sum_l = sum(lambdas)
                intermediate_term = [(l/sum_l)**(-l/sum_l) for l in lambdas]
                # print(intermediate_term)
                decision_space["effective_dimensionality"][k][i] = multiplyList(intermediate_term)

                # print(ranges_node)
                    
                decision_space["sum"][k][i] = sum(ranges_node[cat] for cat in categories) \
                    /sum(ranges_initial[cat] for cat in categories)
                
                decision_space["volume"][k][i] = multiplyList(ranges_node[cat] for cat in categories ) \
                    /multiplyList(ranges_initial[cat] for cat in categories)
                
                decision_space["distance"][k][i] = np.power(multiplyList(ranges_node[cat] for cat in categories ) \
                    /multiplyList(ranges_initial[cat] for cat in categories), \
                    1/n_metrics)

    return df_input_with_final_cluster , nodes, choices, decision_space