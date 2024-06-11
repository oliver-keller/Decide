from utilities.cluster import cluster
from utilities.train_tree import train_tree_and_reorder
import copy
import numpy as np
import os
from numpy.linalg import eig


default_colors = ["b","c","k","g","m","r","y","tab:blue","tab:brown", "tab:orange", "tab:pink","tab:gree","tab:gray"]

def produce_interpretable_tree(df_input, short_names, n_cl, figure_folder=None, tree_size=(15,10), colors=default_colors,plot_all_spyders = True, absolute_values=False, print_info=True):
    """
    Produces an interpretable tree based on the input data.

    Args:
        df_input (DataFrame): The input data frame.
        short_names (list): The list of short names for the input data columns.
        n_cl (int): The number of clusters.
        figure_folder (str, optional): The folder to save the figures. Defaults to None.
        tree_size (tuple, optional): The size of the tree figure. Defaults to (15, 10).
        colors (dict, optional): The colors for plotting. Defaults to default_colors.
        plot_all_spyders (bool, optional): Whether to plot all spyders. Defaults to True.
        absolute_values (bool, optional): Whether to use absolute values. Defaults to False.
        print_info (bool, optional): Whether to print information. Defaults to True.

    Returns:
        tuple: A tuple containing the final cluster data frame, nodes, choices, and decision space.
    """    
    if figure_folder is None:
        figure_folder = "figures"
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)

    df_input_with_initial_cluster = cluster(df_input, short_names, n_cl, figure_folder, print_info=print_info)
    # print(df_input_with_initial_cluster)

    df_input_with_final_cluster, nodes, choices = train_tree_and_reorder(df_input_with_initial_cluster, short_names, figure_folder, tree_size=tree_size, colors=colors, plot_all_spyders=plot_all_spyders, absolute_values=absolute_values, print_info=print_info)
    # print(df_input_with_final_cluster)
    
    categories = df_input_with_final_cluster.columns.to_list() 
    categories.remove("initial cluster")
    categories.remove('cluster_final')
    n_metrics = len(categories)

    decision_space = {}
    decision_space["sum"] = copy.deepcopy(nodes)
    decision_space["volume"] = copy.deepcopy(nodes)
    decision_space["distance"] = copy.deepcopy(nodes)
    decision_space["effective_dimensionality"] = copy.deepcopy(nodes)

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