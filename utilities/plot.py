import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import export_text
import seaborn
seaborn.set(style = 'whitegrid')

def _get_nodes_and_choics(tree_as_text,n_cl,X_header):
    node_list = tree_as_text.split("\n")[:-1]
    depth = max([node_list[i].count("|")  for i in range(len(node_list))])
    nodes = {}
    choices = {}
    splits = {}
    for i in range(depth):
        if i == 0:
            nodes[i] = [[i_cl for i_cl in range(n_cl)]]
        else:
            breakpoints = [idx  for idx in range(len(node_list)) if node_list[idx].startswith((i-1)*"|   "+"|---")]

            nodes[i] = []
            splits[i] = []
            idx_breakpoints = 0
            for j in range(2**(i-1)):
                if len(nodes[i-1][j]) > 1: 
                    split_string = node_list[breakpoints[idx_breakpoints]]
                    thres = float(split_string.split("<=")[1])
                    for index_category in range(len(X_header)):
                        if split_string.__contains__(X_header[index_category]):
                            break
                    split_low = [index_category,thres,"low"]
                    split_high = [index_category,thres,"high"]
                    splits[i].append(split_low)
                    splits[i].append(split_high)
                    for k in range(2):

                        if idx_breakpoints < len(breakpoints)-1:
                            classes = node_list[breakpoints[idx_breakpoints]:breakpoints[idx_breakpoints+1]]
                        else:
                            classes = node_list[breakpoints[idx_breakpoints]:]
                        classes = [int(c[-1]) for c in classes if c.__contains__("class")]
                        classes.sort()
                        nodes[i].append(classes)
                        
                        idx_breakpoints += 1

                elif len(nodes[i-1][j]) == 1:
                    nodes[i].append([])
                    nodes[i].append([])
                    splits[i].append([])
                    splits[i].append([])
                    idx_breakpoints += 1
                else:
                    nodes[i].append([])
                    nodes[i].append([])
                    splits[i].append([])
                    splits[i].append([])
            choices_i = [n for n in node_list if ( n.startswith((i-1)*"|   "+"|---") and (not n.__contains__("class") ))]
            for index_choice in range(len(choices_i)):
                choices_i[index_choice] = choices_i[index_choice].replace((i-1)*"|   "+"|---","")
            choices[i-1] = choices_i

    return nodes, choices, depth, splits

def plot_tree(interpretation_tree,categories,X_header,df_normalized_with_cluster, \
              colors,size=(10,10),fs_choices=6,fs_categories=6):
    #df_normalized_with_cluster["cluster_tree"] = interpretation_tree.predict(df_normalized_with_cluster[categories])
    n_cl = df_normalized_with_cluster["cluster_final"].max() + 1
    
    tree_as_text = export_text(interpretation_tree,feature_names=X_header)
    
    nodes, choices, depth, splits = _get_nodes_and_choics(tree_as_text,n_cl,X_header)


    delta_x = (1/(2**(depth-1)))*0.7
    delta_y = (1/(2*depth))*0.7
    clusters_for_polar_list = []
    splits_list = []
    positions = []
    midlepoints_y = [1- 1/(2*depth) - (1/depth)*k for k in range(depth)]
    for i in range(depth):
        n_plots = 2**i
        midlepoints_x = [1/(2**(i+1)) + 1/(2**(i))*k for k in range(n_plots)]
        for j in range(len(nodes[i])):
            if len(nodes[i][j]) > 0:
                clusters_for_polar_list += [nodes[i][j]]
                positions += [[midlepoints_x[j]-delta_x/2,midlepoints_y[i] -delta_y/2]]
                if i > 0:
                    splits_list += [splits[i][j]]
                else:
                    splits_list += [[]]

    
    fig = plt.figure(figsize = size)

    j=1
    for i in range(len(midlepoints_y)-1):

        y_min = midlepoints_y[i+1] + delta_y/2
        y_max = midlepoints_y[i] - delta_y/2

        for k in range(int(len(choices[i])/2)):
            x_min = positions[j][0] + delta_x/2
            x_max = positions[j+1][0] + delta_x/2

            j+=2

            ax = fig.add_axes([x_min, y_min, x_max-x_min, y_max-y_min])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.annotate('',xy=(0,0),xytext=(0,0.5),arrowprops=dict(arrowstyle="->",color="k"))
            ax.annotate('',xy=(1,0),xytext=(1,0.5),arrowprops=dict(arrowstyle="->",color="k"))
            ax.annotate('',xy=(0,0.5),xytext=(0.5,0.5),arrowprops=dict(arrowstyle="-",color="k"))
            ax.annotate('',xy=(1,0.5),xytext=(0.5,0.5),arrowprops=dict(arrowstyle="-",color="k"))

            ax.annotate('',xy=(0.5,0.6),xytext=(0.5,1),arrowprops=dict(arrowstyle="->",color="k"))
            ax.patch.set_alpha(0)

            ax = fig.add_axes([x_min, y_min, x_max-x_min, y_max-y_min])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #ax.arrow(0,1,0,0,width=0.01)
            ax.plot([0.5],[0.5],"D",markersize=15,color="k", mfc='white')
            ax.text(0.47, 0.505,choices[i][2*k],fontsize=fs_choices)
            ax.text(0.51, 0.505,choices[i][2*k+1],fontsize=fs_choices)
            #ax.scatter([0.5],[0.5],s=80, facecolors='blue', edgecolors='r')
            ax.patch.set_alpha(0)

    n_metrics = len(categories)


    max_metrics = [0]*n_metrics
    for i in range(n_metrics):
        metric = categories[i]
        max_metrics[i] = df_normalized_with_cluster[metric].max()


    for i_clusters in range(len(clusters_for_polar_list)): 
        

        clusters_for_polar = clusters_for_polar_list[i_clusters]
        split = splits_list[i_clusters]
        min_list = [0]*n_metrics
        max_list = [0]*n_metrics


        df_plot_clusters = df_normalized_with_cluster.loc[df_normalized_with_cluster["cluster_final"].isin(clusters_for_polar)]

        for i in range(n_metrics):
            metric = categories[i]
            max_list[i] = df_plot_clusters[metric].max()/max_metrics[i]
            min_list[i] = df_plot_clusters[metric].min()/max_metrics[i]


        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
        delta = (10/360)*2 * np.pi

        # The first value is repeated to close the chart.
        angles=np.concatenate((angles, [angles[0]]))

        ax = fig.add_axes([positions[i_clusters][0], positions[i_clusters][1], delta_x, delta_y],polar = True)

        plt.fill_between(angles, min_list + [min_list[0]], max_list + [max_list[0]],alpha=0.2,color="b",label="total range")
        for i_metric in range(len(min_list)):
            plt.polar([angles[i_metric], angles[i_metric]],[min_list[i_metric], max_list[i_metric]],color="b")

        plt.thetagrids(angles[:-1] * 180 / np.pi,X_header,fontsize=fs_categories)
        
        if len(split) > 0:
            i_metric = split[0]
            if split[2] == "low":
                plt.polar([angles[i_metric], angles[i_metric]], [split[1]/max_metrics[i_metric], min_list[i_metric]], \
                            linewidth=3,color=colors[i_metric])
            else:
                #print(i_metric)
                #print(max_metrics)
                plt.polar([angles[i_metric], angles[i_metric]], [split[1]/max_metrics[i_metric], max_list[i_metric]],  \
                        linewidth=3,color=colors[i_metric])
            delta = (10/360)*2 * np.pi *(0.374/split[1])
            plt.polar([angles[i_metric]-delta, angles[i_metric]+delta], [split[1]/max_metrics[i_metric], split[1]/max_metrics[i_metric]],  \
                        linewidth=3,color=colors[i_metric])

        n_fine = 100
        angles_fine = np.linspace(0, 2 * np.pi, n_fine, endpoint=False)
        angles_fine=np.concatenate((angles_fine, [angles_fine[0]]))
        plt.plot(angles_fine,[0]*len(angles_fine),color="k",linewidth=1)
        plt.plot(angles_fine,[1]*len(angles_fine),color="k",linewidth=1)
        ax.set_xticks(angles)
        ax.set_rmax(1.02)
        ax.set_rmin(-0.2)
        ax.set_yticklabels([0,"","","","",1],fontdict={"fontsize":8})
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
#        plt.show()
        
        plt.tight_layout()

    return nodes, choices

def plot_and_save_spyder_plots(interpretation_tree,categories, \
                               df_normalized_with_cluster,X_header,figure_folder,colors):
    


    tree_as_text = export_text(interpretation_tree,feature_names=X_header)
    n_cl = df_normalized_with_cluster["cluster_final"].max() + 1
    nodes, choices, depth, splits = _get_nodes_and_choics(tree_as_text,n_cl,X_header)


    n_metrics = len(categories)

    for nodes_key in nodes.keys():
        for idx in range(len(nodes[nodes_key])):
            clusters_for_polar = nodes[nodes_key][idx]

            if len(clusters_for_polar) > 0:

                min_list = [0]*n_metrics
                max_list = [0]*n_metrics

                df_plot_clusters = df_normalized_with_cluster.loc[df_normalized_with_cluster["cluster_final"].isin(clusters_for_polar)]

                for i in range(n_metrics):
                    metric = categories[i]
                    max_list[i] = df_plot_clusters[metric].max()
                    min_list[i] = df_plot_clusters[metric].min()
                angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
                delta = (10/360)*2 * np.pi

                # print(clusters_for_polar)
                # print(max_list)

                # The first value is repeated to close the chart.
                angles=np.concatenate((angles, [angles[0]]))

                fig, ax = plt.subplots(figsize=(2,2),subplot_kw={'projection': 'polar'})

                plt.fill_between(angles, min_list + [min_list[0]], max_list + [max_list[0]],alpha=0.2,color="b",label="total range")
                for i_metric in range(len(min_list)):
                    plt.plot([angles[i_metric], angles[i_metric]],[min_list[i_metric], max_list[i_metric]],color="b")

                # Representation of the spider graph
                plt.thetagrids(angles[:-1] * 180 / np.pi,[""]*n_metrics)
#                plt.show()
                
                #plt.title("".join([str(c) for c in clusters_for_polar ]) )

                if nodes_key > 0:
                    split = splits[nodes_key][idx]
                    i_metric = split[0]
                    if split[2] == "low":
                        plt.polar([angles[i_metric], angles[i_metric]], [split[1], min_list[i_metric]], \
                                    linewidth=3,color=colors[i_metric])
                    else:
                        plt.polar([angles[i_metric], angles[i_metric]], [split[1], max_list[i_metric]],  \
                                linewidth=3,color=colors[i_metric])
                    delta = (10/360)*2 * np.pi *(0.374/split[1])
                    plt.polar([angles[i_metric]-delta, angles[i_metric]+delta], [split[1], split[1]],  \
                                linewidth=3,color=colors[i_metric])

                n_fine = 100
                angles_fine = np.linspace(0, 2 * np.pi, n_fine, endpoint=False)
                angles_fine=np.concatenate((angles_fine, [angles_fine[0]]))
                plt.plot(angles_fine,[0]*len(angles_fine),color="k",linewidth=1)
                plt.plot(angles_fine,[1]*len(angles_fine),color="k",linewidth=1)
                ax.set_rmax(1.02)
                ax.set_rmin(-0.2)
                
                plt.tight_layout()
                
                ax.set_yticklabels([0,"","","","",1],fontdict={"fontsize":8})
                ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
                #plt.show()
                plt.savefig(figure_folder+"/"+"".join([str(c) for c in clusters_for_polar ]) + ".svg",  format="svg")
                plt.savefig(figure_folder + "/" + "".join([str(c) for c in clusters_for_polar]) + ".pdf")
