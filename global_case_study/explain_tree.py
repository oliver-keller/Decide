#%%
from sklearn.tree import export_text
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn
import copy

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
seaborn.set(style = 'whitegrid')  

#%%
folder = "results/"
df_input_with_final_cluster = pd.read_csv(folder + "df_input_with_final_cluster.csv",delimiter=";")
df_input_with_final_cluster = df_input_with_final_cluster.set_index("Unnamed: 0")

n_cl = df_input_with_final_cluster["cluster_final"].max() + 1
categories = df_input_with_final_cluster.columns.to_list() 
categories.remove("cluster_final")
categories.remove("initial cluster")
n_metrics = len(categories)


MC_inputs = pd.read_csv("./raw_data/MCA_inputs.csv",delimiter=";")
clean_runs = df_input_with_final_cluster.index
MC_inputs = MC_inputs.loc[MC_inputs["Monte Carlo Experiment"].isin(clean_runs)]
MC_inputs.set_index("Monte Carlo Experiment")
MC_inputs["cluster_final"] = df_input_with_final_cluster["cluster_final"].to_list()
inputs = [ 'GDP', 'Population', \
       'Other Energy Service Demand Drivers', 'Social discount rate', \
       'Elasticity of energy service demand to its own driver', \
       'Elasticity of energy service demand to its own price', \
       'CO2 Storage Potential', 'Wind Potential','Solar Potential', \
       'Biomass Potential', 'Oil & Gas Potential', 'Solar PV  Investment Cost', \
       'Wind Investment Cost', 'Bioenergy with CCS Specific Investment Cost', \
       'Other technologies costs', 'Forcing of non-energy emissions', \
       'Land Use, Land Use Change and Forestry Sinks', \
       'Climate Sensitivity (in oC)']
#%%
target_tree = df_input_with_final_cluster["cluster_final"]
n_leafnodes = [i for i in range(2,30)]
coverage = [0]*len(n_leafnodes)
cross_validation_coverage = [0]*len(n_leafnodes)
interpretability  = [0]*len(n_leafnodes)
n_variables = [0]*len(n_leafnodes)
N_cross_validation = 10
i_score = 0

k = 5  # Number of folds (you can change this as needed)
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for n_leafs in n_leafnodes:
    explanation_tree = tree.DecisionTreeClassifier(max_leaf_nodes=n_leafs,criterion="entropy")
    explanation_tree.fit(MC_inputs[inputs],target_tree)
    coverage[i_score] = explanation_tree.score(MC_inputs[inputs],target_tree)
    tree_as_text = export_text(explanation_tree,feature_names=inputs)
    n_variables[i_score] = sum([tree_as_text.__contains__(input_i) \
                                for input_i in inputs])
    interpretability[i_score] = 1/n_variables[i_score]
    coverage_cv_sum = 0
    for train_indices, test_indices in kf.split(clean_runs):
        MC_train = MC_inputs[inputs].loc[MC_inputs[inputs].index.isin([clean_runs[i]-1 for i in train_indices])]
        MC_test = MC_inputs[inputs].loc[MC_inputs[inputs].index.isin([clean_runs[i]-1 for i in test_indices])]
        target_tree_train = target_tree.loc[target_tree.index.isin([clean_runs[i] for i in train_indices])]
        target_tree_test = target_tree.loc[target_tree.index.isin([clean_runs[i] for i in test_indices])]
        explanation_tree.fit(MC_train,target_tree_train)
        coverage_cv_sum += explanation_tree.score(MC_test,target_tree_test)
    cross_validation_coverage[i_score] = coverage_cv_sum/k
    i_score += 1


#%%
plt.figure()
plt.plot(n_leafnodes,interpretability,marker="^",label="interpretability",color="gray")
plt.plot(n_leafnodes,interpretability,color="gray") 
plt.plot(n_leafnodes,coverage,"x",label="coverage",color="k")
plt.plot(n_leafnodes,coverage,color="k")  
plt.plot(n_leafnodes,cross_validation_coverage,marker="o",label="cross-validation coverage",color="blue")
plt.plot(n_leafnodes,cross_validation_coverage,color="blue")   
plt.xlabel("number of tree leaves")
plt.ylabel("score")
plt.legend()
#%%
plt.savefig("figures/sensitivity_n_leaves.svg")
plt.savefig("figures/sensitivity_n_leaves.pdf")
#%%
MC_inputs_normalized = copy.deepcopy(MC_inputs)
for cat in inputs:
    MC_inputs_normalized[cat] = MC_inputs_normalized[cat] - MC_inputs_normalized[cat].min()
    MC_inputs_normalized[cat] = MC_inputs_normalized[cat]/MC_inputs_normalized[cat].max()



colors=["g","r","b","magenta"]
plt.figure()
for i in clean_runs:
    plt.plot(MC_inputs_normalized['Climate Sensitivity (in oC)'][i-1], \
             MC_inputs_normalized['Elasticity of energy service demand to its own driver'][i-1],"o", \
                color=colors[MC_inputs_normalized["cluster_final"][i-1]])
plt.xlabel('Normalized climate sensitivity')
plt.ylabel('Normalized demand elasticity')
plt.savefig("figures/explanation.svg")
plt.savefig("figures/explanation.pdf")
# %%
