#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys
import copy
sys.path.append('../utilities')
from produce_interpretable_tree import produce_interpretable_tree
#%%
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')
seaborn.set(style = 'whitegrid')  

n_cl = 4
short_names = ["imp","st","te","he","bu"]

folder = "results/"
save_results = True

df_input = pd.read_csv(folder+"df_input_normalized.csv",sep=";")
df_input = df_input.set_index("Unnamed: 0")


#%%
df_input_with_final_cluster, nodes, choices, decision_space = produce_interpretable_tree(df_input, short_names,n_cl)


#%%


if save_results:
    df_input_with_final_cluster.to_csv(folder+"df_input_with_final_cluster.csv", sep = ";")
#%%