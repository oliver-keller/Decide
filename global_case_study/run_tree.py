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

n_cl = 3
short_names = [
 'ReE',
 'Fos',
 'SEI',
 'SEH',
 'SAF']
folder = "results/"
save_results = True

df_input = pd.read_csv(folder+"df_input_normalized.csv",sep=";")
df_input = df_input.set_index("Unnamed: 0")


#%%
colors=["b","c",(0/255,221/255,224/255),(214/255,42/255,42/255),"m"]
df_input_with_final_cluster , nodes, choices, decision_space = \
    produce_interpretable_tree(df_input, short_names,n_cl,colors=colors)


#%%
if save_results:
    df_input_with_final_cluster.to_csv(folder+"df_input_with_final_cluster.csv", sep = ";")

# %%
