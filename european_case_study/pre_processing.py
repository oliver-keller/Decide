#%%
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import copy
import seaborn

df = pd.read_csv("./raw_data/paper_metrics.csv")
folder = "results/"
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')
seaborn.set(style = 'whitegrid')  
save_results = True


categories = ['average_national_import',  'storage_discharge_capacity', 'transport_electrification', \
               'heat_electrification',  'biofuel_utilisation']

df = df.loc[df["metric"].isin(categories)]

n_spores = max(df["spore"]) + 1 
n_metrics = len(categories)

X = np.ndarray((n_spores,n_metrics))
X_header = categories

max_metrics = np.ndarray((1,n_metrics))
for i in range(n_metrics):
    max_metrics[0,i] = df.loc[df["metric"] == X_header[i]]["paper_metrics"].max()

max_df = pd.DataFrame(max_metrics, columns = X_header,index=["max"])



for i_spore in range(0,n_spores):
    for j_metrics in range(0,n_metrics):
        X[i_spore,j_metrics] = df.loc[(df["spore"]==i_spore) & (df["metric"]==categories[j_metrics])]["paper_metrics"]/  \
            float(max_df[categories[j_metrics]])


df_input_normalized = pd.DataFrame(X, columns = X_header)

# %%
if save_results:
    df_input_normalized.to_csv(folder+"df_input_normalized.csv",sep=";")
# %%
