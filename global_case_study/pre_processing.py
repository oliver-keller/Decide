#%%
import pandas as pd
import numpy as np
import math 

#folder ="20230703_results/"
folder = "results/"

save_results = True

df_mca_results_1 = pd.read_csv("./raw_data/MCA_Results_Global_1.csv",delimiter =";")
df_mca_results_2 = pd.read_csv("./raw_data/MCA_Results_Global_2.csv",delimiter =";")

df_mca_results_1 = df_mca_results_1.loc[df_mca_results_1["Scenario"]=="2C_SSP2"]
df_mca_results_2 = df_mca_results_2.loc[df_mca_results_2["Scenario"]=="2C_SSP2"]


df_mca_results_1['Global Electricity Supply from Wind and Solar (EJ/yr.)'] = df_mca_results_1['Global Electricity Supply from Solar (EJ/yr.)'] + df_mca_results_1['Global Electricity Supply from Wind (EJ/yr.)']

categories = df_mca_results_1.columns[3:].to_list() + df_mca_results_2.columns[3:].to_list()
ranges = pd.DataFrame({"category":categories})
ranges["file"] = [1]*len(df_mca_results_1.columns[3:].to_list()) + [2]*len(df_mca_results_2.columns[3:].to_list())

min_2050 = [0]*len(categories)
max_2050 = [0]*len(categories)
min_2100 = [0]*len(categories)
max_2100 = [0]*len(categories)

for i in range(len(categories)):
    if float(ranges.loc[ranges["category"] == categories[i]]["file"]) == 1:
        df = df_mca_results_1
    else:
        df = df_mca_results_2
    
    min_2050[i] = df.loc[df["Years"] == 2050][categories[i]].min()
    max_2050[i] = df.loc[df["Years"] == 2050][categories[i]].max()
    min_2100[i] = df.loc[df["Years"] == 2100][categories[i]].min()
    max_2100[i] = df.loc[df["Years"] == 2100][categories[i]].max()

ranges["min 2050"] = min_2050
ranges["max_2050"] = max_2050
ranges["min_2100"] = min_2100
ranges["max_2100"] = max_2100

if save_results:
    ranges.to_csv(folder+"ranges.csv",sep=";")



clustering_categories_dict = { \
    #'Temperature change (oC)': 2050, \
 #'Marginal CO2 cost (USD/t)': 2100, \
 #'Global Electricity Supply from Wind and Solar (EJ/yr.)': 2100, \
 "Global Primary Energy Consumption of Renewable energy (EJ/yr.)":2100 , \
#  'Global Electricity Supply from Solar (EJ/yr.)': 2100, \
#  'Global Electricity Supply from Wind (EJ/yr.)': 2100, \
 'Global Primary Energy Consumption of Fossil Fuels (EJ/yr.)': 2100, \
 'Share of electricity consumption in industry worldwide': 2100, \
 'Share of electric residential heating in global energy service demand for heating': 2100, \
 #'Total  Global hydrogen supply (PJ/yr.)': 2100, \
 'Share of alternative fuels consumption in transport worldwide': 2100}
clustering_categories = [key + " in " + str(clustering_categories_dict[key]) for key in clustering_categories_dict.keys()]
headers = [df_mca_results_1.columns[2]] + clustering_categories


#data cleaning 
clean_monte_carlo_runs = []
for i in range(1000):
    clean = True
    for category in clustering_categories_dict.keys():
        if float(ranges.loc[ranges["category"] == category]["file"]) == 1:
            df = df_mca_results_1
        else:
            df = df_mca_results_2

        try:
            
            if math.isnan(df.loc[(df["Years"] == clustering_categories_dict[category]) & (df["Monte Carlo Experiment"] == i) ][category]):
                clean = False
        except:
            clean = False

    if clean:
        clean_monte_carlo_runs.append(i)
#print(str(len(clean_monte_carlo_runs)) + " clean Monte Carlo runs to consider")

# share_industry_2100 = []
# for i_cmcr in clean_monte_carlo_runs:
#     E_total = 1000 * float(df_mca_results_1['Global Total Primary Energy Consumption (EJ/yr.)']\
#                            .loc[(df_mca_results_1["Monte Carlo Experiment"] == i_cmcr) & \
#                                 (df_mca_results_1["Years"] == 2100)])
#     E_industry = float(df_mca_results_2['Global total final energy consumption in industry (PJ/yr.)']\
#                            .loc[(df_mca_results_2["Monte Carlo Experiment"] == i_cmcr) & \
#                                 (df_mca_results_2["Years"] == 2100)])
#     share_industry_2100 += [E_industry/E_total] 
# print("minimum share industry : " + str(float(min(share_industry_2100))))
# print("maximum share industry : " + str(float(max(share_industry_2100))))

key_metrics = pd.DataFrame({headers[0]:clean_monte_carlo_runs})
key_metrics.set_index(headers[0])


n_metrics = len(clustering_categories_dict)
n_clean_MC_runs = len(clean_monte_carlo_runs)
X = np.ndarray((n_clean_MC_runs,n_metrics))
max_metrics = np.ndarray((1,n_metrics))

for i in range(1,len(headers)):
    category = [k for k in clustering_categories_dict.keys()][i-1]
    #print(category)

    if float(ranges.loc[ranges["category"] == category]["file"]) == 1:
        df = df_mca_results_1
        #print(1)
    else:
        df = df_mca_results_2
        #print(2)

    # print(df.loc[df["Years"] == clustering_categories_dict[category]][category])

    key_metrics[headers[i]] = df.loc[(df["Years"] == clustering_categories_dict[category]) & (df["Monte Carlo Experiment"].isin(clean_monte_carlo_runs))][category].to_list()

    max_metrics[0,i-1] = df.loc[(df["Years"] == clustering_categories_dict[category]) & (df["Monte Carlo Experiment"].isin(clean_monte_carlo_runs))][category].max()

max_df = pd.DataFrame(max_metrics, columns = headers[1:],index=["max"])

if save_results:
    key_metrics.to_csv(folder+"key_metrics.csv",sep=";")

#we scale all metrics
idx_X = 0
for i_MC_run in clean_monte_carlo_runs:
    for j_metrics in range(0,n_metrics):
        
        X[idx_X,j_metrics] = key_metrics.loc[(key_metrics["Monte Carlo Experiment"]==i_MC_run)][headers[j_metrics+1]]/  \
            float(max_df[headers[j_metrics+1]])
    idx_X += 1
        # use long categorie names !!

df_input_normalized = pd.DataFrame(X, columns = headers[1:],index=clean_monte_carlo_runs)
if save_results:
    df_input_normalized.to_csv(folder+"df_input_normalized.csv",sep=";")

#%%