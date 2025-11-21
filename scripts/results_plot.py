import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast 

df = pd.read_csv("experiment_results.csv")

df['ANN_Nodes'] = df['ANN_Nodes'].apply(lambda x: str(x)) 

def parse_pso_param(s, key):
    return int(ast.literal_eval(s)[key])

df['swarm_size'] = df['PSO_Params'].apply(lambda x: parse_pso_param(x, 'swarm_size'))
df['max_iter'] = df['PSO_Params'].apply(lambda x: parse_pso_param(x, 'max_iter'))

# Bar plot: MAE vs ANN architecture
agg_arch = df.groupby('ANN_Nodes')['MAE_avg'].mean().reset_index()
plt.figure(figsize=(8,5))
plt.bar(agg_arch['ANN_Nodes'], agg_arch['MAE_avg'], color='orange', alpha=0.7, width=0.5)
plt.ylabel("Mean MAE")
plt.title("MAE vs ANN Architecture")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Heatmap: MAE across ANN architecture vs swarm size
pivot = df.pivot(index='ANN_Nodes', columns='swarm_size', values='MAE_avg')
plt.figure(figsize=(8,5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("MAE Heatmap by ANN Architecture and PSO Swarm Size")
plt.ylabel("ANN Nodes")
plt.xlabel("Swarm Size")
plt.show()