import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Question: What are the two Hogwarts courses that are similar 

Find courses pair with:
- Strong linear correlation
- Similar distributions or ranges
- One feature predicts the other well

Answer: Astronomy & Defense Against the Dark Arts

"""

# Define colors once
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Prepare data
data = pd.read_csv("data/dataset_train.csv")
courses = data.columns[6:]
houses = data["Hogwarts House"].unique()



#--------------------------------------------------------------------------
# Plot figure

fig, axes = plt.subplots(13, 12, figsize=(30, 30))

for i in range(13):
    for j in range(i + 1, 13):
        ax = axes[12 - i, 12 - j]
        
        for id, h in enumerate(houses):
            x = np.array(data[data["Hogwarts House"] == h][courses[j]])
            y = np.array(data[data["Hogwarts House"] == h][courses[i]])
        
            ax.scatter(x, y, s=10, c=colors[id], edgecolor="white", linewidths=0.1, label=h)
        
        ax.set(xlabel=courses[j], ylabel=courses[i], xticks=[], yticks=[])

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
for i in range(13):
    for j in range(i, 12):
        fig.delaxes(axes[i, j])

for i, j in [(5, 0), (11, 9)]:
    axes[i, j].patch.set_facecolor('lightgray')

handles, labels = axes[12, 10].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.8, 0.7), fontsize=32, ncol=1, markerscale=7)
fig.suptitle('Hogwarts Course (Features) Pair Matrix', fontsize=32, fontweight='bold', y=0.81)
fig.savefig('figure/scatter.png', dpi=300, bbox_inches='tight')