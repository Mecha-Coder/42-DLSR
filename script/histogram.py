import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Question: Which Hogwarts course has a homogeneous score distribution between all four houses?

Find courses where:
- All houses have similar average scores
- All houses have similar score ranges (min, max)
- All houses have similar spread
- distribution shapes look similar across houses

Keyword:
- Homogeneous = uniform, similar, consistent
- Score distribution = scores are spread out

Answer: Care of Magical Creatures & Arithmancy

"""

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
edge_colors = ['#073354', '#a44807', '#0f490f', '#7a1414']

data = pd.read_csv("data/dataset_train.csv")
courses = data.columns[6:]
houses = data["Hogwarts House"].unique()

#--------------------------------------------------------------------------
# Plot figure

fig, axes = plt.subplots(3, 5, figsize=(15, 10))
axes = axes.flatten() 

for i, course in enumerate(courses):
    ax = axes[i]
    
    plot = []
    for h in houses:
        plot.append(data[data["Hogwarts House"] == h][course])
    
    ax.hist(plot, 
            bins=50, histtype='stepfilled', linewidth=1,
            color=colors, edgecolor=edge_colors, label=houses, alpha=0.6)
    
    ax.set(xlabel=course, xticks=[], yticks=[])

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
   

for i in [13, 14]:
    fig.delaxes(axes[i])

for i in [0, 10]:
    axes[i].patch.set_facecolor('lightgray')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.85, 0.25), fontsize=12, ncol=2)
fig.suptitle('Hogwarts Course Distribution by House', fontsize=16, fontweight='bold', y=0.91)
fig.savefig('figure/histogram.png', dpi=300, bbox_inches='tight')