import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Feature Selection

Criteria
- Clear separation between houses (different colors form distinct clusters)
- Low correlation between features (to avoid redundant information)

We already know that these doesn't meet criteria, so removed them before creating the pair plot matrix.
- Care of Magical Creatures
- Arithmancy
- Defense Against the Dark Arts do not meet 

Next, identify and remove houses do not form distinct clusters as well
- Potions
- Charms
- Flying

Remaining features (final answer):
- Astronomy
- Herbology
- Divination
- Muggle Studies
- Ancient Runes
- History of Magic
- Transfiguration

"""

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
edge_colors = ['#073354', '#a44807', '#0f490f', '#7a1414']

data = pd.read_csv("data/dataset_train.csv")
houses = data["Hogwarts House"].unique()

courses = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 
            'Ancient Runes', 'History of Magic', 'Transfiguration', 'Charms', 'Flying', 'Potions']

#--------------------------------------------------------------------------
# Plot figure

n = len(courses)
fig, axes = plt.subplots(n + 1, n, figsize=(30, 30))

# Scatter plot
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        
        if (i == j): 
            fig.delaxes(ax)
            continue

        for id, h in enumerate(houses):
            x = np.array(data[data["Hogwarts House"] == h][courses[j]])
            y = np.array(data[data["Hogwarts House"] == h][courses[i]])
        
            ax.scatter(x, y, s=10, c=colors[id], edgecolor="white", linewidths=0.1, label=h)
        
        ax.set(xlabel=courses[j], ylabel=courses[i], xticks=[], yticks=[])

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
       # Add quadrant lines at midpoints
        xmid = sum(ax.get_xlim()) / 2
        ymid = sum(ax.get_ylim()) / 2
        
        ax.axhline(y=ymid, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=xmid, color='black', linestyle='--', linewidth=0.8, alpha=0.5)


# Histogram plot
for i, course in enumerate(courses):
    ax = axes[n, i]

    plot = []
    for h in houses:
        plot.append(data[data["Hogwarts House"] == h][course])

    ax.hist(plot,
        bins=50, histtype='stepfilled', linewidth=1,
        color = colors, edgecolor=edge_colors, label = houses, alpha=0.6)

    ax.set(xlabel=course, xticks=[], yticks=[])

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)


# Touch-up
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91), fontsize=16, ncol=4, markerscale=4)
fig.suptitle('Pair Plot: Feature Selection for Logistic Regression', fontsize=32, fontweight='bold', y=0.93)


axes[1,7].patch.set_facecolor('lightgray')
axes[2,7].patch.set_facecolor('lightgray')
axes[3,7].patch.set_facecolor('lightgray')
axes[4,7].patch.set_facecolor('lightgray')
axes[5,7].patch.set_facecolor('lightgray')
axes[6,7].patch.set_facecolor('lightgray')
axes[8,7].patch.set_facecolor('lightgray')
axes[9,7].patch.set_facecolor('lightgray')

axes[0,8].patch.set_facecolor('lightgray')
axes[2,8].patch.set_facecolor('lightgray')
axes[7,8].patch.set_facecolor('lightgray')
axes[9,8].patch.set_facecolor('lightgray')

axes[8,9].patch.set_facecolor('lightgray')
axes[7,9].patch.set_facecolor('lightgray')
axes[5,9].patch.set_facecolor('lightgray')
axes[4,9].patch.set_facecolor('lightgray')
axes[3,9].patch.set_facecolor('lightgray')

fig.savefig('figure/pair_plot.png', dpi=300, bbox_inches='tight')




