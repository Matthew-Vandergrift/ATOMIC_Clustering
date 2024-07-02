# This file contains the processes which combine multiple clustering solutions together.
from nonconvex_data import nonconvex_gen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import umap

# GEECO Settings for making poster plot
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

if __name__ == '__main__':
    np.random.seed(1)

    # Loading/Generating the dataset
    df = nonconvex_gen()
    # df, y, explan = generate2(1000, 100, 3) 

    # Defining indices to combine, this as mentioned in ATOMIC, is given by the user 
    indices_to_combine = [1,2,3]
    labels_all = [0 for _ in range(len(indices_to_combine))]

    # Loading the Labels
    for indx in indices_to_combine:
        # Loading Labels 1
        text_file = open("myarray"+str(indx)+".txt", "r")
        lines = text_file.readlines()
        labels_all[indx-1] = [int(float(line.rstrip('\n'))) for line in lines]
        text_file.close()
    num_points = len(labels_all[0])
    print("Length is :", len(labels_all))

    # Combining the labels, we do this by simply stacking them atop each other, and when they overlap the 
    # higher index takes priority. 
    combined_labels = []
    for i in range(num_points):
        non_added = False
        for real in range(len(labels_all)):

            # This simply accounts for the fact that sometimes the cluster is labeled 0 
            # and not 1. 
            if sum(labels_all[real]) < 0.5 * len(labels_all[real]):
                non_clust = 0
            else:
                non_clust = 1

            if labels_all[real][i] != non_clust:
                combined_labels.append(real+1)
                non_added = True
                break
        
        if non_added == False:
            combined_labels.append(0)

    # Displaying Combined label solution 
    reduction = umap.UMAP(densmap=False, random_state=1) #Normal Seed is 1 
    embedded = reduction.fit_transform(df.to_numpy())
    fig, ax = plt.subplots()
    for g in np.unique(combined_labels):
        kl = np.where(combined_labels == g)
        ax.scatter(embedded[:, 0][kl], embedded[:, 1][kl], label=g)
    ax.legend()
    plt.annotate('Defined by $X_1$ and $X_2$',xy=(-7,-7),xytext=(-7,-9),
             arrowprops=dict(arrowstyle='->',lw=1.5),fontsize=15)
    plt.annotate('Defined by $X_3$ and $X_5$',xy=(15,-7),xytext=(9,-3),
             arrowprops=dict(arrowstyle='->',lw=1.5),fontsize=15)
    plt.annotate('Defined by $X_4$ and $X_6$',xy=(7,13),xytext=(11,12),
             arrowprops=dict(arrowstyle='->',lw=1.5), fontsize=15)
    fig.set_size_inches(12, 10)
    plt.xlabel("dense-Map Axis 1")
    plt.ylabel("dense-Map Axis 2")
    fig.savefig('atomic_example'+str(1)+'DENSE.svg', format='svg', dpi=1200)
    plt.show()