import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# This function generates a p dimensional guassian clustering problem, where there is one large gaussian 
# cluster and the others up to k, are of a smaller size. This represents an ideal situtation for 
# the ATOMIC clustering algorithm. This dataset is used in the poster presented at GEECO 24. 
def generate2(n, p, k):
    # Building One Big Guassain Blob
    blobX, blobY, center = make_blobs(n, p, centers=1, cluster_std=1, return_centers=True, random_state=1)
    print(center[0])

    # Cluster explanations
    true_exps = []

    # For (k-1) clusters
    for c in range(0, k-1):

        # Generating number of sub dimensions and subpoints
        subN_dims = np.random.randint(2, 5, size=1)[0]
        subN_points = np.random.randint(int(n*0.05),int(n*0.20), size = 1)[0]
        
        # Generating sub cluster
        blobX_sub, blobY_sub = make_blobs(subN_points, subN_dims, centers=center+np.random.randint(2, 10), cluster_std=1)

        # Chosing the dimensions and points to use
        sub_dims = np.random.randint(0, p, size = subN_dims)
        sub_points = np.random.randint(0, n, size = subN_points)
        true_exps.append((c, sub_dims)) # Adding true explantions

        # Adding the points to the dataset
        for i in range(0, len(sub_points)):
            for j in range(0, len(sub_dims)):
                blobX[sub_points[i]][sub_dims[j]] = blobX_sub[i][j]
                blobY[sub_points[i]] = c

    # Feature Names
    f_n = ['f'+str(i+1) for i in range(0, p)]

    # Making dataframe
    df = pd.DataFrame(data=blobX, columns=f_n)

    # Returns the dataset, the labels, and the explanations for which features define each cluster.
    return df, blobY, true_exps



if __name__ == '__main__':
    np.random.seed(1)
    df, y, explan = generate2(2000, 100, 4)

    # 3D umap of the dataset, just to get some sense of what it "looks" like. 
    import umap
    reduction = umap.UMAP(densmap=True, n_components=3, random_state=1)
    embedded = reduction.fit_transform(df.to_numpy())

    # PLotting the 3 dimensional version
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], c=list(y))
    #ax.scatter(df['f1'], df['f2'], df['f3'], c=list(y))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    