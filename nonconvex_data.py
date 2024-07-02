import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


# This is a very simple six-dimensional dataset containing non-convex clusters.
# It is made by projecting three different non-convex clustering datasets using uniform random
# additional variables. The 2d clustering datasets which are projected are located in the folder 
# 2d_data
def convex_gen():
    # Importing Jain
    jain = pd.read_csv("./2d_data/jain.txt", delimiter="\t")
    jain_df = pd.DataFrame(jain)
    jain_var = jain_df.loc[:, jain_df.columns != 'Label']
    jain_label = jain_df["Label"]

    # Importing Spiral
    spiral = pd.read_csv("./2d_data/spiral.txt", delimiter="\t")
    spr_df = pd.DataFrame(spiral)
    spr_var = spr_df.loc[:, spr_df.columns != 'Label']
    spr_label = spr_df["Label"]

    # Importing Flame 
    flame = pd.read_csv("./2d_data/flame.txt", delimiter="\t")
    flame_df = pd.DataFrame(flame)
    flame_var = flame_df.loc[:, flame_df.columns != 'Label']
    flame_label = flame_df["Label"]

    # Combining all three datasets

    # Augmenting Jain with Random
    df_rand1 = pd.DataFrame(np.random.rand(373, 2), columns=['XR','YR'])
    df_rand11 = pd.DataFrame(np.random.rand(373, 2), columns=['XR2','YR2'])
    jain_aug = pd.concat([jain_var, df_rand1, df_rand11], axis=1)
    #print(jain_aug.head())

    # Augmenting Spiral with Random
    df_rand2 = pd.DataFrame(np.random.rand(312, 2), columns=['X','Y'])
    df_rand21 = pd.DataFrame(np.random.rand(312, 2), columns=['XR2','YR2'])
    spr_var = spr_var.rename(columns={'X':'XR', 'Y':'YR'})
    spiral_aug = pd.concat([df_rand2, spr_var, df_rand21], axis=1)
    #print(spiral_aug.head())

    # Augmenting Flame with Random
    df_rand3 = pd.DataFrame(np.random.rand(240, 2), columns=['X','Y'])
    df_rand31 = pd.DataFrame(np.random.rand(240, 2), columns=['XR','YR'])
    flame_var = flame_var.rename(columns={'X':'XR2', 'Y':'YR2'})
    flame_aug = pd.concat([df_rand3, df_rand31, flame_var], axis=1)
    #print(flame_aug.head())
    
    # Building Final Dataframe
    #df_rand4 = pd.DataFrame(np.random.rand(100, 6), columns=['X','YR'])
    final_df = pd.concat([jain_aug, spiral_aug, flame_aug]).reset_index(drop=True)
    return final_df



if __name__ == '__main__':
    print("Hello World")
    # Generating the dataset
    final_df = convex_gen()

    # Umap embeeding of this data
    import umap
    reduction = umap.UMAP(densmap=False, random_state=1)
    embedded = reduction.fit_transform(final_df.to_numpy())
    labels = [0] * final_df.shape[0]
    fig, ax = plt.subplots()
    for g in np.unique(labels):
        kl = np.where(labels == g)
        ax.scatter(embedded[:, 0][kl], embedded[:, 1][kl], label=g)
    ax.legend()
    plt.show()