import sys
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
init_mode = True
# This time it's just looking for a feature subset
creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
creator.create("Individual", list, fitness=creator.Fitness)



def custom_population(n, p):
    population = []
    for i in range(n):
        ind = []
        for j in range(p):
            ind.append(random.randint(0,1))
        if sum(ind) <= 1:
            ind[random.randint(0, p-1)] = 1
            ind[random.randint(0, p-1)] = 1

        population.append(creator.Individual(ind))
    return population

def custom_mutate(i1, prob):
    init_sum = sum(i1)

    for i in range(0, len(i1)):
        rnd = random.uniform(0, 1)
        if rnd < prob:
            if init_sum <= 2:
                i1[i] = 1
            else:
                i1[i] = (i1[1] + 1) % 2

    return (i1,)

def custom_mate(i1, i2):
    for i in range(0, len(i1)):
        if (i1[i] == 1 or i2[i] == 1) and (i1[i] != i2[i]):
            i1[i] = 1
            i2[i] = 1

    return(i1, i2)

def tda_eval(i1,data_pd, k_s):
    '''Connected Components style evaluation'''
    li1 = list(i1)
    # Building subset of dataset
    using_these = []
    feature_names = data_pd.columns.values
    for j in range(0, len(feature_names)): # Could be more efficient
        if li1[j] == 1:
            using_these.append(feature_names[j])
    final_df = data_pd[using_these]

    size_of_set = len(using_these)

    if size_of_set > 1:
        # Evaluating Subset
        knn_graph = kneighbors_graph(final_df.to_numpy(), n_neighbors=k_s, mode='connectivity',
                                     include_self=True).toarray()
        path_graph = csr_matrix(knn_graph)
        n_components, labels = connected_components(csgraph=path_graph, directed=False, return_labels=True)
        f1 = -1 * (n_components-2)**2
        return(f1, size_of_set)
    else:
        return(-500, 1)

def tda_display(i1, data_pd, true_labels, k_s,umap_data):
    '''Connected Components style evaluation'''
    li1 = list(i1)
    # Building subset of dataset
    using_these = []
    feature_names = data_pd.columns.values
    for j in range(0, len(feature_names)): # Could be more efficient
        if li1[j] == 1:
            using_these.append(feature_names[j])

    # Checking sanity
    # using_these = ['506.5a', 'PC(O-16:3/2:0)', 'PC(18:3/0:0)', 'PC(20:5/0:0)']

    final_df = data_pd[using_these]

    size_of_set = len(using_these)
    if size_of_set > 1:
        # Evaluating Subset
        knn_graph = kneighbors_graph(final_df.to_numpy(), n_neighbors=k_s, mode='connectivity',
                                     include_self=True).toarray()
        path_graph = csr_matrix(knn_graph)
        n_components, labels = connected_components(csgraph=path_graph, directed=False, return_labels=True)
        f1 = -1 * (n_components-2)**2

        # Outputting Number of connected components
        print("Number of Connected Components are %s, F1=%s" %(n_components, f1))
        if n_components == 1:
            return None
        # Sil Calc
        sil_score = silhouette_score(final_df.to_numpy(), labels, metric='euclidean')
        # DB Calc
        db_score = davies_bouldin_score(final_df.to_numpy(), labels)
        # Ari Calc
        ari = adjusted_rand_score(labels, true_labels)
        print("Sil Score %s and DB Score %s, then ARI is %s" % (sil_score, db_score, ari))
        print("Number of Features used are :", size_of_set)
        print("Features used are : ", using_these)
        group = labels
        fig, ax = plt.subplots()
        for g in np.unique(group):
            kl = np.where(group == g)
            ax.scatter(umap_data[:, 0][kl], umap_data[:, 1][kl], label=g)
        ax.legend()
        plt.show()
    else:
        print("Size is 1")
    return None

def main():
    data_file = pd.read_csv((r"/home/matthewvandergrift/Desktop/Non-Mod Dataset/mod2_no_outliers.csv"))
    recovered_dataframe = pd.DataFrame(data_file)
    df = recovered_dataframe.loc[:, recovered_dataframe.columns != 'Label']
    # df2 = df
    #df = df.drop(columns = ['PC(14:1/0:0)', 'PC(16:1/0:0)', 'PC(18:6/0:0)', '536.4a']) # Cluster 1
    #df = df.drop(columns = ['PC(14:1/0:0)', 'PC(17:1/0:0)', '508.3c']) # Cluster 3 (508.3c)
    #df = df.drop(columns = ['PC(O-12:0/2:0)', 'PC(16:1/0:0)'])

    encoded_labels = [1] * len(df)
    # # Import data
    # data_file = pd.read_csv((r"/home/matthewvandergrift/Desktop/Python_Code/Metabolites_Data/synth_b_test.csv"), sep=",")
    # recovered_dataframe = pd.DataFrame(data_file)
    # true_labels = recovered_dataframe['Label']
    # recovered_dataframe = recovered_dataframe.drop(columns=["Label"])
    # encoded_labels = true_labels
    # # for i in true_labels:
    # #     if i == "RB":
    # #         encoded_labels.append(1)
    # #     else:
    # #         encoded_labels.append(0)
    # #Checking that there is no null elements
    # #print(recovered_dataframe.isnull().sum(axis = 0))
    # #Removing all constant columns
    # recovered_dataframe = recovered_dataframe.loc[:, recovered_dataframe.apply(pd.Series.nunique) != 1]
    # # Scaling the dataset
    # scaler = MinMaxScaler()
    # df = pd.DataFrame(scaler.fit_transform(recovered_dataframe.values), columns=recovered_dataframe.columns,
    #                   index=recovered_dataframe.index)
    # from sklearn.impute import KNNImputer
    # imputer = KNNImputer(n_neighbors=2)
    # imputed = imputer.fit_transform(df)
    # df = pd.DataFrame(imputed, columns=df.columns)

    # ####################################
    # # Step 0 Importing the SCBC Data
    # data_file = pd.read_csv((r"./test_data/Throat_GSE59102.csv"))
    # recovered_dataframe = pd.DataFrame(data_file)
    #
    # true_labels = recovered_dataframe['type']
    # encoded_labels = [0 if x == "larynx_squamous_cell_carcinoma" else 1 for x in true_labels]
    #
    # df = recovered_dataframe.loc[:, recovered_dataframe.columns != 'type']
    # df = df.drop(columns=["samples"])
    #
    # # Trying to scale the dataset
    # scaler = MinMaxScaler()
    # final_df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
    # df = final_df
    #
    # import umap
    # reduction = umap.UMAP(densmap=False, random_state=42)
    # embedded = reduction.fit_transform(df.to_numpy())
    # group = encoded_labels
    # fig, ax = plt.subplots()
    # for g in np.unique(group):
    #     kl = np.where(group == g)
    #     ax.scatter(embedded[:, 0][kl], embedded[:, 1][kl], label=g)
    # ax.legend()
    # plt.show()
    # ####################################

    if init_mode == True:
        for i in range(2, 30):
            k = i
            knn_graph = kneighbors_graph(df.to_numpy(), n_neighbors=k, mode='connectivity',
                                         include_self=True).toarray()
            path_graph = csr_matrix(knn_graph)
            n_components, labels = connected_components(csgraph=path_graph, directed=False, return_labels=True)
            print("For k=%s we have %s components" %(k, n_components))
    k_use = 6

    # DEAP Stuff
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register("evaluate", tda_eval, data_pd=df, k_s= k_use)
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate",custom_mutate, prob=0.45)
    toolbox.register("select", tools.selNSGA2)

    random.seed(64)

    pop = custom_population(n=100, p = df.to_numpy().shape[-1])
    toolbox.register("select", tools.selNSGA2)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    NGEN = 200
    MU = 30
    LAMBDA = 20
    CXPB = 0.4
    MUTPB = 0.6
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)
    return hof, df, encoded_labels, k_use

if __name__ == "__main__":
    h, ppmd, truth, k_use = main()
    # For viz we embeed the data
    import umap
    reduction = umap.UMAP(densmap=True, random_state=42)
    embedded = reduction.fit_transform(ppmd.to_numpy())

    for i in h:
        h_mod = list(i)
        tda_display(h_mod, ppmd, truth, k_s=k_use, umap_data = embedded)