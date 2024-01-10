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
init_mode = False
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
global bank
creator.create("Fitness", base.Fitness, weights=(1.0,1.0))
creator.create("Individual", list, fitness=creator.Fitness)

def custom_population(n, p, data_pd, k_s):
    population = []
    for i in range(n):
        ind = []
        for j in range(p):
            ind.append(random.randint(0, 1))
        if sum(ind) <= 1:
            ind[random.randint(0, p - 1)] = 1
            ind[random.randint(0, p - 1)] = 1

        population.append(creator.Individual(ind))
        bank[int("".join(str(x) for x in ind), 2)] = tda_eval(ind, data_pd, k_s)

    return population

def tda_eval(indiv,data_pd, k_s):
    # Building subset of dataset
    using_these = []
    feature_names = data_pd.columns.values
    for j in range(0, len(feature_names)): # Could be more efficient
        if indiv[j] == 1:
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

def tuple_eval(first, next):
    if first[0] > next[0]:
        return True
    elif first[1] < next[1] and first[0] == next[0]: # Maybe I get rid of this?
        return True
    else:
        return False

def nslc_eval(indiv, data_pd, k_s, percent_bank, num_neighbours,p):
    # Getting individual as a list this is useful later
    list_indiv = list(indiv)

    # Building the k-neighbours graph based entirely on the bank of solutions
    all_in_bank_dec = list(bank.keys())
    all_in_bank = []
    for i in all_in_bank_dec:
        list_i = ([int(d) for d in str(bin(i))[2:]])
        while (len(list_i) != p):
            list_i.insert(0, 0)
        # print(len(list_i))
        all_in_bank.append(list_i)
    # print(all_in_bank)

    neigh = NearestNeighbors(n_neighbors=num_neighbours, radius=0.4)
    neigh.fit(all_in_bank)

    # Getting the pure fitness of the passed individual
    indiv_pure_fitness = tda_eval(list_indiv, data_pd, k_s)

    # Getting k neighbours of indiv and calculating sparseness and local competition
    num_outperformed = 0
    sparse_total = 0
    neighbour_index = neigh.kneighbors([list_indiv],return_distance=False)
    for i in range(0, num_neighbours):
        curr_neighbour = all_in_bank[neighbour_index[0, i]]
        if tuple_eval(indiv_pure_fitness, bank[int("".join(str(x) for x in curr_neighbour), 2)]):
            num_outperformed += 1
        sparse_total += distance.hamming(list_indiv, curr_neighbour)
    sparse_total = sparse_total / num_neighbours

    # Stochastically adding individuals to the bank
    roll = random.uniform(0, 1)
    if roll <= percent_bank:
        # Add to bank
        bank[int("".join(str(x) for x in list_indiv), 2)] = indiv_pure_fitness

    # Returning the two scores
    return (sparse_total, num_outperformed)

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

# This is a strange mate function will probably change later
def custom_mate(i1, i2):
    for i in range(0, len(i1)):
        if (i1[i] == 1 or i2[i] == 1) and (i1[i] != i2[i]):
            i1[i] = 1
            i2[i] = 1

    return(i1, i2)

def tda_display(i1, data_pd, true_labels, k_s,umap_data):
    '''Connected Components style evaluation'''
    li1 = list(i1)
    print(li1)
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

    encoded_labels = [1] * len(df)

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
    toolbox.register("evaluate", nslc_eval, data_pd=df, k_s= k_use, percent_bank = 0.05, num_neighbours = 8, p = df.to_numpy().shape[-1])
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate",custom_mutate, prob=0.45)
    toolbox.register("select", tools.selNSGA2)

    random.seed(64)

    pop = custom_population(n=100, p = df.to_numpy().shape[-1], data_pd=df, k_s=k_use)
    toolbox.register("select", tools.selNSGA2)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    NGEN = 200
    MU = 60
    LAMBDA = 100
    CXPB = 0.4
    MUTPB = 0.6
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)
    return hof, df, encoded_labels, k_use

if __name__ == "__main__":
    bank = {}
    h, ppmd, truth, k_use = main()
    # For viz we embeed the data
    import umap
    reduction = umap.UMAP(densmap=True, random_state=42)
    embedded = reduction.fit_transform(ppmd.to_numpy())
    p_m = 78
    for i in bank: # Will build some thing to filter the bank for final display. i.e a better selection algorithm. 
        list_i = ([int(d) for d in str(bin(i))[2:]])
        while (len(list_i) != p_m):
            list_i.insert(0, 0)

        tda_display(list_i, ppmd, truth, k_s=k_use, umap_data = embedded)