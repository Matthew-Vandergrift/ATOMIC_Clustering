# This is the ATOMIC algorithm for finding clusters defined by spefifc variables. This does not include the metrics 
# used to evaluate it's performance (Silhouette metric etc) these are present in the file compare_atomic.py 

# This algorithm makes use of the DEAP package which can be found at Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, 
# “DEAP: Evolutionary Algorithms Made Easy”, Journal of Machine Learning Research, pp. 2171-2175, no 13, jul 2012. 

# Imports
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from math import inf
import umap
# Sythnetic Data-Generation Imports 
from convex_synth_data import generate2
from true_ex_convex import generate

# This controls if the method begins by finding an optimal k to use, can be turned off if this is already known. 
init_mode = True
# This defines the archive used for NSLC, it is called the bank
global bank

# Set up for DEAP engine. 
creator.create("Fitness", base.Fitness, weights=(1.0,1.0))
creator.create("Individual", list, fitness=creator.Fitness)

# Builds the intial population of random feature spaces
def custom_population(n, p, data_pd, k_s):
    population = []
    for i in range(n):
        ind = []
        for j in range(p):
            ind.append(random.randint(0, 1))
        if sum(ind) < 2: # Might need to make this equal
            change_index = random.randint(0, p - 2)
            ind[change_index] = 1
            ind[change_index+1] = 1

        population.append(creator.Individual(ind))
        bank[int("".join(str(x) for x in ind), 2)] = tda_eval(ind, data_pd, k_s)
    return population

# Computes number of connected components in a passed feature spaced defined by indiv
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
        return "ERROR"  

# Utility Tuple Evaluation function, evalues tuples which are (f, #features)
def tuple_eval(first, next):
    if first[0] > next[0]:
        return True
    elif first[1] < (next[1]//2) and first[0] == next[0]:
        return True
    else:
        return False

# The evaluation for NSLC, i.e computes and returns Novelty and Local Competition 
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
    if roll <= percent_bank: # Should add something where if this is zero the odds are bumped up by a lot
        # Add to bank
        bank[int("".join(str(x) for x in list_indiv), 2)] = indiv_pure_fitness

    # Returning the two scores
    return (sparse_total, num_outperformed)

# Mutation Function
def custom_mutate(i1, prob):
    for i in range(0, len(i1)):
        rnd = random.uniform(0, 1)
        if rnd < prob:
            i1[i] = (i1[1] + 1) % 2
    if sum(list(i1)) <= 2:
        change_index = random.randint(0, len(i1) - 2)
        i1[change_index] = 1
        i1[change_index+1] = 1

    return (i1,)

# Crossover Function 
def custom_mate(i1, i2):
    for i in range(0, len(i1)):
        if (i1[i] == 1 or i2[i] == 1) and (i1[i] != i2[i]):
            i1[i] = 1
            i2[i] = 1

    return(i1, i2)

# TDA Display
def tda_display(i1, data_pd, k_s,umap_data):
    '''Connected Components style evaluation'''
    li1 = list(i1)

    # Building subset of dataset
    using_these = []
    feature_names = data_pd.columns.values
    for j in range(0, len(feature_names)): # Could be more efficient
        if li1[j] == 1:
            using_these.append(feature_names[j])
    print("Using Features :", using_these)
    

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

        group = labels
        fig, ax = plt.subplots()

        for g in np.unique(group):
            kl = np.where(group == g)
            ax.scatter(umap_data[:, 0][kl], umap_data[:, 1][kl], label=g)
        ax.legend()
        plt.show()
        return labels
    else:
        print("Size is 1")
    return None

# Function to choose 'stable k'.
def choose_k(seen_tuples):
    '''Takes in a list of tuples (k, number of connected componenets) returns \hat{k} to use'''
    previous_tuple = (0, inf)
    for i in seen_tuples:
        if i[1] == previous_tuple[1]:
            return previous_tuple[0]
        previous_tuple = i
    # If none found use last checked k value
    return seen_tuples[-1]

# the main atomic function which houses the evolutionary loop. 
def atomic(dataset):
    # Since data is generated need random seed for reproducibility
    np.random.seed(3) # 3 
    df = dataset

    # This is where the k value to be used is chosen, can be skipped if init_mode is false
    seen_list = []
    if init_mode == True:
        for i in range(2, 30):
            k = i
            knn_graph = kneighbors_graph(df.to_numpy(), n_neighbors=k, mode='connectivity',
                                         include_self=True).toarray()
            path_graph = csr_matrix(knn_graph)
            n_components, labels = connected_components(csgraph=path_graph, directed=False, return_labels=True)
            print("For k=%s we have %s components" %(k, n_components))
            seen_list.append((k, n_components))
    k_use = choose_k(seen_list)
    
    #(used 40 for the convex sy, 10 for non-convex) #30 for convex, 10 for non-convex (15th)

    # Code for the using the DEAP package  
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register("evaluate", nslc_eval, data_pd=df, k_s= k_use, percent_bank = 0.05, num_neighbours = 8, p = df.to_numpy().shape[-1])
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate",custom_mutate, prob=0.45)
    toolbox.register("select", tools.selNSGA2)

    # Random Seed for the Evolutionary Algorithm
    random.seed(64)

    # Building the initial population
    pop = custom_population(n=100, p = df.to_numpy().shape[-1], data_pd=df, k_s=k_use)
    toolbox.register("select", tools.selNSGA2)
    # Want to return the Pareto Front of solutions found over the objectives of Novelty and Local Competition 
    hof = tools.ParetoFront()
    # Defining Statistics for DEAP to display as algorithm runs
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Settings for the evolutionary algorithm
    NGEN = 100
    MU = 50
    LAMBDA = 60
    CXPB = 0.1
    MUTPB = 0.6
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)
    # The algorithm returns the hall of fame, and which k value was used to find them. 
    return hof, k_use

# In the main function we call ATOMIC, and visualize individual results. 
if __name__ == '__main__':
    np.random.seed(3)

    # Importing/Generating the Data 
    dataset, y, explan = generate2(1000, 100, 3) # This is a convex high-dimensional synthetic dataset
    # df = convex_gen() # This is a six-dimensional synthetic non-convex dataset

    # Setting up the archive
    bank = {}
    # Running ATOMIC 
    h, k_use = atomic(dataset)

    # To visualize the clusters, we need to project the dataset down to two dimensions and hence we need u-MAP. 
    reduction = umap.UMAP(densmap=False, random_state=1)
    embedded = reduction.fit_transform(dataset.to_numpy())
    p_m = dataset.to_numpy().shape[-1]
    
    # Currently a more advanced method for automatic selection of best results is in progress. 
    # Storage for storing found solutions
    sols = [] 
    for i in bank: 
        list_i = ([int(d) for d in str(bin(i))[2:]])
        while (len(list_i) != p_m):
            list_i.insert(0, 0)

        if sum(list_i) < 30:
            sols.append((sum(list_i), list_i))
    
    # Sorting Solutions by number of variables used 
    label_index = 0
    sorted_by_first = sorted(sols, key=lambda tup: tup[0])
    for i in sorted_by_first:
        print("Showing Labels at Index %s" %(label_index))
        curr_labels = tda_display(i[1], dataset, k_s=k_use, umap_data = embedded)
        # Saving Solutions to a file for the purposes of combining them 
        np.savetxt('myarray'+str(label_index)+'.txt', curr_labels)
        label_index += 1

    # #Checking Loading
    # text_file = open("myarray1.txt", "r")
    # lines = text_file.readlines()
    # lines = [int(float(line.rstrip('\n'))) for line in lines]
    # print(lines)
    # print(len(lines))
    # text_file.close()
