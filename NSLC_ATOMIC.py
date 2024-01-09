import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

class Individual:
    def __init__(self, lis_of_vals, fitness=None, neighbours=None):
        self.vals = lis_of_vals
        self.fitness=fitness
        self.neighbours=neighbours

    def get_vals(self):
        return self.vals

    def update_fitness(self, fitness_val): # fitness val is a tuple ()
        self.fitness = fitness_val

    def get_fitness(self):
        return self.fitness

    def update_neighbours(self, list_of_neighbours): # List of neighbours should be a list of individuals
        self.neighbours = list_of_neighbours

    def get_neighbours(self):
        return self.neighbours

    def __gt__(self, other):
        if self.fitness[0] > other.fitness[0]:
            return True
        elif (self.fitness[0] == other.fitness[0] and self.fitness[1] < self.fitness[1]):
            return True
        else:
            return False

# Constant defined by the data (can be set up automatically)
P = 5
K_S = 3 # Determined through init mode
data_file = pd.read_csv((r"/home/matthewvandergrift/Desktop/Non-Mod Dataset/mod2_no_outliers.csv"))
recovered_dataframe = pd.DataFrame(data_file)
DATA_PD = recovered_dataframe.loc[:, recovered_dataframe.columns != 'Label']

# Constants needed for the algorithm
NUM_GEN = 200
MU = 100 # number of parents each iteration
LAM = 100 # size of population
CXPB = 0.5
MUTB = 0.3

def generate_population(n, p):
    population = []
    for i in range(n):
        ind = []
        for j in range(p):
            ind.append(random.randint(0,1))
        if sum(ind) <= 1:
            ind[random.randint(0, p-1)] = 1
            ind[random.randint(0, p-1)] = 1

        population.append(Individual(lis_of_vals=ind))
    return population

def pure_fitness(indiv, k_s, data_pd):
    '''Connected Components style evaluation'''

    # Building subset of dataset
    using_these = []
    feature_names = data_pd.columns.values
    for j in range(0, len(feature_names)): # Could be more efficient
        if indiv[j] == 1:
            using_these.append(feature_names[j])
    final_df = data_pd[using_these]

    size_of_set = len(using_these)

    if size_of_set <= 0:
        return "ERROR"

    # Evaluating Subset
    knn_graph = kneighbors_graph(final_df.to_numpy(), n_neighbors=k_s, mode='connectivity',
                                 include_self=True).toarray()
    path_graph = csr_matrix(knn_graph)
    n_components, labels = connected_components(csgraph=path_graph, directed=False, return_labels=True)
    f1 = -1 * (n_components-2)**2
    return(f1, size_of_set)

def tda_eval(indiv): # yet to be tested but in theory sound
    neighbours = indiv.get_neighbours() # list of Individual Objects, assumed to have fitness tuples=(f1, size)
    # Computing Local Competition Score and Sparseness
    num_outperformed = 0
    sparse_total = 0
    for i in neighbours:
        if indiv > i:
            num_outperformed += 1
        sparse_total += distance.hamming(indiv.get_vals, i.get_vals)
    sparse_total = sparse_total * (1/len(neighbours))
    return (num_outperformed, sparse_total)


def main():

    print("Start of Evolution")
    print("-------------------")

    # Generating the population
    pop = generate_population(n=LAM, p=P)

    # Building the KNN graph
    neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
    # Currently challenge is how to set the neighbours of each indiv for each generation.
    # Don't have to use the class atribute for this but would be nice



    print("Evaluated %i individuals" % len(pop))

    g = 0
    while g < NUM_GEN:
        g = g + 1
        print("Generation Number :", g)

        # Build KNN graph
        neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
        neigh.fit(pop)
        #


if __name__ == '__main__':
    main()

