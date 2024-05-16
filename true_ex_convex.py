# This file is implementing a data generation method from Fast algorithms for projected clustering
# Link : https://dl.acm.org/doi/pdf/10.1145/304181.304188
# The change I am making is simply just keeping track of which features define which clusters. 
# This is done in two ways 
#   METHOD 1: list of tuples [(c_1, [feature_set]), ..., (c_k, [feature_set]) ] (c_i, is the anchor point)
#   METHOD 2: Dictionary { datapoint index i : [feature_set]}
# Lastly we note that this dataset is making gaussian, i.e convex clusters.

import numpy as np
import pandas as pd 
from random import sample
rng = np.random.default_rng()


#Defining Constants used throughout
MIN_COOR = 0 
MAX_COOR = 100
OUTLIER_PER = 0.05
R = 2
S = 2

def generate(k, mu, d, N):
    '''Returns a pandas dataframe with the synthetic data, along with a list and dictionary for true explanations'''
    # Keeping track of data points in 2d array
    points = []

    # True Explanation Datastructures
    explan_list = []
    explan_dict = {}

    # Generating the anchor points 
    anchor_array = [rng.integers(low=MIN_COOR, high=MAX_COOR, size=d) for i in range(0, k)]
    
    # Generating exponentials used to find the number of points in each cluster
    exponentials = np.random.exponential(scale=1, size = k)
    
    # Flag for first anchor point 
    first_point = True 
    dims_chosen = []
    previous_amount = 0
    # Looping through the rest of the anchor points
    for i in range(0, len(anchor_array)):
        c_i = anchor_array[i]
        exp_i = exponentials[i]
        
        # Determining Number of dimensions assoicated with c_i 
        num_dims = min(max(2, np.random.poisson(mu, 1)), d) # The max(min()) functions bound it within [2,d]
        
        # Chosing dimensions
        if first_point == True:
            dims_chosen = rng.integers(low=0, high=d, size = num_dims)
            previous_chosen = dims_chosen
            previous_amount = num_dims
        else:
            amount_overlap = min(previous_amount, num_dims/2)
            dims_chosen = sample(previous_chosen, amount_overlap) 
            while len(dims_chosen) < num_dims:
                dims_chosen.append(np.random.uniform(low=0, high=d, size=1))
            previous_amount = num_dims
            previous_chosen = dims_chosen
            
        explan_list.append((c_i, dims_chosen))
        # Finding Number of Points
        num_in_ci = int((exp_i / sum(exponentials)) * (N*(1 - OUTLIER_PER)))

        # Building Points
        for i in range(0, num_in_ci):
            point_i = [None] * d
            for j in range(0, d):
                if j in dims_chosen:
                    scale = np.random.uniform(1, S, 1)
                    point_i[j] = np.random.normal(c_i, scale*R)
                else:
                    point_i[j] = np.random.uniform(MIN_COOR,MAX_COOR, size = 1)
            # Adding points to the overall dataset
            points.append(point_i)
            explan_dict[len(points)-1] = dims_chosen

    # Lastly adding random outlier points which are just uniform over the whole space
    while len(points) < N:
        outlier = np.random.uniform(MIN_COOR, MAX_COOR, size = d)
        points.append(outlier)
        explan_dict[(len(points)-1)] = [i for i in range(0, d)]

    df_points = pd.DataFrame(points, columns=['feature ' + str(i) for i in range(0, d)])
    return df_points, explan_list, explan_dict



if __name__ == '__main__':
    print("Hello World!")
    df_points, true_list, true_dict = generate(20, 5, 30, 1000)
    # Testing since not obvious 
    print("Number of Rows is : ", df_points.shape[0])
    print("Number of Columns is : ", df_points.shape[1])
    print("Sample Point Explanation :", true_dict[2])
    print("Random Element from Explanation list: ", true_list[3][1])