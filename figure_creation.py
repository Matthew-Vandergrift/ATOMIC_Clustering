import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    print("Hello World!")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Importing the actual data
    data_file = pd.read_csv(
        (r"/home/matthewvandergrift/Desktop/Python_Code/topology/figure_3D.csv"), sep=",")
    recovered_dataframe = pd.DataFrame(data_file)


    # Adding to graph
    ax.scatter(recovered_dataframe["X"], recovered_dataframe["Y"], recovered_dataframe["Z"], c=recovered_dataframe["Label"])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(recovered_dataframe["X"], recovered_dataframe["Y"], c=recovered_dataframe["Label"])
    ax.legend()
    plt.show()