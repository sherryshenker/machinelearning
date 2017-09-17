'''Sherry Shenker
CMSC25400/STAT 27725
Problem Set 1: Clustering
Question 2
January 12th, 2017

Run this code by typing
 'python3 SherryShenker_Pset1.py [name of dataset] [number for k] ["prefix for plots"]'

For example:

    python3 SherryShenker_Pset1.py toydata.txt 3  "run1_"

'''

import random
import numpy as np
import sys
import matplotlib.pyplot as plt

class EM(object):
    def __init__(self,data,k):
        #numpy array
        self.data = data
        #integer
        self.k = int(k)
        #list of lists where each list is a set of indexes to self.data
        self.clusters = [[] for x in range(self.k)]
        #list of k points
        self.centers = []
        #integer
        self.n = len(data)
        #dimension of each data point
        self.dim = len(data[0])
        #float
        self.cost = 0
        #float
        self.new_cost = 0
        #list of lists
        self.cost_array = []
        self.old_centers = None

    def initialize_random(self):
        ##randomly picks k points as the first set of cluster centers
        random_indexes = random.sample(range(self.n),self.k)
        self.centers = self.data[random_indexes]


    def update_clusters(self):
        #update clusters by minimizing cost function
        #clear current clusters
        self.clusters = [[] for x in range(self.k)]
        for p_index in range(self.n):
            point = self.data[p_index,:]
            self.assign_cluster(p_index,point)

    def assign_cluster(self,p_index,p):
        #assign point to closest cluster
        new_index = 0
        d = float("inf")


        self.clusters[new_index].append(p_index)


    def update_centers(self):
        #calculate center of mass of every cluster to find new centers

    def update_cost(self):
        #store cost data to calculate tolerance and to plot later
        self.cost = self.new_cost
        self.new_cost = self.calculate_cost()
        self.cost_array.append(self.new_cost)

    def calculate_cost(self):
        #calculate cost function using sum of squares
        cost = 0
        for index,cluster in enumerate(self.clusters):
            cluster_data = self.data[cluster]
            center = self.centers[index]
            for x in cluster_data:
                c = self.calc_distance(x,center)
                cost += c

        return cost

    def calc_distance(self,x,center):



    def check_convergence(self):
        #check if cluster centers moved
        if np.array_equal(self.centers,self.old_centers):
            return False
        else:
            return True

    def plot(self,name):
        #plot data assigned to each cluster
        fig = plt.figure()
        colors = ["r","g","b"]
        counter = 0
        for cluster in self.clusters:
            cluster_data = self.data[cluster]
            x = cluster_data[:,0]
            y =  cluster_data[:,1]
            plt.scatter(x,y,15,c=colors[counter])
            counter += 1
        plt.savefig(name)



    def calc_prob(self,point,centers):
        #calculate probability based on distance to chosen centers


def read_data(data):
    #read raw toydata file and convert to a numpy array
    with open(data,'r') as f:
        data_list = []
        for line in f:
            x,y = line.split()
            data_list.append((float(x),float(y)))
    return np.asarray(data_list)

def em_single_run(f,k,i,prefix):
    #single run of EM algorithm
    data = read_data(f)
    #instance of EM class
    my_data = EM(data,k)
    not_converged = True
    my_data.initialize_random()
    while not_converged:
        my_data.update_clusters()
        my_data.update_centers()
        my_data.update_cost()
        not_converged = my_data.check_convergence()
    file_name = prefix + str(i) + "_em.png"
    my_data.plot(file_name)
    return my_data.cost_array


def EM(f,k,prefix):
    #wrapper function to run EM algorithm 20 times
    arrays = []
    for i in range(20):
        cost_array = em_single_run(f,k,i,prefix)
        arrays.append(cost_array)
    fig = plt.figure()
    for array in arrays:
        plt.plot(array)
    plt.title("Distortion Function by Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Distortion Function Value")
    filename = prefix + "_em_costs.png"
    plt.savefig(filename)




if __name__ == '__main__':
    em(sys.argv[1],sys.argv[2],sys.argv[3])
