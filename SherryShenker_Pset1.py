'''Sherry Shenker
CMSC25400/STAT 27725
Problem Set 1: Clustering
Question 2
January 12th, 2017

Run this code by typing
 'python3 SherryShenker_Pset1.py [name of dataset] [number for k] ["k" or "k++" or "extra"] [number for tolerance] ["prefix for plots"]'

For example:

    python3 SherryShenker_Pset1.py toydata.txt 3 "k" 0.001 "run1_"

'''

import random
import numpy as np
import sys
import matplotlib.pyplot as plt

class Kmeans(object):
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

    def initialize(self):
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
        for index,c in enumerate(self.centers):
            d_test = self.calc_distance(c,p)
            if d_test < d:
                d = d_test
                new_index = index

        self.clusters[new_index].append(p_index)


    def update_centers(self):
        #calculate center of mass of every cluster to find new centers
        for index,cluster in enumerate(self.clusters):
            cluster_data = self.data[cluster]
            for d in range(self.dim):
                x = cluster_data[:,d]
                sum_x = np.sum(x)
                self.centers[index,d] = sum_x/len(cluster_data)


    def update_cost(self):
        #store cost data to calculate tolerance and to plot later
        #self.cost_array.append(self.new_cost)
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
                c = np.power(c,2)
                cost += c

        return cost

    def calc_distance(self,x,center):
        #find euclidean distance between two points
        d = 0
        for i in range(self.dim):
            x_i = x[i]
            center_i = center[i]
            d += abs(x_i-center_i)
        return d


    def check_convergence(self,tol):
        #check convergence of cost function
        if abs(self.new_cost - self.cost) < float(tol):
            return False
        else:
            return True

    def plot(self,name):
        #plot data assigned to each cluster
        fig = plt.figure()
        colors = ["r","g","b","c","m","y","k",'0.75','0.5','0.25']
        counter = 0
        for cluster in self.clusters:
            cluster_data = self.data[cluster]
            x = cluster_data[:,0]
            y =  cluster_data[:,1]
            plt.scatter(x,y,15,c=colors[counter])
            counter += 1
        plt.savefig(name)

    def initialize_plus(self):
        #kmeans++ initilization algorithm
        centers = []
        potential_centers = list(range(self.n))
        m1 = random.sample(range(self.n),1)[0]
        centers.append(m1)
        potential_centers.remove(m1)
        for m in range(self.k-1):
            total = 0
            prob_array = np.array([])
            for point in potential_centers:
                prob = self.calc_prob(self.data[point],centers)
                total += prob
                prob_array = np.append(prob_array,prob)
            prob_array = prob_array/total
            chosen = np.random.choice(potential_centers,p=prob_array)
            centers.append(chosen)
            potential_centers.remove(chosen)
        self.centers = self.data[centers]

    def calc_prob(self,point,centers):
        #calculate probability based on distance to chosen centers
        min_d = float("inf")
        for c in centers:
            d = self.calc_distance(self.data[c],point)
            if d < min_d:
                min_d = d
        prob = np.power(min_d,2)
        return prob





def read_data(data):
    #read toydata file
    with open(data,'r') as f:
        data_list = []
        for line in f:
            x,y = line.split()
            data_list.append((float(x),float(y)))
    #print(data_list[:5])
    return np.asarray(data_list)

def kmeans_single_run(f,k,tol,i,prefix):
    #single run of kmeans algorithm
    data = read_data(f)
    my_data = Kmeans(data,k)
    not_converged = True
    my_data.initialize()
    while not_converged:
        my_data.update_clusters()
        my_data.update_centers()
        my_data.update_cost()
        not_converged = my_data.check_convergence(tol)
    file_name = prefix + str(i) + "_kmeans.png"
    my_data.plot(file_name)
    return my_data.cost_array

def kmeans(f,k,tol,prefix):
    #wrapper function for kmeans algorithm
    arrays = []
    for i in range(20):
        cost_array = kmeans_single_run(f,k,tol,i,prefix)
        arrays.append(cost_array)
    fig = plt.figure()
    for array in arrays:
        plt.plot(array[1:])
    plt.title("Distortion Function by Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Distortion Function Value")
    filename = prefix + "kmeans_costs.png"
    plt.savefig(filename)

def kmeansplus_single_run(f,k,tol,i,prefix):
    data = read_data(f)
    my_data = Kmeans(data,k)
    not_converged = True
    my_data.initialize_plus()
    while not_converged:
        my_data.update_clusters()
        my_data.update_centers()
        my_data.update_cost()
        not_converged = my_data.check_convergence(tol)
    file_name = prefix + str(i) + "_plus.png"
    my_data.plot(file_name)
    return my_data.cost_array




def kmeansplus(f,k,tol,prefix):
    arrays = []
    for i in range(20):
        cost_array = kmeansplus_single_run(f,k,tol,i,prefix)
        arrays.append(cost_array)
    fig = plt.figure()
    for array in arrays:
        plt.plot(array[1:])
    plt.title("Distortion Function by Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Distortion Function Value")
    filename = prefix + "_plus_costs.png"
    plt.savefig(filename)

def extracredit(f,k,tol,prefix):
    plus_arrays = []
    for i in range(20):
        cost_array = kmeansplus_single_run(f,k,tol,i,prefix)
        print(cost_array[-1])
        plus_arrays.append(cost_array[-1])
    norm_arrays = []
    for i in range(20):
        cost_array = kmeans_single_run(f,k,tol,i,prefix)
        norm_arrays.append(cost_array[-1])
    diffs = []
    for i in range(len(norm_arrays)):
        diffs.append(plus_arrays[i]/norm_arrays[i])
    #print average ratio between costs
    print(np.mean(np.array(diffs)))




if __name__ == '__main__':
    if sys.argv[3] == "k":
        kmeans(sys.argv[1],sys.argv[2],sys.argv[4],sys.argv[5])
    elif sys.argv[3] == "extra":
        extracredit(sys.argv[1],sys.argv[2],sys.argv[4],sys.argv[5])
    else:
        kmeansplus(sys.argv[1],sys.argv[2],sys.argv[4],sys.argv[5])
