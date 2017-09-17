'''Sherry Shenker
CMSC 254
Problem Set 3
January 30th, 2017'''

import numpy as np
from matplotlib import pyplot as plt
import sys
import copy

def pca(file_name,prefix):
    '''run PCA algorithm for dimensionality reduction
    on file_name data and save plot with the specified prefix
    '''

    #read file
    X,labels = read_data(file_name)

    #center data
    cent_x = X - np.mean(X,axis=0)
    covariance_mat = np.cov([cent_x[:,0],cent_x[:,1],cent_x[:,2]])
    eig_val, eig_vec = np.linalg.eig(covariance_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    #sort by eigenvalue in decreasing order
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    #create 3 * 2 eigenvector matrix
    eig_matrix = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
    #generate new data by doing y = eig^t x X
    transformed = (eig_matrix.T.dot(X.T)).T
    plot_2d(transformed,labels,prefix)


def read_data(file_name):
    #read text file
    #return x,y,z matrix and labels for plotting
    with open(file_name,'r') as f:
        labels = []
        data_list = []
        for line in f:
            x,y,z,label = line.split()
            data_list.append((float(x),float(y),float(z)))
            labels.append(int(label))
    return np.asarray(data_list),labels

def plot_2d(matrix,labels,prefix):
    '''plot 2d data points colored by labels'''

    colors = ["r","g","b","m"]
    fig = plt.figure()
    for index,p in enumerate(matrix):
        plt.scatter(p[0],p[1],color=colors[labels[index]-1])
    name = prefix + "_pca.png"
    plt.savefig(name)


def calc_distance(p1,p2):
    '''
    calculate euclidean distance between p1 and p2
    returns float
    '''

    dist = 0
    for d in range(len(p1)):
        dist += abs(p1[d]-p2[d])

    return dist

def iso_map(file_name,prefix):
    '''implement isomap algorithm on a given file to classify data
    takes in a file with 3d data and prefix for saving the final plot of the data
    output a plot of 3d data plotted in 2d
    '''

    #read file
    data,labels = read_data(file_name)
    #ll = list()

    #construct matrix to store distances between all points
    ll = []
    for i in range(len(data)):
        ll.append(range(len(data)))
    distance_matrix = np.array(ll)

    #calculate 10 nearest neighbors and fill distance matrix
    total_dist = 0
    for index,point in enumerate(data):
        top_10 = {}
        for other_index,other_point in enumerate(data):
            x = calc_distance(point,other_point)
            total_dist += x
            if len(top_10) == 0:
                top_10[other_index] = x
            else:
                max_val = max(top_10, key=top_10.get)
                if (x < max_val):
                    top_10[other_index] = x
            if len(top_10) > 10:
                top_10 = sorted(top_10.items())
                top_10.pop(0)
                top_10 = dict(top_10)
        for other_index,other_point in enumerate(data):
            if other_index in top_10:
                distance_matrix[index,other_index] = top_10[other_index]
            elif other_index == index:
                distance_matrix[index,other_index] = 0
            else:
                distance_matrix[index,other_index] = total_dist + 1

    #compute shortest distances between all points
    shortest_paths = np.array(ll)
    for k in range(len(data)):
        #print(k)
        for i in range(len(data)):
            for j in range(len(data)):
                if distance_matrix[i,j] > (distance_matrix[i,k] + distance_matrix[k,j]):
                    shortest_paths[i,j] = distance_matrix[i,k] + distance_matrix[k,j]

    #classical MDS to find lower dimensional points
    n = len(data)
    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/n
    #centered matrix
    B = -H.dot(shortest_paths**2).dot(H)/2

    #print(B)
    print(B.shape)

    #cent = shortest_paths - np.mean(shortest_paths,axis=0)
    eig_val, eig_vec = np.linalg.eig(B)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    #sort by eigenvalue in decreasing order
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    #create 3 * 2 eigenvector matrix
    l1 = eig_pairs[0][0]
    l2 = eig_pairs[1][0]
    eig_matrix = np.hstack((eig_pairs[0][1].reshape(n,1), eig_pairs[1][1].reshape(n,1)))
    root = np.matrix([[l1, 0], [0, l2]])
    root = np.sqrt(root)
    #print(eig_matrix.shape)
    transformed = eig_matrix.dot(root)
    transformed = transformed.T
    #data.T.dot(eig_matrix)
    #generate new data by doing y = eig^t x X
    #transformed = (eig_matrix.T.dot(data.T)).T
    plot_2d(transformed,labels,prefix)

def split_data(data,labels,k):
    '''split data into k parts'''


    p = np.random.permutation(len(data))
    np.take(data,p,axis=0,out=data)
    shuffled_labels = []

    for index in p:
        shuffled_labels.append(labels[index][0])


    split_data = []
    split_labels = []
    step = len(data)/k
    counter = 0
    for i in range(k):
        split_labels.append(shuffled_labels[int(counter):int(counter+step)])
        split_data.append(data[int(counter):int(counter+step),:])
        counter += step

    return split_data,split_labels

def read_digits(digits,labels):
    '''
    read in text files containing digits and corresponding labels
    output numpy array containing digit data and list containing labels
    '''
    with open(digits,'r') as f:
        data_list = []
        for line in f:
            l = line.split()
            l = [float(i) for i in l]
            data_list.append(l)
    with open(labels,'r') as f:
        labels = []
        for line in f:
            l = line.split()
            l = [float(i) for i in l]
            labels.append(l)

    return np.asarray(data_list),labels


def choose_k(digits,labels):
    k_data,labels = split_data(digits,labels,10)
    error_dict = {}



    for k in range(10):
        test_data = k_data[k]
        test_labels = labels[k]
        train_data = k_data[:k]
        train_data.extend(k_data[k+1:])
        train_labels = labels[:k]
        train_labels.extend(labels[k+1:])

        train_matrix = train_data[0]

        #reconstruct training matrix after having removed testing data
        for j in range(8):
            train_matrix = np.append(train_matrix,train_data[j+1],axis=0)

        train_labels2 = []
        for sub in train_labels:
            for label in sub:
                train_labels2.extend([label])

        #create enlarged training data where we repeat the data "k" times
        dup_train_labels = []

        if k == 0:
            dup_train_labels.extend(train_labels2)
        else:
            train_matrix = np.tile(train_matrix,(k+1,1))
            dup_train_labels.extend(train_labels2)
            for i in range(k):
                dup_train_labels.extend(train_labels2)


        n = len(train_matrix[0])
        w = np.zeros(n)

        for index,x in enumerate(train_matrix):
            #print(index)
            if np.dot(w,x) > 0:
                predict = 1
            else:
                predict = -1

            if ((predict == -1)&(dup_train_labels[index]==1)):
                w = w + x
            elif ((predict == 1)&(dup_train_labels[index]==-1)):
                w = w - x

        #test on "kth" portion of data
        error = 0

        for index,x in enumerate(test_data):
            if np.dot(w,x) > 0:
                predict = 1
            else:
                predict = -1
            true = test_labels[index]
            if (predict != true):
                error += 1

        error_dict[k] = error

    min_error = min(error_dict, key=error_dict.get)
    return min_error
    #print(min_error)







def perceptron(digits,labels,test_digits):
    '''implemenet perceptron algorithm
    digits file contains training data
    labels contains training data labels
    test_digits contains testing data
    ouputs a text file containing predicted labels for test_digits'''

    data,labels = read_digits(digits,labels)
    #k_data = split_data(data,k)
    n = len(data[0])
    eta = 0.01

    #initiate weights to zero
    final_w = np.zeros(n)

    #get rid of later


    k = choose_k(data,labels)
    print(k)
    data = np.tile(data,(k+1,1))
    print("shape")
    print(data.shape)
    new_labels = []
    new_labels.extend(labels)
    for i in range(k):
        new_labels.extend(new_labels)

    predictions = np.zeros(len(data))
    print("LENGTH")
    print(len(predictions))

    w = np.zeros(n)

    for index,x in enumerate(data):
        if np.dot(w,x) > 0:
            predict = 1
        else:
            predict = -1
        predictions[index] = predict

        #update weight vector

        if ((predict == -1)&(new_labels[index][0]==1)):
            w = w + x
        elif ((predict == 1)&(new_labels[index][0]==-1)):
            w = w - x

    #predict on test data

    #read test data
    with open(test_digits,'r') as f:
        test_data = []
        for line in f:
            l = line.split()
            l = [float(i) for i in l]
            test_data.append(l)

    test_data = np.array(test_data)

    test_predictions = []
    for index,x in enumerate(test_data):
        if np.dot(w,x) > 0:
            predict = 1
        else:
            predict = -1
        test_predictions.append(predict)

    #output predictions to text file
    with open("test35.predictions",'w') as f:
        for p in test_predictions:
            f.write(str(p) + "\n")



if __name__ == '__main__':
    if sys.argv[1] == "PCA":
        pca(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == "ISO":
        iso_map(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == "PER":
        perceptron(sys.argv[2],sys.argv[3],sys.argv[4])
