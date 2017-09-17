'''
Sherry Shenker
CMSC 254
Problem Set 4

run code by typing:
python3 sshenker_pset4_final.py [face img folder] [background img folder] [test image file name] [prefix for saving output] [value for final testing threshold] [number of training images] [number of feature types (integer from 1 to 3)] [stride length]
'''

import numpy as np
from scipy import misc
import math
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.patches as patches



def find_faces(faces_folder,nonfaces_folder,test_image,prefix=None,gamma=0,num_images = 500,features = 1,stride=4):
    '''
    train model using data in faces_folder and nonfaces_folder
    test model on test_image
    '''
    #lists of images converted to arrays of pixels
    faces = read_data(faces_folder)[:int(num_images)]
    nonfaces = read_data(nonfaces_folder)[:int(num_images)]

    num_nonfaces = len(nonfaces)

    # for latter knowing which background images have been removed from training set
    nonface_indexes = np.arange(0,len(nonfaces),1)

    #generate matrix of feature coordinates
    dim = 64
    feature_matrix = create_feature_matrix(dim,int(features),int(stride))
    print("Using {} features".format(len(feature_matrix)))

    #generate sum of pixels matrix for each image
    sum_faces = []
    sum_nonfaces = []

    for f in faces:
        sum_ = fill_pixel_sums(f)
        sum_faces.append(sum_)
    for f in nonfaces:
        sum_ = fill_pixel_sums(f)
        sum_nonfaces.append(sum_)

    nonfaces_remain = True

    cascade = []

    #create matrix to evaluate feature values for each feature and image

    feat_val_matrices_faces = []

    for i,f in enumerate(faces):
        l = []
        for feature in feature_matrix:
            l.append(calc_feature_value(f,feature,sum_faces[i]))
        feat_val_matrices_faces.append(l)

    feat_val_matrices_nonfaces = []

    for i,f in enumerate(nonfaces):
        l = []
        for feature in feature_matrix:
            l.append(calc_feature_value(f,feature,sum_nonfaces[i]))
        feat_val_matrices_nonfaces.append(l)

    #wrapper function to entire algorithm

    while nonfaces_remain:
        face_weights = initialize_weights(len(faces))
        nonface_weights = initialize_weights(len(nonface_indexes))
        continue_stage = True
        chosen_features = []
        counter = 0
        while continue_stage:
            face_weights, nonface_weights = normalize_weights(face_weights,nonface_weights)
            errors = []
            print("counter: {}".format(counter))
            counter += 1
            for index,feat in enumerate(feature_matrix):
                feat_vals_faces = []
                feat_vals_nonfaces = []
                for i,f in enumerate(faces):
                    val = feat_val_matrices_faces[i][index]
                    feat_vals_faces.append((i,val,1))
                for i in nonface_indexes:
                    val = feat_val_matrices_nonfaces[i][index]
                    feat_vals_nonfaces.append((i,val,-1))
                error,theta,polarity = calc_error(feat_vals_faces,feat_vals_nonfaces,face_weights,nonface_weights)
                errors.append((error,theta,polarity,index))
            lowest = sorted(errors,key=lambda x: x[0])[0]

            print("chose feature with error {}".format(lowest[0]))

            chosen_features.append(lowest)
            min_error = lowest[0]

            #calculate false positive rate and update weights

            theta, continue_stage,false_pos_indexes,false_neg_indexes = calc_false_pos(chosen_features,faces,nonface_indexes,feat_val_matrices_faces,feat_val_matrices_nonfaces)
            face_weights, nonface_weights = update_weights(face_weights,nonface_weights,lowest,feat_val_matrices_faces,feat_val_matrices_nonfaces,nonface_indexes)


            #end of stage, update training set
            if continue_stage == False:
                new_nonface_indexes = []
                for nf_index in nonface_indexes:
                    if nf_index in false_pos_indexes:
                        new_nonface_indexes.append(nf_index)
                nonface_indexes = new_nonface_indexes
        cascade.append(chosen_features)
        print("section of cascade has length {}".format(len(chosen_features)))

        #too few nonfaces left, terminate

        if len(nonface_indexes)/num_nonfaces < 0.01:
            nonfaces_remain = False
        else:
            print("round of boosting complete")
    #return cascade,cascade_thetas,feature_matrix
    print("FINAL CASCADE: {}".format(cascade))

    #test on image
    find_faces_test(test_image,cascade,float(gamma),feature_matrix,prefix)


def find_faces_test(test_image,cascade,gamma,feature_matrix,prefix):
    '''
    identify all faces in a given test_image, using the final strong classifier
    '''
    img = misc.imread(test_image, flatten=1)
    sum_ = fill_pixel_sums(img)
    max_x = img.shape[0]
    max_y = img.shape[1]
    step = 25
    patch_dimensions = []
    for x in np.arange(0,max_x-64,step):
        for y in np.arange(0,max_y-64,step):
            patch = img[x:x+64,y:y+64]
            sum_patch = sum_[x:x+64,y:y+64]
            #test if patch is classified as face
            if test_patch(patch,cascade,gamma,feature_matrix,sum_patch):
                patch_dimensions.append([y,x])

    draw_faces(patch_dimensions,img,prefix)


def test_patch(patch,cascade,gamma,feature_matrix,sum_patch):
    '''
    use final strong classifier to determine if patch is a face
    returns true if face and False otherwise
    '''
    image = True

    for stage_index,stage in enumerate(cascade):
        hypothesis = 0
        errors = []
        for feat in stage:
            error,theta,polarity,index = feat[:4]

            feat_val = calc_feature_value(patch,feature_matrix[index],sum_patch)
            alpha = math.log((1-error)/error)
            errors.append(alpha)

            hypothesis += alpha * np.sign(polarity*(feat_val-theta))

        if hypothesis + gamma*(np.sum(errors)) < 0:
            image = False
            break

    return image


def draw_faces(patch_dimensions,test_image,prefix):

    fig,ax = plt.subplots(1)
    merged = []
    indexes = []

    #merge overlapping identifications

    for index1,sq in enumerate(patch_dimensions):
        group = []
        if index1 not in indexes:
            for index2,sq2 in enumerate(patch_dimensions):
                if index2 != index1:
                    if (sq2[0] < sq[0] + 60) and (sq2[0] > sq[0] - 60) and (sq2[1] < sq[1] + 60) and (sq2[1] > sq[1] - 60):
                        indexes.append(index2)
                        group.append(sq2)

            if len(group) > 0:
                x = np.mean([i[0] for i in group])
                y = np.mean([i[1] for i in group])
            else:
                x = sq[0]
                y = sq[1]
            merged.append((x,y))

    # Display the image
    ax.imshow(test_image,cmap=plt.cm.gray)
    for patch in merged:
        rect = patches.Rectangle((patch[0],patch[1]),64,64,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    filename = prefix + "_output.png"
    fig.savefig(filename)

def calc_false_pos(chosen_features,faces,nonface_indexes,feat_val_matrices_faces,feat_val_matrices_nonfaces):
    '''
    check if false positive rate below 30%
    return true if rate above 30%, meaning keep going
    false, otherwise, stage over
    '''
    num_false_pos = 0

    face_scores = []

    #find value of big theta such that there are no false negatives

    for index,img in enumerate(faces):
        sum_w = 0
        for index_feat,f in enumerate(chosen_features):
            error = f[0]
            incorrect_face = f[-2]
            alpha = math.log((1-error)/error)
            polarity = f[2]
            little_theta = f[1]
            feature_number = f[3]
            val = feat_val_matrices_faces[index][feature_number]
            sum_w += alpha * np.sign(polarity*(val-little_theta))
        face_scores.append(sum_w)

    face_scores = np.array(face_scores)

    false_negatives = face_scores < 0

    big_theta = np.amin(face_scores[false_negatives]) if false_negatives.sum() > 0 else 0

    print("big theta: {}".format(big_theta))

    nonface_scores = []

    #find false positive rate

    for index in nonface_indexes:
        sum_w = 0
        for f in chosen_features:
            error = f[0]
            alpha = math.log((1-error)/error)
            polarity = f[2]
            little_theta = f[1]
            feature_number = f[3]
            val = feat_val_matrices_nonfaces[index][feature_number]
            sum_w += alpha * np.sign(polarity*(val-little_theta))
        nonface_scores.append((index,sum_w))


    false_positives = []

    for index,sum_w in nonface_scores:
        if sum_w - big_theta > 0:
            false_positives.append(index)


    #check if greater than 0.3
    rate = len(false_positives)/len(nonface_indexes)
    print("rate: {}".format(rate))
    print("false positives: {}".format(false_positives))

    if rate > 0.3:
        return big_theta, True, false_positives,false_negatives
    else:
        return big_theta, False, false_positives, false_negatives




def update_weights(face_weights,nonface_weights,lowest,feat_vals_matrices_faces,feat_val_matrices_nonfaces,nonface_indexes):
    '''
    update weights based on current learner that was chosen
    return two new lists of weights
    '''
    error,theta,polarity,feat_index = lowest[:4]
    new_face_weights = []
    new_nonface_weights = []
    beta = error/(1-error)
    for index,w in enumerate(face_weights):
        val = feat_vals_matrices_faces[index][feat_index]
        if polarity*val < polarity * theta:
            new_face_weights.append(w*beta)
        else:
            new_face_weights.append(w)

    for num,index in enumerate(nonface_indexes):
        val = feat_val_matrices_nonfaces[index][feat_index]
        if polarity*val < polarity * theta:
            new_nonface_weights.append(nonface_weights[num])
        else:
            new_nonface_weights.append(nonface_weights[num]*beta)
    return new_face_weights,new_nonface_weights

def read_data(folder):
    '''
    returns list of numpy arrays where each array is a single image
    represented by pixel intensities
    '''
    images = []
    for filename in os.listdir(folder):
        if filename[-3:] == "jpg":
            path = folder + filename
            img = misc.imread(path, flatten=1)
            images.append(img)
    return images

def initialize_weights(num_points):
    return [1/(2*num_points) for i in range(num_points)]

def normalize_weights(face_weights,nonface_weights):
    '''
    normalize weights to sum to one
    '''
    sum_weights = np.sum(face_weights) + np.sum(nonface_weights)
    new_face = [i/sum_weights for i in face_weights]
    new_nonface = [i/sum_weights for i in nonface_weights]

    return new_face,new_nonface

def fill_pixel_sums(image):
    '''
    calculate integral sums of image
    '''

    dimx = image.shape[0]
    dimy = image.shape[1]
    matrix = np.empty([dimx,dimy])
    for x in range(dimx):
        for y in range(dimy):
            if x > 0:
                matrix[x,y] = matrix[x-1,y] + np.sum(image[x,:y+1])
            elif y == 0:
                matrix[x,y] = image[x,y]
            else:
                matrix[x,y] = matrix[x,y-1] + image[x,y]
    return matrix


def create_feature_matrix(dim,features,stride):
    '''
    generate feature matrix with 1,2,or 3 types of features and
    given stride and image dimensions
    '''
    matrix = []
    #coordinates of horizontal features

    for w in np.arange(4,32,stride):
        for h in np.arange(4,32,stride):
            step_x = w
            step_y = h/2
            for x1 in np.arange(0,dim-w,w):
                for y1 in np.arange(0,dim-h,h):
                    matrix.append([x1,y1,
                                   x1+step_x,y1+step_y,
                                   x1,y1+step_y,
                                  x1+step_x,y1+2*step_y,1])
        #coordinates of vertical features
            step_x = w/2
            step_y = h
            for  x1 in np.arange(0,dim-w,w):
                for y1 in np.arange(0,dim-h,h):
                    matrix.append([x1,y1,
                                  x1+step_x,y1+step_y,
                                  x1+step_x,y1,
                                  x1+2*step_x,y1+step_y,1])
    #extra credit features: 3 rectangles
    if features > 1:
        for w in np.arange(3,12,3):
            for h in np.arange(1,12,2):
                for x in np.arange(0,dim-w,w):
                    for y in np.arange(0,dim-h,h):
                        matrix.append([x,y,x+w/3,y+h,
                                       x+w/3,y,x+2*(w/3),y+h,
                                       x+2*(w/3),y,x+w,y+h,2
                                      ])

    #extra credit features: 4 rectangles
    if features > 2:
        for w in np.arange(2,30,2):
            for h in np.arange(2,30,2):
                for x in np.arange(0,dim-w,w):
                    for y in np.arange(0,dim-h,h):
                        matrix.append([x,y,x+w/2,y+h/2,
                                       x+w/2,y,x+w,y+h/2,
                                       x,y+h/2,x+w/2,y+h,
                                       x+w/2,y+h/2,x+w,y+h,3
                                      ])

    return matrix

def calc_feature_value(image,feature_dims,pixel_sums):
    '''
    calculate the feature value of a given feature for a single image
    return a float
    '''
    dims = feature_dims[:-1]
    feat_type = feature_dims[-1]

    #two rectangle features
    if feat_type == 1:
        x1,y1,x1_,y1_,x2,y2,x2_,y2_ = dims
        i1 = (pixel_sums[x1_,y1_] + pixel_sums[x1,y1]
                 - pixel_sums[x1_,y1] - pixel_sums[x1,y1_])
        i2 = (pixel_sums[x2_,y2_] + pixel_sums[x2,y2]
                 - pixel_sums[x2_,y2] - pixel_sums[x2,y2_])
        return i1 - i2

    #three rectangle features
    elif feat_type == 2 :
        x1,y1,x1_,y1_,x2,y2,x2_,y2_,x3,y3,x3_,y3_ = dims
        i1 = (pixel_sums[x1_,y1_] + pixel_sums[x1,y1]
                 - pixel_sums[x1_,y1] - pixel_sums[x1,y1_])
        i2 = (pixel_sums[x2_,y2_] + pixel_sums[x2,y2]
                 - pixel_sums[x2_,y2] - pixel_sums[x2,y2_])
        i3 = (pixel_sums[x3_,y3_] + pixel_sums[x3,y3]
                 - pixel_sums[x3_,y3] - pixel_sums[x3,y3_])
        return i2 - i1 - i3

    #four rectangle features
    elif feat_type == 3:
        x1,y1,x1_,y1_,x2,y2,x2_,y2_,x3,y3,x3_,y3_,x4,y4,x4_,y4_ = dims
        i1 = (pixel_sums[x1_,y1_] + pixel_sums[x1,y1]
                 - pixel_sums[x1_,y1] - pixel_sums[x1,y1_])
        i2 = (pixel_sums[x2_,y2_] + pixel_sums[x2,y2]
                 - pixel_sums[x2_,y2] - pixel_sums[x2,y2_])
        i3 = (pixel_sums[x3_,y3_] + pixel_sums[x3,y3]
                 - pixel_sums[x3_,y3] - pixel_sums[x3,y3_])
        i4 = (pixel_sums[x4_,y4_] + pixel_sums[x4,y4]
                 - pixel_sums[x4_,y4] - pixel_sums[x4,y4_])
        return i1 + i4 - i2 - i3





def calc_error(feat_vals_faces,feat_vals_nonfaces,face_weights,nonface_weights):
    '''
    calculates the error for a specific feature
    '''
    all_vals = feat_vals_faces + feat_vals_nonfaces
    sorted_vals = sorted(all_vals, key=lambda x: x[1])
    error = []
    for i in range(len(sorted_vals)):
        #calculate error for the given cutoff
        left = sorted_vals[:i+1]
        right = sorted_vals[i+1:]

        indexes_left_face = [tup[0] for tup in left if tup[2]==1]
        indexes_left_non = [tup[0] for tup in left if tup[2]==-1]
        w_left_face = [face_weights[i] for i in indexes_left_face]
        w_left_nonface = [nonface_weights[i] for i,x in enumerate(indexes_left_non)]

        w_face = np.sum(face_weights)
        w_nonface = np.sum(nonface_weights)
        l = np.sum(w_left_face) +(w_nonface - np.sum(w_left_nonface))
        r = np.sum(w_left_nonface) + (w_face - np.sum(w_left_face))

        #store error value, division point, and polarity
        if l < r:
            polarity = 1
        else:
            polarity = -1
        error.append((min(l,r),i,polarity))

    #sort error from smallest to greatest
    sorted_error = sorted(error,key=lambda x: x[0])
    min_error = sorted_error[0][0]

    cut_off_index = sorted_error[0][1]

    if len(sorted_vals) > cut_off_index + 1:
        theta = sorted_vals[cut_off_index][1] + (sorted_vals[cut_off_index+1][1] - sorted_vals[cut_off_index][1])/2
    else:
        theta = sorted_vals[cut_off_index][1] + 0.00001

    polarity = sorted_error[0][2]

    return min_error,theta,polarity

if __name__ == '__main__':
    find_faces(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])
