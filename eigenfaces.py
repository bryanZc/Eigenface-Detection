#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:11:18 2018

@author: chengzhong
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def image_to_matrix():
    fullpath = "FACESdata/s1/1.pgm"
    path_list = [fullpath]
    img1 = Image.open(fullpath).convert('L')
    imagearray1 = np.array(img1)
    original_shape = imagearray1.shape
    flat1 = imagearray1.ravel()
    facevector1 = np.matrix(flat1)
    facematrix = facevector1
    #shape = flat1.shape
    #print(shape)
    
    n = 10
    file_counter = 0
    for j in range(1,41):
        for i in range(1,n+1):
            fullpath = "FACESdata/s"+str(j)+"/"+str(i)+".pgm"
            path_list.append(fullpath)
            #print(fullpath)
            img = Image.open(fullpath).convert('L')
            # make a 1-dimensional view of imagearray
            imagearray = np.array(img)
            # convert it to a matrix
            flat = imagearray.ravel()
            facevector = np.matrix(flat)
            facematrix = np.r_[facematrix,facevector] # row cat
            file_counter = file_counter+1
    
    facematrix = np.delete(facematrix, (0), axis=0)
    facematrix_t = np.transpose(facematrix)
    #print(facematrix) 
    print("The Transpose of the matrix is: ")
    print(facematrix_t)
    print("The sahpe is:")
    print(facematrix_t.shape)
    return path_list,original_shape,facematrix,facematrix_t

def plot_mean_face(facematrix_t,original_shape):
    matrix_mean = np.mean(facematrix_t, axis = 1)
    print("The mean vector is: ")
    print(matrix_mean) 
    Norm_Face_Matrix = facematrix_t - matrix_mean
    print("The Normalized Matrix is: ")
    print(Norm_Face_Matrix) 
    print("The Normalized Matrix shape is: ")
    print(Norm_Face_Matrix.shape)
    
    mean_face = plt.figure()
    print("=========================================================")
    print("The mean face is: ")
    plt.imshow(matrix_mean.reshape(original_shape), cmap = plt.get_cmap("gray"))
    plt.show()
    mean_face.savefig("mean_face.jpg")
    return matrix_mean,Norm_Face_Matrix

def image_pca(Norm_Face_Matrix):
    Norm_Face_Matrix_t = np.transpose(Norm_Face_Matrix)
    CovMatrix = np.matmul(Norm_Face_Matrix_t,Norm_Face_Matrix)
    #print("The eigenvalues are\n",CovMatrix)    
    evals,evects = np.linalg.eig(CovMatrix)
    ## Order eigenvalues
    evals = evals[evals.argsort()[::-1]]
    evects = evects[evals.argsort()[::-1]]
    evals = evals.real
    evects = evects.real
    print("The eigenvalues are\n",evals)
    print("The eigenvectors are\n:",evects)   
    return evals,evects

def k_select():
    ## choose k
    flag = True
    while flag:
        print("=========================================================")
        k = int(input("Please Enter the K value (Between 1 and 400): "))
        if k in range(1,400):
            flag = False
        else:
            print("Please input a number between 1 and 400")
    return k

def projected_eigenvec(k,Norm_Face_Matrix,evects,original_shape):
    ## choose top k eigenvectors
    eigenface_matrix = np.matmul(Norm_Face_Matrix, evects[:,0:k])
    print("The top ",k,"\teigenvectors:")
    print(eigenface_matrix)    
    #figure = plt.figure()
    for i in range(k):
        figure = plt.figure()
        plt.imshow(eigenface_matrix[:,i].reshape(original_shape), cmap = plt.get_cmap("gray"))
        plt.show()
        figure.savefig("eigenface"+str(i+1)+".jpg")
    return eigenface_matrix

def face_prediction(filename,Norm_Face_Matrix,matrix_mean,eigenface_matrix,path_list,k):
    file = plt.figure()
    test_image = Image.open(filename).convert('L')
    print("The test image input is: ")
    plt.imshow(test_image, cmap = plt.get_cmap("gray"))
    plt.show()
    file.savefig("TEST_Image.jpg")
    
    test_eigenface = np.array(test_image).ravel() - matrix_mean.ravel()
    test_k_values = np.matmul(np.transpose(eigenface_matrix), np.transpose(test_eigenface))
    
    v = np.zeros([400, k]) 
    dist = [0 for i in range(400)] 
    
    for i in range(400):
        temp = np.matmul(np.transpose(eigenface_matrix), Norm_Face_Matrix[:,i])
        v[i,:] = np.transpose(temp)
        dist[i] = np.sum(np.square(v[i,:] - np.transpose(test_k_values)))
    
    ## Use euclidean distance to match image
    num_file = dist.index(min(dist))
    print("=========================================================")
    print("Closest File Path: ", path_list[num_file+1])
    print("The Euclidean Distance is: ", min(dist))
    result = Image.open(path_list[num_file+1]).convert('L')
    figure = plt.figure()
    plt.imshow(result, cmap = plt.get_cmap("gray"))
    plt.show()
    if min(dist) == 0:
        print("=========================================================")
        print("Same Face Detected!")
        figure.savefig("PREDICTED_Image.jpg")
    else:
        print("=========================================================")
        print("Showing the Cloest Face!")
        figure.savefig("PREDICTED_Image.jpg")

if __name__ == "__main__":
    path_list,original_shape,facematrix,facematrix_t = image_to_matrix()
    matrix_mean,Norm_Face_Matrix = plot_mean_face(facematrix_t,original_shape)
    evals,evects = image_pca(Norm_Face_Matrix)
    k = k_select()
    eigenface_matrix = projected_eigenvec(k,Norm_Face_Matrix,evects,original_shape)
    filename1 = 'TEST_Image.pgm'
    #filename2 = 'TEST_Image_2.pgm'
    face_prediction(filename1,Norm_Face_Matrix,matrix_mean,eigenface_matrix,path_list,k)
    #face_prediction(filename2,Norm_Face_Matrix,matrix_mean,eigenface_matrix,path_list,k)

