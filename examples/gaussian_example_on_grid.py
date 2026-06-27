#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:25:22 2026

@author: andrew
"""

import numpy as np
import networkx as nx
from numpy.random import RandomState
from scipy.linalg import inv
import pandas as pd
from collections import deque
from math import ceil
from partopt_plus.Partition_Searcher import Partition_Searcher


# Number of areal units (must be a square number for the grid setup)
n = 196
rs = np.sqrt(n).astype('int')

# Number of covariates to cluster
d = 4

# Number of particles
L = 5

# Number of observations per area (is a constant value over all areas, 
# code cannot handle different numbers of observations per area )
m = 5

# Spatial strength
rho = 0.95

# Tempering parameter (higher numbers force unique particles)
mu = 1000

# Hyper-prior values for EPPF of pitman-yorr
prior_1 = 1
prior_2 = 0.0

# Separation between cluster means
max_range = 10

# Assignment used in the example (any number from 0 to 35)
ass = 29

# Variance of the gaussian likelihood
tau = 0.5

# 
lambda_ = .5
seed = 12345
tau_prior_1 = 0.5
tau_prior_2 = 0.5
k_0 = 0.01

######---------------######
######DATA GENERATION######
######---------------######

def gen_X(n_obs,m,d,constant,zero,seed):
    prng = RandomState(seed)
    X = prng.normal(1,1,size=(n,m,d))
    if constant:
        X[:,:,0] =  1
    return X

def Sigma_mat(W,rho):
    n_k = W.shape[0]
    F = [1-rho+rho*np.sum(W[i,:]) for i in range(n_k)]
    Sigma_mat = np.diag(F)-rho*W
    return inv(Sigma_mat)


X = gen_X(n,m,d,1,0,seed)

W = nx.generators.lattice.grid_2d_graph(rs,rs, periodic=False)
W.add_edges_from([((x, y), (x+1, y+1)) for x in range(rs-1) for y in range(rs-1)]+[((x+1, y), (x, y+1)) for x in range(rs-1) for y in range(rs-1)]) 
W_mat = nx.adjacency_matrix(W).toarray()

assignment = np.load('assignments.npy')[ass]
num_clusters = np.max(np.unique(assignment))+1
prng = RandomState(seed)
theta_ks = prng.normal(
    loc=0.0,
    scale=np.sqrt((tau*lambda_) / k_0),
    size=(num_clusters, d)
)

order = []
thetas = np.zeros((n,d))
for index,cluster in enumerate(range(max(assignment)+1)):
    theta_k = theta_ks[index]
    indicies = [i for i, x in enumerate(assignment) if x == cluster]
    n_k = len(indicies)
    order = order+indicies
    Sigma = Sigma_mat(W_mat[indicies,:][:,indicies],rho)
    thetas[indicies] = prng.multivariate_normal(np.kron(np.ones((n_k,1)),np.eye(d))@theta_k,tau*lambda_*np.kron(Sigma,np.eye(d))).reshape(n_k,d)

Y = np.einsum('ik,ijk->ij', thetas, X)


######---------------######
######RUNNING PARTOPT+######
######---------------######

# Make a searcher object
searcher = Partition_Searcher(assignment, #actual assignment (used for simulation tests can be np.zeros(n))
                            Y, #response observations numpy array of shape (n,m)
                            X, #covariate observations numpy array of shape (n,m,d)
                            nx.from_numpy_array(W_mat), #Graph of adjacency matrix (must be fully connected)
                            L=L,
                            Kmeans_initliaise=1, #Initialise with k-means
                            mu =mu,
                            lambda_=lambda_,
                            k_min = 2,k_max = 4,
                            k_0=k_0,rho=rho,
                            alpha=tau_prior_1,beta=tau_prior_2
                            ,prior_alpha=prior_1,prior_theta=prior_2,
                            z_start=0 #initialise with destroy/repair moves
                            )
# Create the particle set according to initialisation choice
searcher.instantiate_set()

# Find the optimal particle set.  both equal to 1 gives full neighbourhood.
searcher.optimise_PARTOPT(zeal=1, global_m=1)



