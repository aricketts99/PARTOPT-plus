#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:18:59 2026

@author: andrew
"""

import numpy as np
import networkx as nx
from numpy.random import RandomState
from scipy.linalg import inv
import pandas as pd
from collections import deque
from math import ceil
from partopt_plus.Poisson_Partition_Searcher import Poisson_Partition_Searcher
from partopt_plus.utils import fit_Poisson_individual


# Number of areal units (must be a square number for the grid setup)
n = 196
rs = np.sqrt(n).astype('int')

# Number of covariates to cluster
d = 4

# Number of non-clustered global covariates
d_beta = 4

# Number of particles
L = 5

# Number of observations per area (is a constant value over all areas, 
# code cannot handle different numbers of observations per area )
m = 15

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
tau = 0.05

# Variance of beta generation
sigma = 0.1
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
    X = prng.normal(0,0.005,size=(n,m,d))
    if constant:
        X[:,:,0] =  1
    return X

def Sigma_mat(W,rho):
    n_k = W.shape[0]
    F = [1-rho+rho*np.sum(W[i,:]) for i in range(n_k)]
    Sigma_mat = np.diag(F)-rho*W
    return inv(Sigma_mat)

# Covariate data to cluster
X = gen_X(n,m,d,1,0,seed)

# Non-clustered covariate data
Z = gen_X(n,m,d_beta,0,0,seed)

# Offset
C = np.random.rand(n,m)

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

clustered_term = np.einsum('ik,ijk->ij', thetas, X)

betas = prng.normal(0,sigma,size = d_beta)

global_term = np.einsum('k,ijk->ij', betas, Z)

Y = prng.poisson(np.exp(clustered_term+global_term+C))

indiviudal_area_fit = fit_Poisson_individual(Y,X,Z,C)

starting_point = indiviudal_area_fit.x

estimated_thetas = starting_point[:n*d]

def fourth_central_moment(X):
    mu = np.mean(X, axis=0)          # shape (d,)
    centered = X - mu                # broadcast subtraction
    m4 = np.mean(centered**4, axis=0)
    return m4

const = np.power((1-rho),-1)+np.power(k_0,-1)

b_1 = 0.25*np.min((np.max(estimated_thetas.reshape(n,d).T[1:].T,axis=0)-np.min(estimated_thetas.reshape(n,d).T[1:].T,axis=0))**2)/(const*(np.log(n)+1)**2)
b_2 = np.median(np.var(estimated_thetas.reshape(n,d).T[1:].T,axis=0,ddof=1)/const)

a_11 = 2+b_1
a_12 = 2+b_2

beta_11 = (a_11-1)*(a_11-2)
beta_12 = (a_12-1)*(a_12-2)

a_21 = (2*np.median(fourth_central_moment(estimated_thetas.reshape(n,d).T[1:].T))/(3*const**2)-b_1**2)/(np.median(fourth_central_moment(estimated_thetas.reshape(n,d).T[1:].T))/(3*const**2)-b_1**2)
a_22 = (2*np.median(fourth_central_moment(estimated_thetas.reshape(n,d).T[1:].T))/(3*const**2)-b_2**2)/(np.median(fourth_central_moment(estimated_thetas.reshape(n,d).T[1:].T))/(3*const**2)-b_2**2)

beta_21 = np.sqrt(np.median(fourth_central_moment(estimated_thetas.reshape(n,d).T[1:].T))*(a_21-1)*(a_21-2)/(3*const))
beta_22 = np.sqrt(np.median(fourth_central_moment(estimated_thetas.reshape(n,d).T[1:].T))*(a_22-1)*(a_22-2)/(3*const))

alpha_found = a_21
beta_found = beta_21



print(alpha_found)
print(beta_found)
print('tau found: '+str(beta_found/(alpha_found-1)))

######---------------######
######RUNNING PARTOPT+######
######---------------######

# Make a searcher object
searcher = Poisson_Partition_Searcher(assignment, #actual assignment (used for simulation tests can be np.zeros(n))
                            Y, #response observations numpy array of shape (n,m)
                            X, #covariate observations numpy array of shape (n,m,d)
                            Z,
                            nx.from_numpy_array(W_mat), #Graph of adjacency matrix (must be fully connected)
                            C,
                            L=L,
                            Kmeans_initliaise=1, #Initialise with k-means
                            mu =mu,
                            lambda_=lambda_,
                            k_min = 2,k_max = 4,
                            k_0=k_0,rho=rho,
                            alpha=tau_prior_1,beta=tau_prior_2
                            ,prior_alpha=prior_1,prior_theta=prior_2,
                            z_start=0, #initialise with destroy/repair moves
                            start_1 = starting_point)
# Create the particle set according to initialisation choice
searcher.instantiate_set()

# Find the optimal particle set.  both equal to 1 gives full neighbourhood.
searcher.optimise_PARTOPT(global_m=1)



