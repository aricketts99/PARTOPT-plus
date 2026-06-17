#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:40:31 2024

@author: andrew
"""

import numpy as np
import networkx as nx
from scipy import linalg
def pinvh(A):
    if A.shape == (1,1):
        return np.reciprocal(A)
    else:
        return linalg.pinvh(A)

class Particle:
    def __init__(self,assignment,y,X,W,prior="EP",k_0=0.1,mu=1,nu=1,rho=0.9,prior_alpha = 0.1,prior_theta = 0.1,lambda_=1):
        '''
        Particle takes an assignment (list of labels) and can then calculate the liklihood,
        estimate paramaters, and propose changes to the assignment.

        Parameters
        ----------
        assignment : LIST
            list of numerical labels, must not have gaps, i.e [0,1,2] is fine [0,3,5] is not
        y : NUMPY ARRAY dimension (number of area units,number of observation per unit)
            multi-dimensional observations
        X : NUMPY ARRAY dimension (number of area units,number of observations,number of covariates)
            multi-dimensional covariates for each observation in each area unit
        W : NETWROKX GRAPH
            graph object of adjacency matrix
        prior : STRING, optional
            String indiciating prior on clustering. The default is "EP".
        k_0 : FLOAT>1, optional
            k_0 term in variance of MCAR. The default is 0.1.
        mu : FLOAT>0, optional
            Scale parameter of Inverse Gamma Prior. The default is 1.
        nu : FLOAT>0, optional
            Shape parameter of Invser Gamma Prior. The default is 1.
        rho : FLOAT>0, optional
            Weighting for MCAR prior. The default is 1.

        Returns
        -------
        None.

        '''
        self.assignment = assignment
        self.n_obs = self.assignment.size
        #print(self.n_obs)
        self.K = np.unique(assignment).size
        self.prior = prior
        self.objective = 1      
        self.X = X
        self.y = y
        self.m = self.y.shape[1]
        #print(self.m)
        self.k_0 = k_0
        self.dim_cov = self.X.shape[2]
        #print(self.dim_cov)
        self.W = W
        self.W_mat = nx.adjacency_matrix(W).toarray()
        self.theta = np.zeros((self.n_obs,self.dim_cov))
        self.theta_k = np.zeros((self.K,self.dim_cov))
        self.mu = float(mu)
        self.nu = float(nu)
        self.rho = rho
        self.prior_theta = prior_theta
        self.prior_alpha = prior_alpha
        self.lambda_ = lambda_


    def pairwise_assignment(self):
        '''
        Method to compare which points are together in a cluster.

        Returns
        -------
        None.

        '''
        # matrix = []
        # for i in range(self.n_obs):
        #     paired = []
        #     for j in range(self.n_obs):
        #         if self.assignment[i] == self.assignment[j]:
        #             paired.append(1)
        #         else:
        #             paired.append(0)
        #     matrix.append(paired)
        # matrix = np.array(matrix)
        matrix = np.equal.outer(self.assignment,self.assignment)
        self.pairwise_matrix_assignment = matrix
        matrix = matrix[np.triu_indices_from(matrix, k=1)]
        self.pairwise_matrix_assignment_hash = hash(tuple(matrix))

