#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 12:38:28 2025

@author: andrew
"""


import time
import numpy as np
import networkx as nx
from scipy.linalg import block_diag, inv,det,eigh
from scipy import linalg,integrate
from scipy.stats import multivariate_normal,multivariate_t,poisson
from scipy import linalg
import scipy.special as sc
from connected_Particle import connected_Particle
from Particle import pinvh
from scipy import optimize
from joblib import Parallel, delayed
import statsmodels.api as sm



def callback(*, intermediate_result):
    print(intermediate_result.fun)
    print(intermediate_result.x)

class Particle_Poisson(connected_Particle):
    def __init__(self,assignment,y,X,Z,W,C,prior="EP",k_0=0.1,mu=1,nu=2,rho=0.95,prior_alpha = 0.1,prior_theta = 0.1,lambda_=1,alpha=1,beta=1):
        super().__init__(assignment,y,X,W,prior,k_0,mu,nu,rho,prior_alpha,prior_theta,lambda_)
        self.C = C
        self.Z = Z
        self.beta_dim_cov = self.Z.shape[2]
        self.alpha = float(alpha)
        self.beta = float(beta)

    
    def calculate_prior(self):
        '''
        Calculates the prior value.

        Returns
        -------
        log_prior : FLOAT
            Logarithm of the prior.

        '''
        log_prior  = -sc.gammaln(self.prior_alpha+self.n_obs)+sc.gammaln(self.prior_alpha)
        for cluster in range(self.K):
            indicies = [i for i, x in enumerate(self.assignment) if x == cluster]
            n_k = len(indicies)
            log_prior  += sc.gammaln(n_k-self.prior_theta)-sc.gammaln(1-self.prior_theta)+np.log(self.prior_alpha+cluster*self.prior_theta)
        return log_prior
    
    def calc_gram(self,X):
        '''
        Calculates the gram matrix values for all of the X data in its input form since
        the gram of the block diagonal matrix is the individual gram matrices of each block.

        Parameters
        ----------
        X : NUMPY ARRAY
            X data as inputed with shape (n_obs,m,dim_cov).

        Returns
        -------
        gram : LIST OF NUMPY ARRAYS
            gram is a list of matrices each the gram of their individual block.  For example
            element 0 is the gram matrix of X1^TX1 for area 1.

        '''
        gram = []
        #gram_inv = []
        for x in X:
            gram.append((x.T@x))
            #gram_inv.append(inv(gram[-1]))
        return gram
    
    def compute_det(self,X,B,lambda_):
        '''
        Computes the determinant term in the likelihood.  

        Parameters
        ----------
        X : NUMPY ARRAY
            Numpy array of shape (n_k,m,dim_cov).
        B : NUMPY ARRAY
            Matrix of the B_k term.
        lambda_ : FLOAT
            lambda value for the current iteration.

        Returns
        -------
        log_of_determinant: FLOAT
            The determinant value in log space.

        '''
        #gram = self.calc_gram(X)
        #gram = block_diag(*gram)
        #mat = (gram@B)*lambda_
        start = time.time()
        Xs = block_diag(*X)
        mat = (lambda_*B@Xs.T@Xs)
        mat = mat+np.eye(mat.shape[0])
        log_of_determinant = np.log(det(mat))

        ####print("Det Time: "+str(time.time()-start))
        return log_of_determinant
    
    def fprime(self,vector):
        tic_1 = time.time()
        tic = time.time()
        order = np.array(sorted(list(zip(self.assignment,range(self.n_obs))),key=lambda x: x[0])).T[1]
        flip = np.array(sorted(list(zip(order,range(self.n_obs))),key=lambda x: x[0])).T[1]
        theta = vector[:self.n_obs*self.dim_cov].reshape(self.n_obs,self.dim_cov)
        beta = vector[self.n_obs*self.dim_cov:]
        
        ####print(theta)
        ####print(beta)
        
        #Set up Q matrix
        sum_1 = 0
        vec_1 = []
        vec_2 = []
        for k in range(self.K):
            indicies = np.where(self.assignment==k)[0]
            sum_1 += theta[indicies].flatten().T@self.Q_inv[k]@theta[indicies].flatten()
            vec_1.append(self.Q_inv[k]@theta[indicies].flatten())
            tic_Q = time.time()
            for i in indicies:
                sum_2 = 0
                for m in range(self.m):
                    ##print('Example: '+str(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i]))
                    sum_2 += np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])*self.X[i,m]
                vec_2.append(sum_2)
            toc_Q = time.time()
            #print('Q time: '+str(toc_Q-tic_Q))
        toc = time.time()
        
        # vec_2 = []
        # for i in range(self.n_obs):
        #     sum_2 = 0
        #     for m in range(self.m):
        #         sum_2 += np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])*self.X[i,m]
        #     vec_2.append(sum_2)
        #print('Fprime time 1: '+str(toc-tic))
        vec_1 = np.concat(vec_1)
        vec_2 = np.concat(vec_2)
        tic = time.time()
        sum_3 = 0
        for k in range(self.K):
            indicies = np.where(self.assignment==k)[0]
            for i in indicies:
                ##print('Example 2: '+str(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i]))
                for m in range(self.m):
                    sum_3 += np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])*self.Z[i,m]
        vec_3 = self.y[order].flatten().T@block_diag(*self.X[order])-vec_2#-(2*self.alpha+self.n_obs*self.dim_cov)*0.5*np.power(self.beta,-1)*np.power((1+0.5*np.power(self.beta,-1)*sum_1),-1)*vec_1
        vec_3 = vec_3.reshape(self.n_obs,self.dim_cov)[flip].flatten()
        vec_4 = self.y[order].flatten()@self.Z[order].reshape(self.n_obs*self.m,self.beta_dim_cov)-sum_3-(2*self.mu+self.beta_dim_cov)*0.5*np.power(self.nu,-1)*np.power((1+0.5*np.power(self.nu,-1)*beta@beta),-1)*beta
        toc = time.time()
        ##print('Fprime time 2: '+str(toc-tic))
        ###print(self.y[order].flatten().T@block_diag(*self.X[order]))
        ###print(self.y[order].flatten()@self.Z[order].reshape(self.n_obs*self.m,self.beta_dim_cov))
        
        ###print((2*self.alpha+self.n_obs*self.dim_cov)*0.5)
        ###print((2*self.mu+self.beta_dim_cov)*0.5)
        
        ###print(vec_1)
        ###print(beta)
        
        ###print(np.power(self.beta,-1)*np.power((1+0.5*np.power(self.beta,-1)*sum_1),-1))
        ###print(np.power(self.nu,-1)*np.power((1+0.5*np.power(self.nu,-1)*beta@beta),-1))
        
        ###print(vec_2)
        ###print(sum_3)
        toc = time.time()
        #print('Fprime time: '+str(toc-tic_1))
        return -1*np.concat([vec_3,vec_4])/(self.n_obs*self.m)
    
    def fdoubleprime(self,vector):
        tic_1 = time.time()
        tic = time.time()
        order = np.array(sorted(list(zip(self.assignment,range(self.n_obs))),key=lambda x: x[0])).T[1]
        flip = sorted(list(zip(order,np.array(range(self.n_obs*self.dim_cov)).reshape(self.n_obs,self.dim_cov))),key=lambda x: x[0])
        flip = np.array([i[1] for i in flip]).flatten()
        theta = vector[:self.n_obs*self.dim_cov].reshape(self.n_obs,self.dim_cov)
        beta = vector[self.n_obs*self.dim_cov:]
        ###print(theta)
        #Set up Q matrix
        sum_1 = 0
        vec_1 = []
        vec_2 = []
        Q = []
        for k in range(self.K):
            indicies = np.where(self.assignment==k)[0]
            sum_1 += theta[indicies].flatten().T@self.Q_inv[k]@theta[indicies].flatten()
            ###print('Sum_1: '+str(sum_1))
            vec_1.append(self.Q_inv[k]@theta[indicies].flatten())
            for i in indicies:
                sum_2 = 0
                for m in range(self.m):
                    sum_2 += np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])*self.X[i,m]
                vec_2.append(sum_2)
        vec_1 = np.concat(vec_1)
        vec_2 = np.concat(vec_2)
        toc = time.time()
        ##print('Fdouble time 1: '+str(toc-tic))
        tic = time.time()
        ###print(vec_1)
        ###print(vec_2)
        ###print('Sum_1: '+str(sum_1))
        
        
        M = []
        for i in order:
            M_i = np.zeros((self.dim_cov,self.dim_cov))
            for d in range(self.dim_cov):
                for c in range(self.dim_cov):
                    sum_3 = 0
                    for m in range(self.m):
                        sum_3 += np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])*self.X[i,m,d]*self.X[i,m,c]
                        ###print(np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i]))
                        ###print(self.X[i,m,d]*self.X[i,m,c])
                    M_i[d,c] = sum_3
            M.append(M_i)
            ###print(M_i)
        M = block_diag(*M)
        toc = time.time()
        ##print('Fdouble time 2: '+str(toc-tic))
        tic = time.time()
        ###print(M)
        ###print('Sum_1: '+str(sum_1))
        quad = (1+0.5*np.power(self.beta,-1)*sum_1)
        to_outer = block_diag(*self.Q_inv)@theta[order].flatten()
        constant_1 = -0.5*(2*self.alpha+self.n_obs*self.dim_cov)*np.power(quad,-2)
        
        ###print(to_outer)
        ###print(quad)
        ###print(block_diag(*Q))
        ###print(block_diag(*Q)*quad)
        matrix = (np.power(self.beta,-1)*block_diag(*self.Q_inv)*quad-np.power(self.beta,-2)*np.outer(to_outer,to_outer))
        
        M = -M#constant_1*matrix-M
        ##print(M)
        
        sum_4 = 0
        for i in range(self.n_obs):
            for m in range(self.m):
                sum_4 -= np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])*np.outer(self.Z[i,m],self.Z[i,m])
        quad_2 = (1+0.5*np.power(self.nu,-1)*beta@beta)
        constant_2 = -0.5*(2*self.mu+self.beta_dim_cov)*np.power(quad_2,-2)
        
        matrix = np.power(self.nu,-1)*np.eye(self.beta_dim_cov)*quad_2-np.power(self.nu,-2)*np.outer(beta,beta)
        sum_4 += matrix*constant_2
        toc = time.time()
        ##print('Fdouble time 3: '+str(toc-tic))
        tic = time.time()
        off = []
        for i in order:
            sum_5 = 0
            for m in range(self.m):
                sum_5 -= np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])*np.outer(self.X[i,m],self.Z[i,m])
            off.append(sum_5)
        off = np.concatenate(off)[flip]
        
        hessian = np.block([[M[:,flip][flip,:],off],[off.T,sum_4]])
        toc = time.time()
        ##print('Fdouble time 4: '+str(toc-tic))
        toc = time.time()
        ##print('Fdouble time: '+str(toc-tic_1))
        return -1*hessian/(self.n_obs*self.m)
    
    def quad_term(self,Y,X,B,lambda_):
        '''
        Computes the quadratic term in the  multivariate t distribution for each cluster.

        Parameters
        ----------
        Y : NUMPY ARRAY
            Numpy array of the Y values for the current cluster.
        X : NUMPY ARRAY
            Numpy array of the X values for the current cluster shape (n_k,m,dim_cov).
        B : NUMPY ARRAY
            B_k matrix for the current cluster.
        lambda_ : FLOAT
            lambda value for current iteration.

        Returns
        -------
        quad_term_value : FLOAT
            The quadratic term in the likelihood cluster wise.

        '''
        
        mat = block_diag(*X)
        
        mat = (mat@B@mat.T)
        
        mat = lambda_*mat+np.eye(mat.shape[0])
        

        
        
        x = linalg.solve(mat,Y,assume_a="pos")

        
        quad_term_value = Y.T@x

        
        return quad_term_value
    
    def gen_B(self,indicies):
        '''
        

        Parameters
        ----------
        indicies : LIST
            list of indicies for the current cluster.

        Returns
        -------
        B : NUMPY ARRAY
            Matrix B_k for current cluster of shape (d*n_k,d*n_k).
        B_inv : NUMPY ARRAY
            Inverse of B.

        '''
        start = time.time()
        W = self.W_mat[indicies,:][:,indicies]
        rho = self.rho
        k_0 = self.k_0
        d = self.dim_cov
        n_k = W.shape[0]
        F = [1-rho+rho*np.sum(W[i,:]) for i in range(n_k)]
        Sigma_mat = np.diag(F)-rho*W

        one_B = np.kron(np.ones((n_k,1)),np.eye(self.dim_cov))
        B = np.kron(inv(Sigma_mat),np.eye(d))+(k_0**(-1))*one_B@np.eye(self.dim_cov)@one_B.T
        #B = np.kron(B,np.eye(d)).astype("float32")
        row_sum = np.sum(Sigma_mat,axis=1)
        row_sum_sum = np.sum(row_sum)
        col_sum = np.sum(Sigma_mat,axis=0)
        B_inv = -(np.power(k_0,-1)/(1+np.power(k_0,-1)*row_sum_sum))*(Sigma_mat@np.ones((n_k,n_k))@Sigma_mat)
        B_inv = Sigma_mat+B_inv
        B_inv = np.kron(B_inv,np.eye(d))
        
        # B_inv = np.kron(Sigma_mat,np.eye(d))
        # constant = 1+np.power(k_0,-1)*(one_B.T@B_inv@one_B)
        # B_inv = B_inv - np.power(constant,-1)*(B_inv@one_B@one_B.T@B_inv)*(np.power(k_0,-1))


        return B,B_inv#,np.kron((Sigma_mat),np.eye(self.dim_cov))

    
    def calculate_likelihood_given_clustering_thetabeta(self,thetabeta,print_it=0):
        '''
        Procedure for calculating the log likelihood.  Does not return but updates particle 
        properties with both the log likelihood and the individual cluster terms involved in
        the calculation for future use.

        Parameters
        ----------
        changed_clusters : LIST
            List of clusters that need to be computed since they have changed.  Otherwise the
            values from previous_likes will be used to pluggin.
        previous_likes : LIST OF TUPLES
            List of tuples whose first elements are the determinant term for each cluster and the
            second elements are the quadratic terms for each cluster.
        lambda_ : FLOAT
            Lambda value for current iteration.

        Returns
        -------
        None.

        '''
        tic = time.time()
        order = np.array(sorted(list(zip(self.assignment,range(self.n_obs))),key=lambda x: x[0])).T[1]
        theta = thetabeta[:,:self.n_obs*self.dim_cov].reshape(thetabeta.shape[0],self.n_obs,self.dim_cov)
        beta = thetabeta[:,self.n_obs*self.dim_cov:]
        marg_prior_theta = multivariate_t(loc=np.zeros(self.n_obs*self.dim_cov),shape=(self.beta/self.alpha)*block_diag(*self.Q_inv),df=2*self.alpha)
        marg_prior_beta = multivariate_t(loc=np.zeros(self.beta_dim_cov),shape=(self.nu/self.mu)*np.eye(self.beta_dim_cov),df=2*self.mu)
        likelihood = 0
        likelihood += marg_prior_theta.logpdf(theta[:,order].reshape(thetabeta.shape[0],self.n_obs*self.dim_cov))+marg_prior_beta.logpdf(beta)
        #print('Prior theta: '+str(marg_prior_theta.logpdf(theta[:,order].reshape(thetabeta.shape[0],self.n_obs*self.dim_cov))))
        #likelihood += marg_prior_theta.logpdf(theta[:,order].reshape(thetabeta.shape[0],self.n_obs*self.dim_cov))
        # for i in range(self.n_obs):
        #     for m in range(self.m):
        #         lmbda = np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])
        #         rv = poisson(lmbda)
        #         likelihood += rv.logpmf(self.y[i,m])
        lmbda_1 = np.tile(self.C,(thetabeta.shape[0],1,1))
        lmbda_2 = np.matvec(self.Z.reshape(self.n_obs*self.m,self.beta_dim_cov),beta).reshape(thetabeta.shape[0],self.n_obs,self.m)
        lmbda_3 = np.matvec(block_diag(*self.X),theta.reshape(thetabeta.shape[0],self.n_obs*self.dim_cov)).reshape(thetabeta.shape[0],self.n_obs,self.m)
        lmbda = np.exp(lmbda_1+lmbda_2+lmbda_3)
        rv = poisson(lmbda)
        like_2 = rv.logpmf(np.tile(self.y,(thetabeta.shape[0],1,1)))#.reshape(thetabeta.shape[0],self.n_obs*self.m))
        likelihood += np.sum(like_2.reshape(thetabeta.shape[0],self.n_obs*self.m),axis=1)
        if print_it:
            print('WRONG FUNCTION')
            # print('Order: '+str(order))
            # print('Prior beta: '+str(marg_prior_beta.logpdf(beta)))
            # print('Prior theta: '+str(marg_prior_theta.logpdf(theta[:,order].reshape(thetabeta.shape[0],self.n_obs*self.dim_cov))))
            # print('Poisson Likelihood: '+str(np.sum(like_2.reshape(thetabeta.shape[0],self.n_obs*self.m),axis=1)))
        #hessian = self.fdoubleprime(np.array(self.theta.tolist()+self.beta.tolist()))
        #_, logabsdet = np.linalg.slogdet(hessian)
        #*(1-0.5*hessian.shape[0])-0.5*logabsdet
        toc = time.time()
        #print('Likelihood time: '+str(toc - tic))
        return likelihood
    
    def wrapped(self,thetabeta):
        return -1*self.calculate_likelihood_given_clustering_thetabeta_unvec(thetabeta)#self.calculate_likelihood_given_clustering_thetabeta(np.array([thetabeta]))[0]/(self.n_obs*self.m)
    
    def calculate_likelihood_given_clustering_thetabeta_unvec(self,thetabeta,print_it=False):
        '''
        Procedure for calculating the log likelihood.  Does not return but updates particle 
        properties with both the log likelihood and the individual cluster terms involved in
        the calculation for future use.

        Parameters
        ----------
        changed_clusters : LIST
            List of clusters that need to be computed since they have changed.  Otherwise the
            values from previous_likes will be used to pluggin.
        previous_likes : LIST OF TUPLES
            List of tuples whose first elements are the determinant term for each cluster and the
            second elements are the quadratic terms for each cluster.
        lambda_ : FLOAT
            Lambda value for current iteration.

        Returns
        -------
        None.

        '''
        order = np.array(sorted(list(zip(self.assignment,range(self.n_obs))),key=lambda x: x[0])).T[1]

        theta = thetabeta[:self.n_obs*self.dim_cov].reshape(self.n_obs,self.dim_cov)
        beta = thetabeta[self.n_obs*self.dim_cov:]
        Q_mat = self.Q
        marg_prior_theta = multivariate_t(loc=np.zeros(self.n_obs*self.dim_cov),shape=(self.beta/self.alpha)*block_diag(*Q_mat),df=2*self.alpha)
        marg_prior_beta = multivariate_t(loc=np.zeros(self.beta_dim_cov),shape=(self.nu/self.mu)*np.eye(self.beta_dim_cov),df=2*self.mu)
        #marg_prior_theta = -0.5*np.linalg.slogdet((self.alpha/self.beta)*block_diag(*self.Q))[1]-(2*self.alpha+theta.shape[0]*theta.shape[1])*0.5*np.log(1+np.power(2*self.beta,-1)*theta[order].flatten()@block_diag(*self.Q)@theta[order].flatten())
        likelihood = 0
        likelihood += marg_prior_theta.logpdf(theta[order].flatten())+marg_prior_beta.logpdf(beta)
        #likelihood = marg_prior_theta + marg_prior_beta.logpdf(beta)
        for i in range(self.n_obs):
            for m in range(self.m):
                lmbda = np.exp(self.C[i,m]+self.Z[i,m]@beta+self.X[i,m]@theta[i])
                rv = poisson(lmbda)
                likelihood += rv.logpmf(self.y[i,m])
        # if print_it:
        #     print('theta: '+str(theta))
        #     print('alpha,beta,nu,mu: '+str([self.alpha,self.beta,self.nu,self.mu]))
        #     print('Det part: '+str(-0.5*np.linalg.slogdet((self.beta/self.alpha)*block_diag(*self.Q))[1]))
        #     print('Quad part: '+str(theta[order].flatten()@block_diag(*self.Q_inv)@theta[order].flatten()))
        #     print('Non-det part: '+str(-(2*self.alpha+theta.shape[0]*theta.shape[1])*0.5*np.log(1+np.power(2*self.beta,-1)*theta[order].flatten()@block_diag(*self.Q_inv)@theta[order].flatten())))
        #     print('Order: '+str(order))
        #     print('Prior beta: '+str(marg_prior_beta.logpdf(beta)))
        #     print('Prior theta: '+str((marg_prior_theta.logpdf(theta[order].flatten()))))
        #     print('Poisson Likelihood: '+str(likelihood-marg_prior_theta.logpdf(theta[order].flatten())-marg_prior_beta.logpdf(beta)))
        #     print('Shape: '+str(block_diag(*Q_mat).shape))

        #hessian = self.fdoubleprime(np.array(self.theta.tolist()+self.beta.tolist()))
        #_, logabsdet = np.linalg.slogdet(hessian)
        likelihood#*(1-0.5*hessian.shape[0])-0.5*logabsdet
        return likelihood
    
    def point_estimate_theta_and_theta_k(self,starting_val=None,start_1=True):
        '''
        Procedure for calculating theta and theta_k for heuristics.  Updates properties
        instead of returning.

        Returns
        -------
        None.

        '''
        self.Q = []
        self.Q_inv = []
        order = []
        Xs = []
        for k in range(self.K):
            indicies = np.where(self.assignment==k)[0]
            order = order+ indicies.tolist()
            Xs.append(self.X[indicies].reshape(len(indicies)*self.m,self.dim_cov))
            Q_k,Q_inv = self.gen_B(indicies)
            self.Q.append(Q_k)
            self.Q_inv.append(Q_inv)
            
        # model_no_indicators = sm.GLM(
        #     self.y.flatten(),
        #     self.Z.reshape(self.n_obs*self.m,self.beta_dim_cov),
        #     offset=self.C.flatten(),
        #     family=sm.families.Poisson(),
        # )
        # results = model_no_indicators.fit()

        # model_indicators = sm.GLM(
        #     self.y.flatten(),
        #     block_diag(*self.X),
        #     offset=self.Z.reshape(self.n_obs*self.m,self.beta_dim_cov)@results.params+self.C.flatten(),
        #     family=sm.families.Poisson(),
        # )
        # coeffs = model_indicators.fit()
        # start_2 = np.concat([coeffs.params,results.params])
        #print('Start: '+str(start_2))
        
        if start_1:
            
            model_no_indicators = sm.GLM(
                self.y.flatten(),
                np.hstack([block_diag(*self.X),self.Z.reshape(self.n_obs*self.m,self.beta_dim_cov)]),
                offset=self.C.flatten(),
                family=sm.families.Poisson(),
            )
            
            results = model_no_indicators.fit()
            start = results.params
            
            # print('Params shape: '+str(results.params.shape))
            #print('Start 1: '+str(repr(results.params)))
            self.start_1 = results.params
        else:
            
            X_data = []
            for k in range(self.K):
                indicies = np.where(self.assignment==k)[0]
                X_data.append(np.vstack(self.X[indicies]))
                #print('Cluster: '+str(k))
                #print('Data shape: '+str(np.vstack(self.X[indicies]).shape))
            #print('Design matrix shape: '+str(block_diag(*X_data).shape))
            #print('Design matrix: '+str((block_diag(*X_data))))
            model_no_indicators = sm.GLM(
                self.y[order].flatten(),
                np.hstack([block_diag(*X_data),self.Z[order].reshape(self.n_obs*self.m,self.beta_dim_cov)]),
                offset=self.C[order].flatten(),
                family=sm.families.Poisson(),
            )
            
            results = model_no_indicators.fit()
            #print('Cluster means: '+str(repr(results.params[:-self.beta_dim_cov].reshape(self.K,self.dim_cov))))
    
            start = np.zeros((self.n_obs,self.dim_cov))
            betas = np.array(results.params[-self.beta_dim_cov:])
            thetas = np.array(results.params[:self.dim_cov*self.K]).reshape(self.K,self.dim_cov)
            
            #print(betas)
            #print(thetas)
            for k in range(self.K):
                indicies = np.where(self.assignment==k)[0]
                start[indicies] = thetas[k]
    
            start = np.concat([start.flatten(),betas])
            # print('Start 2: '+str(repr(start)))
            
            
            
            
            #print('Start 2: '+str(repr(start)))
            self.start_2 = start
            #print(start)

        
        
        if starting_val is not None:
            start = starting_val
        # x0 = np.random.normal(0, .003, size=(10,self.n_obs*self.dim_cov+self.beta_dim_cov))+start
        # x0 = [start]
        #x0 = np.array([0.5*(start*avg+(1-avg)*start_2) for avg in np.linspace(0,1,num=10)])
        # start = np.zeros(self.n_obs*self.dim_cov+self.beta_dim_cov)
        # # x0 = [start]
        # # tic = time.time()
        # # sol = Parallel(n_jobs=-1)(delayed(optimize.fsolve)(self.fprime, x0=p,fprime=self.fdoubleprime, full_output=True) for p in x0)
        # # thetabetas = np.array([p[0] for p in sol])
        # # func = self.calculate_likelihood_given_clustering_thetabeta(thetabetas)
        # # print(max(func))
        # # thetabeta = sol[max(range(len(func)), key=func.__getitem__)][0]
        #sol_2 = Parallel(n_jobs=-1)(delayed(optimize.minimize)(self.wrapped,jac=self.fprime, hess=self.fdoubleprime,x0=p,method='Newton-CG',options={'xtol':1e-5}) for p in x0)
        
        # # sol_4 = Parallel(n_jobs=-1)(delayed(optimize.minimize)(self.wrapped,jac=self.fprime, hess=self.fdoubleprime,x0=p,method='dogleg',callback=callback) for p in x0)
        # # sol_5 = Parallel(n_jobs=-1)(delayed(optimize.minimize)(self.wrapped,jac=self.fprime, hess=self.fdoubleprime,x0=p,method='trust-ncg',callback=callback) for p in x0)
        # # sol_6 = Parallel(n_jobs=-1)(delayed(optimize.minimize)(self.wrapped,jac=self.fprime, hess=self.fdoubleprime,x0=p,method='trust-krylov',callback=callback) for p in x0)
        # # sol_7 = Parallel(n_jobs=-1)(delayed(optimize.minimize)(self.wrapped,jac=self.fprime, hess=self.fdoubleprime,x0=p,method='trust-exact',callback=callback) for p in x0)
        
        # # sol = Parallel(n_jobs=-1)(delayed(optimize.minimize)(self.wrapped,x0=p,method='BFGS',options={'gtol':self.n_obs*self.m*1e-7,'disp':False}) for p in [x0[0]])
        
        # sol = optimize.minimize(self.wrapped,x0=start,method='BFGS',options={'disp':True})
        # thetabeta = sol['x']
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[0]))
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[1]))
        # print(repr(thetabeta[-4:]))
        # sol = optimize.minimize(self.wrapped,x0=start,method='BFGS',jac=self.fprime,options={'disp':True})
        # thetabeta = sol['x']
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[0]))
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[1]))
        # print(repr(thetabeta[-4:]))
        # sol = optimize.minimize(self.wrapped,x0=start,method='dogleg',jac=self.fprime,hess=self.fdoubleprime,options={'disp':True})
        # thetabeta = sol['x']
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[0]))
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[1]))
        # print(repr(thetabeta[-4:]))
        # sol = optimize.minimize(self.wrapped,x0=start,method='trust-ncg',jac=self.fprime,hess=self.fdoubleprime,options={'disp':True})
        # thetabeta = sol['x']
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[0]))
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[1]))
        # print(repr(thetabeta[-4:]))
        # sol = optimize.minimize(self.wrapped,x0=start,method='trust-krylov',jac=self.fprime,hess=self.fdoubleprime,options={'disp':True})
        # thetabeta = sol['x']
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[0]))
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[1]))
        # print(repr(thetabeta[-4:]))
        # sol = optimize.minimize(self.wrapped,x0=start,method='trust-exact',jac=self.fprime,hess=self.fdoubleprime,options={'disp':True})
        # thetabeta = sol['x']
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[0]))
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[1]))
        # print(repr(thetabeta[-4:]))
        # sol = Parallel(n_jobs=-1)(delayed(optimize.minimize)(self.wrapped,x0=p,method='Newton-CG',jac=self.fprime,hess=self.fdoubleprime,options={'xtol':1e-5}) for p in x0)
        # thetabeta = min(sol,key=lambda x: x['fun'])['x']
        # print( min(sol,key=lambda x: x['fun'])['success'])
        
        sol = optimize.minimize(self.wrapped,x0=start,method='Newton-CG',jac=self.fprime,hess=self.fdoubleprime,options={'xtol':1e-5})
        thetabeta = sol['x']
        # for solution in sol:
        #     print('Some funcs: '+str(solution['fun']))
        # sol = optimize.minimize(self.wrapped,x0=start,method='Newton-CG',jac=self.fprime,hess=self.fdoubleprime)
        # thetabeta = sol['x']
        # print('Func 1: '+str(sol['fun']))
        
        # sol_2 = optimize.minimize(self.wrapped,x0=start_2,method='Newton-CG',jac=self.fprime,hess=self.fdoubleprime)
        # thetabeta_2 = sol_2['x']
        # print('Func 2: '+str(sol_2['fun']))
        
        
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[0]))
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[1]))
        # print(repr(thetabeta[-4:]))
        
        #sol_2 = optimize.minimize(self.wrapped,jac=self.fprime,hess=self.fdoubleprime,x0=start,method='Newton-CG',options={'xtol':1e-25,'c1':1e-1,'c2':0.99,'disp':True},callback=callback)
        # sol_3 = Parallel(n_jobs=-1)(delayed(optimize.fsolve)(self.fprime, x0=p,fprime=self.fdoubleprime, full_output=True,xtol=1e-12) for p in x0)
        # sol_3,_,_,_ = optimize.fsolve(self.fprime,x0=start,fprime=self.fdoubleprime,full_output=True)
        # print('fsolve X: '+str(sol_3))
        # print('fsolve Func: '+str(self.wrapped(sol_3)))
        # #thetabeta = sol_2['x']
        # print('Func:' +str(sol['fun']))
        # print('Succes: '+str(sol['success']))
        # print('X:' +str(sol['x']))
        # print("X': "+str(self.fprime(sol['x'])))
        # print('Hess: '+str(self.fdoubleprime(sol['x'])))
        
        # thetabeta = sol_2['x']
        # print('Func:' +str(sol_2['fun']))
        # print('Succes: '+str(sol_2['success']))
        # print('X:' +str(sol_2['x']))
        # print("X': "+str(self.fprime(sol_2['x'])))
        # print('Hess: '+str(self.fdoubleprime(sol_2['x'])))
        
        # # thetabetas = np.array([p['x'] for p in sol_2 if p['success']==True])
        # # if thetabetas.size == 0:
        # #     thetabeta = start
        # #     print('Empty Func 3: '+str(self.wrapped(thetabeta)))
        # # else:
        # #     thetabeta = max(sol,key=lambda x: x['fun'])['x']
        # #     print('Func 3: '+str(print(max(sol,key=lambda x: x['fun']))))
        # #thetabeta = max(sol,key=lambda x: x['fun'])['x']
        # #print(max(sol,key=lambda x: x['fun']))
        # #print([(x['fun'],x['success']) for x in sol])
        #sol_2 = [solu for solu in sol_2 if solu['success']=='True']
        # thetabeta = sol['x']
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[0]))
        # print(repr(thetabeta[:-4].reshape(self.n_obs,self.dim_cov).T[1]))
        # print(repr(thetabeta[-4:]))
        # # print(max(sol_2,key=lambda x: x['fun']))
        # # print(max(sol_4,key=lambda x: x['fun']))
        # # print(max(sol_5,key=lambda x: x['fun']))
        # # print(max(sol_6,key=lambda x: x['fun']))
        # # print(max(sol_7,key=lambda x: x['fun']))

        # thetabetas = np.array([p[0] for p in sol_3 if p[2]==1])
        # if thetabetas.size == 0:
        #     thetabeta = start
        #     print('EMpty Func 2: '+str(self.wrapped(thetabeta)))
        #     print(thetabeta)
        # else:
        #     func = self.calculate_likelihood_given_clustering_thetabeta(thetabetas)
        #     thetabeta = sol_3[max(range(len(func)), key=func.__getitem__)][0]
        #     print('Func 2: '+str(max(func)))
        #     print(thetabeta)
            

        # # print(max(func))
        # # toc = time.time()
        # # #print('Opt time: '+str(toc-tic))
        
        # # tic = time.time()
        # # sol = [optimize.fsolve(self.fprime, x0=x0[0],fprime=self.fdoubleprime, full_output=True)]
        # # toc = time.time()
        # #print('Opt time 1: '+str(toc-tic))
        # #func = Parallel(n_jobs=-1)(delayed(self.calculate_likelihood_given_clustering_thetabeta)(p[0]) for p in sol)
        # # thetabetas = np.array([p[0] for p in sol])
        # # func = self.calculate_likelihood_given_clustering_thetabeta(thetabetas)
        # # thetabeta = sol[max(range(len(func)), key=func.__getitem__)][0]
        
        # #print(thetabeta)
        # #print(func)
        self.theta = thetabeta[:self.n_obs*self.dim_cov].reshape(self.n_obs,self.dim_cov)
        self.thetabeta = thetabeta
        theta_k = []
        for k in range(self.K):
            indicies = np.where(self.assignment==k)[0]
            theta_k.append(np.mean(self.theta[indicies]))
        self.theta_k = np.array(theta_k)
        return
    
    def calculate_likelihood_given_clustering(self):
        gradient = self.fprime(self.thetabeta)*(self.n_obs*self.m)
        outer_prod = np.outer(gradient,gradient)
        hessian = self.fdoubleprime(self.thetabeta)*((self.n_obs*self.m)**2)*self.wrapped(self.thetabeta)-outer_prod/(self.wrapped(self.thetabeta)*self.n_obs*self.m)
        #log_hessian = np.log(inv_hessian)
        #print(log_hessian)
        # print('Hessian of log: '+str(self.fdoubleprime(self.thetabeta)))
        # print('Wrapped: '+str(self.wrapped(self.thetabeta)))
        # print('Outer prod: '+str(outer_prod))
        # print('Gradient: '+str(gradient))
        # print('Inv of hessian: '+str(inv_hessian))
        s,logdet = np.linalg.slogdet(hessian)
        # print('Sign: '+str(s))
        # print('Log det: '+str(logdet))
        # s_2,logdet_2 = np.linalg.slogdet(hessian)
        # print('Sign: '+str(s_2))
        # print('Log det: '+str(logdet_2))
        
        # deter = np.linalg.det(log_hessian)
        # print('Det: '+str(deter))
        #print('DET: '+str(logabsdet))
        #print('LIKE: '+str(self.calculate_likelihood_given_clustering_thetabeta(np.array([self.thetabeta]))[0]))
        self.likelihood = self.calculate_likelihood_given_clustering_thetabeta_unvec(self.thetabeta,print_it=1)-0.5*logdet
        #print('Like:' +str(self.likelihood))
        return #self.calculate_likelihood_given_clustering_thetabeta(np.array([self.thetabeta]))[0]-0.5*logabsdet


