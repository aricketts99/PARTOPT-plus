# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:14:43 2024

@author: Andrew
"""



import time
import numpy as np
import networkx as nx
from numpy.linalg import slogdet
from scipy.linalg import block_diag, inv,det
from scipy import linalg
from scipy.stats import multivariate_normal
import scipy.special as sc
from partopt_plus.connected_Particle import connected_Particle


class Particle_MCAR(connected_Particle):
    def __init__(self,assignment,y,X,W,prior="EP",k_0=0.1,mu=1,nu=1,rho=0.9,prior_alpha = 0.1,prior_theta = 0.1,lambda_=1):
        super().__init__(assignment,y,X,W,prior,k_0,mu,nu,rho,prior_alpha,prior_theta,lambda_)

    
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
        sign, logabsdet = slogdet(mat)

        #print("Det Time: "+str(time.time()-start))
        return logabsdet
    
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
    
    def quad_term_2(self, Y, X, Binv, lambda_):
        """
        Computes
            Y^T (I + lambda X B X^T)^(-1) Y
        using the Woodbury identity.
        """
    
        # Build the block-diagonal design matrix
        Xblk = block_diag(*X)
        
        middle = Binv / lambda_ + Xblk.T @ Xblk
    
        rhs = Xblk.T @ Y
    
        z = linalg.solve(middle, rhs, assume_a="pos")
    
        quad_term_value = Y.T @ Y - rhs.T @ z

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


        return B,B_inv,np.kron((Sigma_mat),np.eye(self.dim_cov))
    
    
    def calc_like_given_clus(self,changed_clusters,previous_likes,lambda_,tau):
        matrices = []
        Ys = []
        for cluster in range(self.K):
            indicies = [i for i, x in enumerate(self.assignment) if x == cluster]
            X_k = self.X[indicies,:]
            X_k = block_diag(*X_k).astype("float32")
            B_k,_,_ = self.gen_B(indicies)
            cov = lambda_*X_k@B_k@X_k.T
            cov = np.eye(cov.shape[0])+cov
            matrices.append(cov)
            Y_k = self.y[indicies].reshape(self.m*len(indicies))
            Ys = Ys+Y_k.tolist()
        cov = block_diag(*matrices)
        var = multivariate_normal(mean=np.zeros(len(Ys)), cov=tau*cov)
        result = var.logpdf(Ys)
        return result
    
    # def calculate_likelihood_given_clustering(self,changed_clusters,previous_likes,lambda_):
    #     def integral(tau):
    #         return self.calc_like_given_clus(changed_clusters,previous_likes,lambda_,tau)
    #     return integrate.quad(integral,0,10)[0]
    
    def calculate_likelihood_given_clustering(self,changed_clusters,previous_likes,lambda_):
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
        start = time.time()
        determinant = 0
        quad = 0
        det_terms = []
        quad_terms = []
        for cluster in changed_clusters:
            indicies = [i for i, x in enumerate(self.assignment) if x == cluster]
            X_k = self.X[indicies,:]
            B_k,B_k_inv,_ = self.gen_B(indicies)
            Y_k = self.y[indicies].reshape(self.m*len(indicies))
            det_term = self.compute_det(X_k,B_k,lambda_)
            #quad_term = self.quad_term(Y_k,X_k,B_k,lambda_)
            quad_term = self.quad_term_2(Y_k,X_k,B_k_inv,lambda_)
            #print('Quad term diff: '+str(quad_term-quad_term_2))
            det_terms.append(det_term)
            quad_terms.append(quad_term)
            determinant+=det_term
            quad+=quad_term
            
        
        changed_terms = list(zip(changed_clusters,list(zip(det_terms,quad_terms))))
        
        unchanged = list(set(range(self.K))-set(changed_clusters))
        unchanged_likes = [previous_likes[i] for i in unchanged]
        for like in unchanged_likes:
            determinant += like[0]
            quad += like[1]
        unchanged_terms = list(zip(unchanged,unchanged_likes))
        
        terms = changed_terms+unchanged_terms
        terms.sort()
        terms = [t[1] for t in terms]
        #print("Quad: "+str(quad))
        #print("Transformed Quad: "+str(-0.5*(self.n_obs*self.m+2*self.mu)*np.log((1+quad/(2*self.nu)))))
        #print("Det: "+str(-0.5*(determinant)))
        # print("n: "+str(self.n_obs))
        # print("m: "+str(self.m))
        # print("mu: "+str(self.mu))
        # print("nu: "+str(self.nu))
        #input()
        log_likelihood = -0.5*(determinant)-0.5*(self.n_obs*self.m+2*self.mu)*np.log((1+quad/(2*self.nu)))+sc.loggamma(self.mu+self.n_obs*self.m/2)-sc.loggamma(self.mu)-(self.n_obs*self.m/2)*(np.log(2*self.mu)+np.log(np.pi))
        
        #print("Determinant: "+str(determinant))
        #print("Quad: "+str(quad))
        #print("Second term: "+str(np.log((1+quad/(2*self.nu)))))
        #print("Constant: "+str(-0.5*(self.n_obs*self.m+2*self.mu)))
        #input()
        
        self.likelihood = log_likelihood
        
        self.terms = terms
        #print("Time 1: "+str(time.time()-start))
        
        return

    
    
    def calculate_likelihood_given_clustering_2(self,changed_clusters,previous_likes,lambda_):
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
        start = time.time()
        determinant = 0
        quad = 0
        det_terms = []
        quad_terms = []
        for cluster in changed_clusters:
            indicies = [i for i, x in enumerate(self.assignment) if x == cluster]
            X_k = self.X[indicies,:]
            _,_,S_k = self.gen_B(indicies)
            S_k = S_k*lambda_
            one = np.kron(np.ones((len(indicies),1)),np.eye(self.dim_cov))
            A_k = (self.k_0/lambda_)*np.eye(self.dim_cov)+one.T@S_k@one
            xs = block_diag(*X_k).astype("float32")
            mat = xs.T@xs
            B_k = mat+S_k
            B_k = B_k-S_k@one@inv(A_k)@one.T@S_k.T
            Y_k = self.y[indicies].reshape(self.m*len(indicies))
            det_term = np.log(det(S_k))+np.log(det(A_k))+np.log(det(B_k))
            #print("DET: "+str(det_term))
            #det_term = self.compute_det(X_k,B_k,lambda_)
            quad_term = self.quad_term(Y_k,X_k,B_k,lambda_)
            det_terms.append(det_term)
            quad_terms.append(quad_term)
            determinant+=det_term
            quad+=quad_term
        
        changed_terms = list(zip(changed_clusters,list(zip(det_terms,quad_terms))))
        
        unchanged = list(set(range(self.K))-set(changed_clusters))
        unchanged_likes = [previous_likes[i] for i in unchanged]
        for like in unchanged_likes:
            determinant += like[0]
            quad += like[1]
        unchanged_terms = list(zip(unchanged,unchanged_likes))
        
        terms = changed_terms+unchanged_terms
        terms.sort()
        terms = [t[1] for t in terms]
        # print("Quad: "+str(quad))
        # print("Transformed Quad: "+str(-0.5*(self.n_obs*self.m+2*self.mu)*np.log((1+quad/(2*self.nu)))))
        # print("Det: "+str(-0.5*(determinant)))
        # print("n: "+str(self.n_obs))
        # print("m: "+str(self.m))
        # print("mu: "+str(self.mu))
        # print("nu: "+str(self.nu))
        #input()
        log_likelihood = -0.5*(determinant)-0.5*(self.n_obs*self.m+2*self.mu)*np.log(2*self.nu+quad)-self.dim_cov*self.K*np.log(lambda_/self.k_0)
        
        
        #print("Determinant: "+str(determinant))
        #print("Quad: "+str(quad))
        #print("Second term: "+str(np.log((1+quad/(2*self.nu)))))
        #print("Constant: "+str(-0.5*(self.n_obs*self.m+2*self.mu)))
        #input()
        
        self.likelihood = log_likelihood
        
        self.terms = terms
        #print("Time 1: "+str(time.time()-start))
        return
    
    def point_estimate_theta_and_theta_k(self):
        '''
        Procedure for calculating theta and theta_k for heuristics.  Updates properties
        instead of returning.

        Returns
        -------
        None.

        '''
        start = time.time()
        #Calculate Gram Matrix
        gram = self.calc_gram(self.X)
        theta = []
        theta_k = []
        for k in range(self.K):
            
            
            #Get current cluster indicies
            indicies = [i for i, x in enumerate(self.assignment) if x == k]
            
            #Get cluster data and calculate the B_k matrix
            X_k = self.X[indicies,:]
            Y_k = self.y[indicies].reshape(self.m*len(indicies))
            B_k,B_k_inv,_ = self.gen_B(indicies)
            gram_k = block_diag(*[gram[i] for i in indicies])
            
            #For theta_k we do not use B_k use Sigma_k
            W = nx.adjacency_matrix(self.W).toarray()[indicies,:][:,indicies]
            n_k = W.shape[0]
            F = [1-self.rho+self.rho*np.sum(W[i,:]) for i in range(n_k)]
            Sigma_mat = np.diag(F)-self.rho*W
            
            #Compute second matrix (I+lambda*X_k@Sigma_k@X_k^T) and invert it
            mat_2 = self.lambda_*block_diag(*X_k)@np.kron(Sigma_mat,np.eye(self.dim_cov))@block_diag(*X_k).T
            mat_2 = mat_2+np.eye(mat_2.shape[0])

            
            #Calculate expansion matrix
            I_B = np.kron(np.ones((n_k,1)),np.eye(self.dim_cov))
            
            #Compute end term vectors
            end = block_diag(*X_k)@I_B
            
            x_1 = linalg.solve(mat_2,Y_k)
            x_2 = linalg.solve(mat_2,end)
            
            #Compute theta_k by combining results, there is only on theta_k per clutser
            mat_3 = end.T@x_2@end.T@x_1
            
            theta_k.append(mat_3)
            
            #Calculate first matrix used in expectations that are used (Gram_k+lambda^-1*inv(B_k))
            mat = gram_k+B_k_inv*self.lambda_**(-1)

            
            
            #Calculate ending vector X^T_k@Y_k
            end = block_diag(*X_k).T@Y_k
            x_3 = linalg.solve(mat,end)
            
            #Group up terms to get the theta values of shape dim_cov
            for thet in (x_3).reshape(len(indicies),self.dim_cov):
                theta.append(thet)

        
        self.theta_k = np.array(theta_k).reshape(self.K,self.dim_cov)
        
        
        
        #Re-order theta since currently it is ordered by cluster then index instead of just index
        indicies = []
        for k in range(self.K):
            indicies = indicies + [i for i, x in enumerate(self.assignment) if x == k]

        to_reorder = zip(indicies,theta)
        self.theta = np.array([x for _, x in sorted(to_reorder)])        
        #print("Time 2: "+str(time.time()-start))
        return
    
    def det_block(self,blocks):
        if len(blocks) == 1:
            return det(np.eye(self.dim_cov)+self.lambda_*(blocks[0][0]*np.eye(self.dim_cov)+self.k_0**(-1)*blocks[0][2]*np.eye(self.dim_cov))@self.X[blocks[0][1]].T@self.X[blocks[0][1]])
        else:
            return 

