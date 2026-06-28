#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 12:01:59 2025

@author: andrew
"""

import time
import traceback
from math import comb
from collections import Counter
import numpy as np
from partopt_plus.Particle_Poisson import Particle_Poisson as Particle
import networkx as nx
from scipy import optimize
from scipy.stats import invgamma,multivariate_normal,multivariate_t
from collections import deque
from joblib import Parallel, delayed
import statsmodels.api as sm
from scipy.linalg import block_diag, inv, eigh
from sklearn import cluster
import networkx as nx
import sys


def callback(*, intermediate_result):
    print(intermediate_result.fun)
    print(intermediate_result.x)


class Poisson_Partition_Searcher():
    
    def __init__(self,actual_assignment,Y,X,Z,W,C,L=1,Kmeans_initliaise=False,mu = 1,lambda_=1,k_min = 2,k_max = 4,k_0=1,rho=0.9,alpha=1,beta=1,prior_alpha=1,prior_theta=0.2,z_start=False,low_var=True,start_1=True):
        self.Particle_set = []
        self.lambda_Particle_set = []
        self.visited = {}
        self.actual_assignment = actual_assignment
        self.Y = Y
        self.X = X
        self.W = W
        self.Z = Z
        self.C = C
        self.L = L
        self.Kmeans_initliaise = Kmeans_initliaise
        self.mu = mu
        self.lambda_ = lambda_
        self.k_min = k_min
        self.k_max = k_max
        self.k_0 = k_0
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.prior_alpha = prior_alpha
        self.prior_theta = prior_theta
        self.w = []
        self.n = self.X.shape[0]
        self.likelihood_dict = {}
        self.moves_accepted = [[] for i in range(self.L)]
        self.moves_accepted_key = [[] for i in range(self.L)]
        self.moves_rand = [[] for i in range(self.L)]
        self.potentials = []
        self.z_start = z_start
        self.low_var = low_var
        self.start_1 = start_1
        
    def hyper_param_search(self):
        model_no_indicators = sm.GLM(
            self.Y.flatten(),
            np.hstack([block_diag(*self.X),self.Z.reshape(self.n*self.Z.shape[1],self.Z.shape[2])]),
            offset=self.C.flatten(),
            family=sm.families.Poisson(),
        )
        results = model_no_indicators.fit()
        start_1 = results.params
        
        
        
        K = self.prior_alpha*np.log(self.n).astype('int')
        print(K)
        
        clustering = cluster.AgglomerativeClustering(n_clusters=K,connectivity=nx.to_numpy_array(self.W), linkage="ward").fit(start_1[:self.n*self.X.shape[2]].reshape(self.n,self.X.shape[2]))
        print(repr(clustering.labels_))
        emp_part = Particle(clustering.labels_,self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
        emp_part.point_estimate_theta_and_theta_k(start_1=self.start_1)
        start_1 = emp_part.thetabeta
        order = np.array(sorted(list(zip(clustering.labels_,range(self.n))),key=lambda x: x[0])).T[1]
        k_0s = np.linspace(1e-3, 1e-1, num=10)
        alpha_betas = []
        for k_0 in k_0s:
            Q_mat = []
            for k in range(K):
                indicies = np.where(clustering.labels_==k)[0]
                W = nx.to_numpy_array(self.W)[indicies,:][:,indicies]
                rho = self.rho
                d = self.X.shape[2]
                n_k = W.shape[0]
                F = [1-rho+rho*np.sum(W[i,:]) for i in range(n_k)]
                Sigma_mat = np.diag(F)-rho*W
    
                one_B = np.kron(np.ones((n_k,1)),np.eye(self.X.shape[2]))
                B = np.kron(inv(Sigma_mat),np.eye(d))+(k_0**(-1))*one_B@np.eye(self.X.shape[2])@one_B.T
                Q_mat.append(B)
            def obj(params):
                alpha,beta = params
                matrix = (beta/alpha)*block_diag(*Q_mat)
                E = np.linalg.eigvalsh(matrix)
                if np.linalg.cond(matrix) < 1/sys.float_info.epsilon:
                    if np.all(E > -1e-8):
                        marg_prior_theta = multivariate_t(loc=np.zeros(self.n*self.X.shape[2]),shape=matrix,df=2*alpha)
                        return -1*marg_prior_theta.logpdf(start_1[:self.n*self.X.shape[2]].reshape(self.n,self.X.shape[2])[order].flatten())
                    else:
                        return np.finfo('d').max
                else:
                    return np.finfo('d').max
            result = optimize.minimize(obj, [1,1],method='Nelder-Mead',bounds = ((0.1, 10), (0.1, 10)),callback=callback)
            alpha_betas.append([result['x'],result['fun'],k_0])
        params = min(alpha_betas, key=lambda x:x[1])
        self.alpha = params[0][0]
        self.beta = params[0][1]
        self.k_0 = params[2]
        return
    
    def instantiate_set(self,num_k=8):
        self.Particle_set = []
        if self.Kmeans_initliaise:
            random_k = np.random.randint(1,self.n,self.L)
            
            if self.low_var == True:
                particle_initial = np.zeros(self.n).astype("int")
            else:
                particle_initial = np.array(list(range(self.n))).astype('int')
            emp_part = Particle(particle_initial,self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
            emp_part.point_estimate_theta_and_theta_k(start_1=self.start_1)
            particle_initial = np.zeros(self.n).astype("int")
            emp_part.assignment = particle_initial
            emp_part.K = 1
            posts = []
            assigns = []
            particles = []
            for k in range(2,num_k):#int(np.log(100*self.n))):
                split_assign = emp_part.K_split(k, 0)
                new_emp_part = Particle(split_assign,self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                new_emp_part.point_estimate_theta_and_theta_k(start_1=self.start_1)
                new_emp_part.calculate_likelihood_given_clustering()
                post = new_emp_part.likelihood+new_emp_part.calculate_prior()
                print('Like 2: '+str(new_emp_part.likelihood))
                print('Post calc: '+str(post))
                assigns.append(split_assign)
                posts.append(post)
                particles.append(new_emp_part)
            
            posts = posts-(max(posts))
            posts = np.exp(posts)
            norm = np.nansum(posts)
            posts = np.nan_to_num(np.divide(posts,norm))
            print('HERE ARE THE POSTS: '+str(posts))
            for l in range(self.L):
                index = np.random.choice(list(range(len(posts))),p=posts,)
                #index = max(list(zip(range(len(posts)),posts)), key=lambda item: item[1])[0]
                particles[index].pairwise_assignment()
                self.Particle_set.append(particles[index])
                self.likelihood_dict[particles[index].pairwise_matrix_assignment_hash] = (particles[index].likelihood,particles[index].calculate_prior())
            
            self.w = self.update_weights(self.Particle_set)
        elif self.z_start == True:
            particle_initial = np.zeros(self.n).astype("int")
            for i in range(self.L):
                self.Particle_set.append(Particle(particle_initial,self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_))
            
            self.Particle_set[0].point_estimate_theta_and_theta_k(start_1=self.start_1)
            self.Particle_set[0].calculate_likelihood_given_clustering()
            
            
            for i in self.Particle_set:
                i.likelihood = self.Particle_set[0].likelihood
                i.theta = self.Particle_set[0].theta
                i.thetabeta = self.Particle_set[0].thetabeta
                i.theta_k = self.Particle_set[0].theta_k
                i.pairwise_assignment()

            self.w = self.update_weights(self.Particle_set)

            self.likelihood_dict[i.pairwise_matrix_assignment_hash] = (i.likelihood,i.calculate_prior())
            
            for index, particle in enumerate(self.Particle_set):
                self.Particle_set[index] = Particle(np.array(self.zelous_moves(index,0)),self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                self.Particle_set[index].point_estimate_theta_and_theta_k(start_1=self.start_1)
                self.Particle_set[index].calculate_likelihood_given_clustering()
                self.Particle_set[index].pairwise_assignment()
                self.likelihood_dict[self.Particle_set[index].pairwise_matrix_assignment_hash] = (self.Particle_set[index].likelihood,self.Particle_set[index].calculate_prior())
            self.w = self.update_weights(self.Particle_set)
            
        else:
            # particle_initial = self.actual_assignment#np.zeros(self.n).astype("int")
            # for i in range(self.L):
            #     self.Particle_set.append(Particle(particle_initial,self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_))
            
            # self.Particle_set[0].point_estimate_theta_and_theta_k(start_1=self.start_1)
            # self.Particle_set[0].calculate_likelihood_given_clustering()
            
            
            # for i in self.Particle_set:
            #     i.likelihood = self.Particle_set[0].likelihood
            #     i.theta = self.Particle_set[0].theta
            #     i.theta_k = self.Particle_set[0].theta_k
            #     i.thetabeta = self.Particle_set[0].thetabeta
            #     i.pairwise_assignment()

            # self.w = self.update_weights(self.Particle_set)

            # self.likelihood_dict[i.pairwise_matrix_assignment_hash] = (i.likelihood,i.calculate_prior())
            
            # for i in range(self.L):
            #     particle_initial = np.round(np.mean(self.Y,axis=1),i-3)
            #     maps = dict(zip(np.unique(particle_initial),(range(self.n))))
            #     particle_initial = np.array([maps[p] for p in particle_initial])
            #     print(particle_initial)
            #     K = max(particle_initial)+1
            #     for cluster in np.unique(particle_initial):
            #         indicies = [i for i,x in enumerate(particle_initial) if x == cluster]
            #         subgraph = nx.subgraph(self.W, indicies)
            #         if len(indicies)>1:
            #             if not nx.is_connected(subgraph):
            #                 #print(list(nx.connected_components(subgraph)))
            #                 for comp in nx.connected_components(subgraph):
            #                     comp = list(comp)
            #                     particle_initial[comp] = K
            #                     K += 1
            #     maps = dict(zip(np.unique(particle_initial),(range(self.n))))
            #     particle_initial = np.array([maps[p] for p in particle_initial]).astype('int')
            #     for cluster in np.unique(particle_initial):
            #         indicies = [i for i,x in enumerate(particle_initial) if x == cluster]
            #         subgraph = nx.subgraph(self.W, indicies)
            #         if len(indicies)>1:
            #             if not nx.is_connected(subgraph):
            #                 print('Not connected')
            #     #print(particle_initial)
            #     self.Particle_set.append(Particle(particle_initial,self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_))
            #     self.Particle_set[i].point_estimate_theta_and_theta_k(start_1=self.start_1)
            #     self.Particle_set[i].calculate_likelihood_given_clustering()
            #     self.Particle_set[i].pairwise_assignment()
            #     self.likelihood_dict[self.Particle_set[i].pairwise_matrix_assignment_hash] = (self.Particle_set[i].likelihood,self.Particle_set[i].calculate_prior())
            # self.w = self.update_weights(self.Particle_set)
            
            model_no_indicators = sm.GLM(
                self.Y.flatten(),
                self.Z.reshape(self.n*self.Z.shape[1],self.Z.shape[2]),
                offset=self.C.flatten(),
                family=sm.families.Poisson(),
            )
            results = model_no_indicators.fit()

            model_indicators = sm.GLM(
                self.Y.flatten(),
                block_diag(*self.X),
                offset=self.Z.reshape(self.n*self.Z.shape[1],self.Z.shape[2])@results.params+self.C.flatten(),
                family=sm.families.Poisson(),
            )

            res_2 = model_indicators.fit()
            vectors = res_2.params.reshape(self.n,self.X.shape[-1])
            
            posts = []
            assigns = []
            particles = []
            for paras in [vectors.T[1]]:
                for k in [0.001,0.01,0.1,1]:
                    param = np.round(paras*k).astype('int')
                    keys = np.unique(param)
                    values = range(len(keys))
                    di = dict(zip(keys,values))
                    particle_initial = np.array([di[i] for i in param])
                    print(repr(particle_initial))
                    K = max(particle_initial)+1
                    for cluster in np.unique(particle_initial):
                        indicies = [i for i,x in enumerate(particle_initial) if x == cluster]
                        subgraph = nx.subgraph(self.W, indicies)
                        if len(indicies)>1:
                            if not nx.is_connected(subgraph):
                                #print(list(nx.connected_components(subgraph)))
                                for comp in nx.connected_components(subgraph):
                                    comp = list(comp)
                                    particle_initial[comp] = K
                                    K += 1
                    maps = dict(zip(np.unique(particle_initial),(range(781))))
                    particle_initial = np.array([maps[p] for p in particle_initial]).astype('int')
                    #print(particle_initial)
                    for cluster in np.unique(particle_initial):
                        indicies = [i for i,x in enumerate(particle_initial) if x == cluster]
                        subgraph = nx.subgraph(self.W, indicies)
                        if len(indicies)>1:
                            if not nx.is_connected(subgraph):
                                print('Not connected')
                    particle = Particle(particle_initial,self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    particle.point_estimate_theta_and_theta_k(start_1=self.start_1)
                    particle.calculate_likelihood_given_clustering()
                    particle.pairwise_assignment()
                    particles.append(particle)
                    posts.append(particle.likelihood+particle.calculate_prior())
                    assigns.append(particle_initial)
                
            
            posts = posts-(max(posts))
            posts = np.exp(posts)
            norm = np.nansum(posts)
            posts = np.nan_to_num(np.divide(posts,norm))
            print('HERE ARE THE POSTS: '+str(posts))
            for l in range(self.L):
                index = np.random.choice(list(range(len(posts))),p=posts,)
                #index = max(list(zip(range(len(posts)),posts)), key=lambda item: item[1])[0]
                particles[index].pairwise_assignment()
                self.Particle_set.append(particles[index])
                print(self.Particle_set[l].likelihood+self.Particle_set[l].calculate_prior())
                self.likelihood_dict[particles[index].pairwise_matrix_assignment_hash] = (particles[index].likelihood,particles[index].calculate_prior())
            particle_zero = Particle(np.zeros(self.n).astype('int'),self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
            particle_zero.point_estimate_theta_and_theta_k(start_1=self.start_1)
            particle_zero.calculate_likelihood_given_clustering()
            print('Part 0: '+str(particle_zero.likelihood+particle_zero.calculate_prior()))
            self.w = self.update_weights(self.Particle_set)

                
        return

    def update_weights(self,P):
        M = []
        uniqiue_M = []
        indicies = np.triu_indices(P[0].pairwise_matrix_assignment.shape[0])
        max_post = -1e308
        for index,particle in enumerate(P):
            M.append((particle.pairwise_matrix_assignment[indicies]))
            unique = True
            for uni_matrix in uniqiue_M:
                if (uni_matrix==M[index]).all():
                    unique = False
                    break
            if unique == True:
                uniqiue_M.append(M[index])
            tmp_log_post = particle.likelihood
            if tmp_log_post>max_post:
                max_post = tmp_log_post
        M = np.array(M)
        uniqiue_M = np.array(uniqiue_M)
        particle_unique = []
        for matrix in M:
            for index,uniqiue_matrix in enumerate(uniqiue_M):
                if (matrix == uniqiue_matrix).all():
                    particle_unique.append(index)
                    break

        particle_counts = Counter(particle_unique)
        p_star = []
        tmp_norm = 0
        for i in range(len(uniqiue_M)):
            tmp_log_post = P[i].likelihood-max_post
            tmp_p_star = np.exp(1/self.mu * tmp_log_post)
            tmp_norm += tmp_p_star
            p_star.append(tmp_p_star)
        for index,p in enumerate(p_star):
            p_star[index] = p/tmp_norm
        w=[]
        for index,particle in enumerate(P):
            weight = p_star[particle_unique[index]]/particle_counts[particle_unique[index]]
            w.append(weight)


        return w
    
    def weight_entropy(self,weights):
        '''
        Calculate the entropy term by combining weights that are the same.

        Parameters
        ----------
        weights : LIST
            List of weight values.

        Returns
        -------
        entropy : FLOAT
            The entropy of the weights.

        '''
        #Count occurences of weights and their value
        condensed_weights = np.array(list(Counter(weights).keys()))
        occurences = np.array(list(Counter(weights).values()))
        
        #Multiply each weight by its occurence
        weights = condensed_weights*occurences
        
        #Calculate the entropy using shortended list
        #print('WEIGHTS: '+str(weights))
        entropy = np.sum(np.multiply(weights,np.log(weights)))
        return entropy
    
    def rand(self,pair_wise_2):
        a = 0
        b = 0
        c = 0
        d = 0
        for i in range(len(self.pair_wise_1)):
            if self.pair_wise_1[i] == 1 and pair_wise_2[i] == 1:
                a +=1
            elif self.pair_wise_1[i] == 0 and pair_wise_2[i] == 0:
                b +=1
            elif self.pair_wise_1[i] == 1 and pair_wise_2[i] == 0:
                c +=1
            else:
                d +=1
        return (a+b)

    def optimise_PARTOPT(self,temp=False,zeal=False,global_m = False):
        print("ZEAL: "+str(zeal))
        pair_1 = self.pair_wise_ass_2(self.actual_assignment)
        self.pair_wise_1 = pair_1
        if temp:
            objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,0)
        else:
            objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,-1)
        
        previous_objective = np.nextafter(0, 1)
        
        iter_count = -1
        
        for index, particle in enumerate(self.Particle_set):
            self.moves_accepted[index].append(particle.assignment)
            self.moves_accepted_key[index].append("start")
            
            pair_2 = self.pair_wise_ass_2(particle.assignment)
            
            self.moves_rand[index].append(self.rand(pair_2)/comb(self.n,2))
        

        while previous_objective != objective and iter_count < 200:
            
            iter_count += 1
            print('Iter Count: '+str(iter_count))

            previous_objective = objective
            self.potentials = []
            
            print('START')
            
            #Loop over particle set
            for index, particle in enumerate(self.Particle_set):
                print(particle.assignment)
                self.potentials = []
                #For each particle generate a hash
                hash_key = tuple(self.pair_wise_ass_2(particle.assignment))
                if (zeal == 1 or zeal == 2):
                    cluster_labels = list(range(particle.K))
                    
                    tic = time.time()
                    # #with tqdm_joblib(tqdm(desc="My calculation", total=1000)) as progress_bar:
                    zeals = Parallel(n_jobs=-1)(delayed(self.zelous_moves)(index, p) for p in cluster_labels)
                    self.potentials = self.potentials+list(zip(zeals,['z']*len(zeals)))
                self.local_moves(particle)
                print(len(self.potentials))
                if global_m == 1:
                    self.global_moves(particle)
                print(len(self.potentials))
                
                if zeal == 2:
                    
                    self.potentials.append((np.array(self.zealous_start(particle)),'zl'))
                assert len(self.potentials) > 0, "Proposals is empty, critical failure."
                print(len(self.potentials))
                pairwise_assignment_hashes = [tuple(self.pair_wise_ass_2(p[0])) for p in self.potentials]
                for i in pairwise_assignment_hashes:
                    if i in self.visited.keys():
                        self.visited[i] += 1
                    else:
                        self.visited[i] = 1
                #self.potentials = [p for i,p in enumerate(self.potentials) if pairwise_assignment_hashes[i] != hash_key]
                if temp:
                    objective = self.max_finder(index,particle,pairwise_assignment_hashes,objective,iter_count) 
                else:
                    objective = self.max_finder(index,particle,pairwise_assignment_hashes,objective)
        self.iter_count = iter_count
        #self.last_islands(particle,objective)
#        self.last_zeal(objective)
#        if save_obj != objective:
#            print('Change')
#            self.optimise_PARTOPT(temp,zeal)
        return
    
    def local_moves(self,particle):
        for cluster in range(particle.K):
            try:
                for tmp_assign in particle.islands(cluster,0.025):
                    self.potentials.append((tmp_assign,"i"))
            except AssertionError as e:
                    print(e)
            except Exception as e:
                traceback.print_exc()
                print("Problem with ISLANDS")
        return
    
    def global_moves(self,particle):
        for cluster in range(particle.K):
            for k in range(self.k_min,self.k_max+1):
                try:
                    self.potentials.append((particle.K_split(k,cluster),"s"))
                except AssertionError as e:
                    print(e)
                except Exception as e:
                    print("PROBLEM IN K SPLIT")
                    traceback.print_exc()
                else:
                    
                    #Then we will try merge these splits for split merge moves
                    
                    #Changed is a list of cluster labels which have been split
                    changed = []
                    for i in zip(particle.assignment,self.potentials[-1][0]):
                        if i[0]!=i[1]:
                            changed.append(i[0])
                            changed.append(i[1])
                    changed = list(np.unique(changed))
                    changed = [change for change in changed if change <= max(self.potentials[-1][0])]
                    
                    #Create new particle which has the splits as the assignment
                    split_merge_part = Particle(self.potentials[-1][0],self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    
                    #Calculate likelihood and theta for merge term
                    split_merge_part.point_estimate_theta_and_theta_k(start_1=self.start_1)
                    split_merge_part.calculate_likelihood_given_clustering()
                    
                    #For each of these clusters in the new split clusters, try merging with neighbours
                    for cluster_2 in changed:
                        try:
                            self.potentials.append((split_merge_part.merge(cluster_2),"sm"))
                        except AssertionError as e:
                            print(e)
                        except Exception as e:
                            traceback.print_exc()
                            print("Problem with MERGE in SPLIT MERGE")
                
                #Try merging every cluster with its most similar neighbour
            try:
                self.potentials.append((particle.merge(cluster),"m"))
            except AssertionError as e:
                print(e)
            except Exception as e:
                traceback.print_exc()
                print("Problem with MERGE")
        return
    
    def objective(self,w,w_entropy,P,iter_count=-1):
        annealing_schedule = np.logspace(1,0,num=1000)
        if iter_count>999:
            iter_count = -1
        obj = -self.mu*w_entropy+np.sum(np.multiply(w,np.array([i.likelihood+i.calculate_prior() for i in P])))
        return obj#**annealing_schedule[iter_count]


    def pair_wise_ass_2(self,assignment):
        '''
        Generates hash key of assignment data for label swapping comparison.

        Parameters
        ----------
        n_obs : INT
            Number of areial units.
        assignment : NUMPY ARRAY
            Numpy array of integers exprerssing aerial unit clustering labels.

        Returns
        -------
        key : INT
            Hash integer.

        '''
        start = time.time()
        # n_obs = len(assignment)
        # matrix = []
        # for i in range(n_obs):
        #     paired = []
        #     for j in range(n_obs):
        #         if assignment[i] == assignment[j]:
        #             paired.append(1)
        #         else:
        #             paired.append(0)
        #     matrix.append(paired)
        matrix = np.equal.outer(assignment,assignment)
        #matrix = np.array(matrix)
        matrix = matrix[np.triu_indices_from(matrix, k=1)]
        #print('Compute Pairwise: '+str(time.time()-start))
        return matrix
    
    def potential_check(self,i,p,pairwise_assignment_hashes,index,particle,iter_count):
        # #If the likelihood has been calculated before
        # start_1 = time.time()
        # if pairwise_assignment_hashes[i] in likelihood_dict:
            
        #     start_1 = time.time()
        #     #The likelihoods are the whole particle sets likelihoods
        #     likelihoods = np.array([j.likelihood+j.calculate_prior() for j in Particle_set])
            
        #     #Change current particles likelihood to potential's likelihood
        #     likelihoods[index] = likelihood_dict[pairwise_assignment_hashes[i]][0]+likelihood_dict[pairwise_assignment_hashes[i]][1]
        #     #Create a temp particle and compute needed values
        #     temp_particle = Particle(p,particle.y,particle.X,particle.W,k_0=particle.k_0,mu=particle.mu,nu=particle.nu,rho=0.9,prior_alpha = particle.prior_alpha,prior_theta = particle.prior_theta,lambda_=particle.lambda_)
        #     temp_particle.likelihood = likelihood_dict[pairwise_assignment_hashes[i]][0]
        #     temp_particle.pairwise_assignment()
        #     #Create the a temp particle set and weights
        #     temp_set = Particle_set[:index]+[temp_particle]+Particle_set[index+1:]
        #     temp_w = update_weights(temp_set, mu)
        
        # #If this is is a new clustering
        # else:
        start_2 = time.time()
        #Find what clusters have changed from the particle to the new potential
        changed = []

        for i in zip(particle.assignment,p):
            if i[0]!=i[1]:
                changed.append(i[0])
                changed.append(i[1])
        changed = list(np.unique(changed))
        changed = [change for change in changed if change <= max(p)]
        
        #Create a temp particle
        temp_particle = Particle(p,self.Y,self.X,self.Z,self.W,self.C,k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
        
        #Calculate the likelihood and inherit the particle terms from before
        temp_particle.point_estimate_theta_and_theta_k(start_1=self.start_1)
        temp_particle.calculate_likelihood_given_clustering()
        temp_particle.pairwise_assignment()

        #Add to the dictionary of calculated likelihoods
        self.likelihood_dict[temp_particle.pairwise_matrix_assignment_hash] = [temp_particle.likelihood,temp_particle.calculate_prior()]
        
        #Create temp particle set and likelihoods and weights
        temp_set = self.Particle_set[:index]+[temp_particle]+self.Particle_set[index+1:]
        likelihoods = np.array([j.likelihood+j.calculate_prior() for j in temp_set])
        temp_w = self.update_weights(temp_set)
        
        #Calculate temp entropy and then objective
        temp_objective = self.objective(temp_w, self.weight_entropy(temp_w),temp_set,iter_count)
        #print("CHECK TIME: "+str(time.time()-start))
        return [temp_objective, temp_particle]
    
    def max_finder(self,index,particle,pairwise_assignment_hashes,objective,iter_count=-1):
        objectives = []
        #tic = time.time()
        # for i,p in enumerate(self.potentials):
        #     objectives.append((p[0],self.potential_check(i,p[0],pairwise_assignment_hashes,index,particle,iter_count)))
        # print(time.time()-tic)
        # # with tqdm_joblib(tqdm(desc="My calculation", total=len(objectives))) as progress_bar:
        
        # tic = time.time()
        objectives = list(zip([p[0] for p in self.potentials],Parallel(n_jobs=-1)(delayed(self.potential_check)(p[0], p[1][0],pairwise_assignment_hashes,index,particle,iter_count) for p in list(enumerate(self.potentials)))))
        
        # max_objectives = max(zip(range(len(objectives)),objectives), key=lambda item: item[1][1])
        # print(time.time()-tic)
        #Loop over potentials
        max_objectives = max(zip(range(len(objectives)),objectives), key=lambda item: item[1][1][0])
        
        max_objective = max_objectives[1][1][0]
        max_potential = max_objectives[1][0]
        index_of_max = max_objectives[0]
        # #After finding the max potenetial check if it actually improves on the objective
        if max_objective > objective:
            self.moves_accepted[index].append(str(self.potentials[index_of_max][0]))
            self.moves_accepted_key[index].append(str(self.potentials[index_of_max][1]))
            print(str(self.potentials[index_of_max][1]))
            
            pair_2 = self.pair_wise_ass_2(max_potential)
            self.moves_rand[index].append(self.rand(pair_2)/comb(self.n,2))
            self.Particle_set[index] = max_objectives[1][1][1]
            self.Particle_set[index].pairwise_assignment()
            

            self.w = self.update_weights(self.Particle_set)
            objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,iter_count)

        return objective
    
    def last_islands(self, particle, objective):
        self.potentials = []
        for index, particle in enumerate(self.Particle_set):
            k = 0
            while k <= particle.K:
                try:
                    for tmp_assign in particle.islands(k,2,no_borders=True):
                        self.potentials.append((tmp_assign,"Li"))
                except AssertionError as e:
                    print(e)
                except Exception as e:
                    traceback.print_exc()
                    print("Problem with ISLANDS")
                pairwise_assignment_hashes = [tuple(self.pair_wise_ass_2(p[0])) for p in self.potentials]
                #if n<=450 and len(potentials)<750:
                objective = self.max_finder(index, particle, pairwise_assignment_hashes, objective)

                particle = self.Particle_set[index]
                k += 1
        return
    
    def last_zeal(self,objective):
        print('LAST ZEAL')
        self.potentials = []
        for index, particle in enumerate(self.Particle_set):
            for c in range(particle.K):
                self.zelous_moves(particle, c)
            self.potentials.append((np.array(self.zealous_start(particle)),'zl'))
            pairwise_assignment_hashes = [tuple(self.pair_wise_ass_2(p[0])) for p in self.potentials]
            #if n<=450 and len(potentials)<750:
            objective = self.max_finder(index, particle, pairwise_assignment_hashes, objective)
        return


    
    from collections import deque
    
    def bfs(self,adj_matrix, start):
        """
        Perform Breadth-First Search on a graph represented by an adjacency matrix.
        
        Args:
            adj_matrix (list of list of int): The adjacency matrix of the graph.
            start (int): The starting node index.
    
        Returns:
            list of int: The order of nodes visited in BFS traversal.
        """
        visited = [False] * len(adj_matrix)
        queue = deque([start])
        order = []
    
        visited[start] = True
    
        while queue:
            node = queue.popleft()
            order.append(node)
    
            for neighbor, is_connected in enumerate(adj_matrix[node]):
                if is_connected and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    
        return order


    
    def zelous_moves(self,particle,cluster):
        #print('Cluster: '+str(cluster))
        options = []
        #Generate random start
        not_indicies = [i for i,x in enumerate(self.Particle_set[particle].assignment) if x != cluster]
        indicies = [i for i,x in enumerate(self.Particle_set[particle].assignment) if x == cluster]
        #print('Indicies: '+str(indicies))
        if len(indicies) <= 1:
            return
        random_start = np.random.choice(indicies)
        #print('Start: '+str(random_start))
        #Initilaise queue with neighbours
        W_mat = nx.adjacency_matrix(self.W).toarray()
        Q = np.nonzero(W_mat[:,random_start])[0].tolist()
        #Visited = queue.copy()
        V = not_indicies+[random_start]
        #Assign first element to cluster 0
        assignment = self.Particle_set[particle].assignment[not_indicies+[random_start]].tolist()
        order = not_indicies+[random_start]
        tic = time.time()
        #while queue is not empty:
        print(cluster)
        while len(Q) > 0:
            index = Q[0]
            #print('Index: '+str(index))
            Q = Q[1:]
            if index in V:
                continue
            V.append(index)
            neighbours = np.nonzero(W_mat[:,index])[0].tolist()
            #print('Neighbours: '+str(neighbours))
            #print('Assignment of 0: '+str(assignment[0]))
            Q = Q+list(set(neighbours)-set(V))
            #Q = list(set(Q))
            temp_assignments = []
            temp_particle_set = []
            order = order+[index]
            for count,p in enumerate(self.Particle_set):
                
                if count == particle:
                    continue
                else:
                    temp_parts_a = p.assignment[order]
                    
                    uniques = np.unique(temp_parts_a).tolist()
                    keys = dict(zip(uniques, range(len(uniques))))
                    temp_parts_a = np.array([keys[i] for i in temp_parts_a])
                    temp_parts = Particle(temp_parts_a,self.Y[order],self.X[order],self.Z[order],self.W.subgraph(order),self.C[order],k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    temp_parts.point_estimate_theta_and_theta_k(start_1=self.start_1)
                    temp_parts.calculate_likelihood_given_clustering()
                    temp_parts.pairwise_assignment()
                    
                    temp_particle_set.append(temp_parts)

            for j in range(0,max(assignment)+2):
                
                indicies = [i[1] for i in zip(assignment,order) if i[0]==j]
                #print('Temp Label: '+str(j))
                #print('Current indicies of j: '+str(indicies))
                if sum(W_mat[:,index][indicies]) >= 1 or j == max(assignment)+1:

                    temp_assign = np.array(assignment.copy()+[j])
                    #print('Tmp Assignment of 0: '+str(temp_assign[0]))
                    #print('Tmp Assignment of 242: '+str(temp_assign[338]))
                    temp_particle = Particle(temp_assign,self.Y[order],self.X[order],self.Z[order],self.W.subgraph(order),self.C[order],k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    temp_particle.point_estimate_theta_and_theta_k(start_1=self.start_1)
                    temp_particle.calculate_likelihood_given_clustering()

                    temp_particle.pairwise_assignment()

                    temp_weights = self.update_weights(temp_particle_set+[temp_particle])
                    obj_temp = self.objective(temp_weights, self.weight_entropy(temp_weights),temp_particle_set+[temp_particle],-1)

                    temp_assignments.append((obj_temp,temp_assign,j))
            
            options.append(temp_assignments)
            #print('Assignment: '+str(max(temp_assignments,key=lambda x: x[0])[-1]))

            assignment = list(max(temp_assignments, key=lambda item: item[0])[1])
            #print('Assignment of 242: '+str(assignment[338]))
            #input()
        #for j in 0 to max(assignment)+1
        #Find indicies of cluster
        #See if connected
        #Define particles of potential
        #Calculate potential
        #choose best
        #add neighbours such that not in visited
        assignment = [i[0] for i in sorted(zip(assignment, order), key=lambda x:x[1])]
        for cluster in np.unique(assignment):
            indicies = [i for i,x in enumerate(self.Particle_set[particle].assignment) if x == cluster]
            subgraph = nx.subgraph(self.W, indicies)
            if len(indicies)>1:
                if not nx.is_connected(subgraph):
                    print('Not Connected')
                    print(options)
                    print(len(assignment))
                    return
        if len(assignment) != self.n:
            return
        #self.potentials.append((np.array(assignment),'z'))
        return np.array(assignment)#,options,order
    
    def zealous_start(self,particle):
        options = []
        #Generate random start
        random_start = np.random.choice(self.n)
        #Initilaise queue with neighbours
        W_mat = nx.adjacency_matrix(self.W).toarray()
        Q = np.nonzero(W_mat[:,random_start])[0].tolist()
        #Visited = queue.copy()
        V = [random_start]
        #Assign first element to cluster 0
        assignment = [0]
        order = [random_start]
        #while queue is not empty:
        while len(Q) > 0:
            index = Q[0]
            Q = Q[1:]
            if index in V:
                continue
            V.append(index)
            neighbours = np.nonzero(W_mat[:,index])[0].tolist()
            Q = Q+list(set(neighbours)-set(V))
            #Q = list(set(Q))
            temp_assignments = []
            temp_particle_set = []
            order = order+[index]
            for p in self.Particle_set:
                if p == particle:
                    continue
                else:
                    temp_parts_a = p.assignment[order]
                    
                    uniques = np.unique(temp_parts_a).tolist()
                    keys = dict(zip(uniques, range(len(uniques))))
                    temp_parts_a = np.array([keys[i] for i in temp_parts_a])
                    temp_parts = Particle(temp_parts_a,self.Y[order],self.X[order],self.W.subgraph(order),k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    
                    temp_parts.point_estimate_theta_and_theta_k(start_1=self.start_1)
                    temp_parts.calculate_likelihood_given_clustering()
                    temp_parts.pairwise_assignment()

                    temp_particle_set.append(temp_parts)
            for j in range(0,max(assignment)+2):
                indicies = [i[1] for i in zip(assignment,order) if i[0]==j]
                
                if sum(W_mat[:,index][indicies]) >= 1 or j == max(assignment)+1:
                    temp_assign = np.array(assignment.copy()+[j])

                    uniques = np.unique(temp_assign).tolist()
                    keys = dict(zip(uniques, range(len(uniques))))
                    temp_assign = np.array([keys[i] for i in temp_assign])
                    temp_particle = Particle(temp_assign,self.Y[order],self.X[order],self.Z[order],self.W.subgraph(order),self.C[order],k_0=self.k_0,alpha=self.alpha,beta=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    temp_particle.point_estimate_theta_and_theta_k(start_1=self.start_1)
                    temp_particle.calculate_likelihood_given_clustering()
                    temp_particle.pairwise_assignment()

                    temp_weights = self.update_weights(temp_particle_set+[temp_particle])
                    obj_temp = self.objective(temp_weights, self.weight_entropy(temp_weights),temp_particle_set+[temp_particle],-1)
                    temp_assignments.append((obj_temp,temp_assign,j))
            options.append(temp_assignments)
            assignment = list(max(temp_assignments, key=lambda item: item[0])[1])
        #for j in 0 to max(assignment)+1
        #Find indicies of cluster
        #See if connected
        #Define particles of potential
        #Calculate potential
        #choose best
        #add neighbours such that not in visited
        assignment = [i[0] for i in sorted(zip(assignment, order), key=lambda x:x[1])]
        for cluster in np.unique(assignment):
            indicies = [i for i,x in enumerate(particle.assignment) if x == cluster]
            subgraph = nx.subgraph(self.W, indicies)
            if len(indicies)>1:
                if not nx.is_connected(subgraph):
                    print('Not Connected')
                    print(options)
                    print(len(assignment))
                    return np.zeros(self.n).astype("int")
        if len(assignment) != self.n:
            return np.zeros(self.n).astype("int")
        return assignment
