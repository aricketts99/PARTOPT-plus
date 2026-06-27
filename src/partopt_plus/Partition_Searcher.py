#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:07:41 2025

@author: andrew
"""
import time
import traceback
from math import comb
from collections import Counter
import numpy as np
from partopt_plus.Particle_MCAR import Particle_MCAR as Particle
import networkx as nx
from scipy import optimize
from scipy.stats import invgamma
import csv


class Partition_Searcher():
    
    def __init__(self,actual_assignment,Y,X,W,L=1,Kmeans_initliaise=False,mu = 1,lambda_=1,k_min = 2,k_max = 4,k_0=1,rho=0.9,alpha=1,beta=1,prior_alpha=7,prior_theta=0.1,z_start=False,low_var=True):
        self.Particle_set = []
        self.lambda_Particle_set = []
        self.visited = {}
        self.actual_assignment = actual_assignment
        self.Y = Y
        self.X = X
        self.W = W
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
    
    def instantiate_set(self):
        self.Particle_set = []
        if self.Kmeans_initliaise:
            random_k = np.random.randint(1,self.n,self.L)
            
            if self.low_var == True:
                particle_initial = np.zeros(self.n).astype("int")
            else:
                particle_initial = np.array(list(range(self.n))).astype('int')
            emp_part = Particle(particle_initial,self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
            emp_part.point_estimate_theta_and_theta_k()
            particle_initial = np.zeros(self.n).astype("int")
            emp_part.assignment = particle_initial
            emp_part.K = 1
            posts = []
            assigns = []
            particles = []
            for k in range(2,int(np.log(100*self.n))):
                split_assign = emp_part.K_split(k, 0)
                new_emp_part = Particle(split_assign,self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                new_emp_part.calculate_likelihood_given_clustering(list(range(max(split_assign)+1)), [0], self.lambda_)
                post = new_emp_part.likelihood+new_emp_part.calculate_prior()
                assigns.append(split_assign)
                posts.append(post)
                particles.append(new_emp_part)
            
            norm = np.sum(posts)
            posts = [p/norm for p in posts]
            for l in range(self.L):
                index = np.random.choice(list(range(len(posts))),p=posts)
                index = max(list(zip(range(len(posts)),posts)), key=lambda item: item[1])[0]
                particles[index].pairwise_assignment()
                particles[index].point_estimate_theta_and_theta_k()
                self.Particle_set.append(particles[index])
                self.likelihood_dict[particles[index].pairwise_matrix_assignment_hash] = (particles[index].likelihood,particles[index].calculate_prior())
            
            self.w = self.update_weights(self.Particle_set)
        elif self.z_start == True:
            particle_initial = np.zeros(self.n).astype("int")
            for i in range(self.L):
                self.Particle_set.append(Particle(particle_initial,self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_))

            self.Particle_set[0].calculate_likelihood_given_clustering(list(range(1+int(max(particle_initial)))),[0],self.lambda_)
            self.Particle_set[0].point_estimate_theta_and_theta_k()
            
            for i in self.Particle_set:
                i.likelihood = self.Particle_set[0].likelihood
                i.terms = self.Particle_set[0].terms
                i.theta = self.Particle_set[0].theta
                i.theta_k = self.Particle_set[0].theta_k
                i.pairwise_assignment()

            self.w = self.update_weights(self.Particle_set)

            self.likelihood_dict[i.pairwise_matrix_assignment_hash] = (i.likelihood,i.calculate_prior())
            
            for index, particle in enumerate(self.Particle_set):
                self.Particle_set[index] = Particle(np.array(self.zealous_start(particle)),self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                self.Particle_set[index].calculate_likelihood_given_clustering(list(range(1+int(max(self.Particle_set[index].assignment)))),[0],self.lambda_)
                self.Particle_set[index].pairwise_assignment()
                self.Particle_set[index].point_estimate_theta_and_theta_k()
                self.likelihood_dict[self.Particle_set[index].pairwise_matrix_assignment_hash] = (self.Particle_set[index].likelihood,self.Particle_set[index].calculate_prior())
            self.w = self.update_weights(self.Particle_set)
            
        else:
            particle_initial = np.zeros(self.n).astype("int")
            for i in range(self.L):
                self.Particle_set.append(Particle(particle_initial,self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_))

            self.Particle_set[0].calculate_likelihood_given_clustering(list(range(1+int(max(particle_initial)))),[0],self.lambda_)
            self.Particle_set[0].point_estimate_theta_and_theta_k()
            
            for i in self.Particle_set:
                i.likelihood = self.Particle_set[0].likelihood
                i.terms = self.Particle_set[0].terms
                i.theta = self.Particle_set[0].theta
                i.theta_k = self.Particle_set[0].theta_k
                i.pairwise_assignment()

            self.w = self.update_weights(self.Particle_set)

            self.likelihood_dict[i.pairwise_matrix_assignment_hash] = (i.likelihood,i.calculate_prior())
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
        pair_1 = self.pair_wise_ass_2(self.actual_assignment)
        self.pair_wise_1 = pair_1
        if temp:
            objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,0)
        else:
            objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,-1)
        
        previous_objective = objective -1
        
        iter_count = -1
        
        for index, particle in enumerate(self.Particle_set):
            self.moves_accepted[index].append(particle.assignment)
            self.moves_accepted_key[index].append("start")
            
            pair_2 = self.pair_wise_ass_2(particle.assignment)
            
            self.moves_rand[index].append(self.rand(pair_2)/comb(self.n,2))
        

        while previous_objective != objective and iter_count < 200:
            print('Iter Count: '+str(iter_count))
            iter_count += 1
            
            previous_objective = objective
            self.potentials = []
            
            #Loop over particle set
            for index, particle in enumerate(self.Particle_set):
                self.potentials = []
                #For each particle generate a hash
                hash_key = tuple(self.pair_wise_ass_2(particle.assignment))
                print('Len of potentials: '+str(len(self.potentials)))
                if zeal==1:
                    for c in range(particle.K):
                        self.zelous_moves(particle,c)
                if zeal == 2 and iter_count%3:
                    for c in range(particle.K):
                        self.zelous_moves(particle,c)
                print('Len of potentials: '+str(len(self.potentials)))
                self.local_moves(particle)
                print('Len of potentials: '+str(len(self.potentials)))
                if global_m == 1:
                    self.global_moves(particle)
                print('Len of potentials: '+str(len(self.potentials)))
                
                if iter_count%3 == 0 and (zeal == 1 or zeal==2):
                    
                    self.potentials.append((np.array(self.zealous_start(particle)),'zl'))
                print('Len of potentials: '+str(len(self.potentials)))
                assert len(self.potentials) > 0, "Proposals is empty, critical failure."

                pairwise_assignment_hashes = [tuple(self.pair_wise_ass_2(p[0])) for p in self.potentials]
                print('Done with Pairwise')
                for i in pairwise_assignment_hashes:
                    if i in self.visited.keys():
                        self.visited[i] += 1
                    else:
                        self.visited[i] = 1
                
                self.potentials = [p for i,p in enumerate(self.potentials) if pairwise_assignment_hashes[i] != hash_key]
                print(len(self.potentials))
                if temp:
                    objective = self.max_finder(index,particle,pairwise_assignment_hashes,objective,iter_count) 
                else:
                    objective = self.max_finder(index,particle,pairwise_assignment_hashes,objective)
            #for index,particle in enumerate(self.lambda_Particle_set):
            minimum = optimize.minimize(self.lambda_objective, self.lambda_,bounds=[(0, None)])
            self.lambda_ = minimum['x'][0]
            #reinstantiate particles (CHANGE DICTONARY TO INVOLVE LAMBDA SO AS TO NOT BREAK THINGS)
            for index, particle in enumerate(self.Particle_set):
                self.Particle_set[index] = Particle(np.array(self.zealous_start(particle)),self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                self.Particle_set[index].calculate_likelihood_given_clustering(list(range(1+int(max(self.Particle_set[index].assignment)))),[0],self.lambda_)
                self.Particle_set[index].pairwise_assignment()
                self.Particle_set[index].point_estimate_theta_and_theta_k()
                self.likelihood_dict[self.Particle_set[index].pairwise_matrix_assignment_hash] = (self.Particle_set[index].likelihood,self.Particle_set[index].calculate_prior())
        save_obj = objective
        self.iter_count = iter_count
        self.last_islands(particle,objective)
        self.last_zeal(objective)
        if save_obj != objective:
            print('Change')
            self.optimise_PARTOPT(temp,zeal)
        return
    
    def lambda_objective(self,lambda_):
        temp_particle_set = []
        for i in range(self.L):
            temp_particle_set.append(Particle(self.Particle_set[i].assignment,self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=lambda_))
            temp_particle_set[i].calculate_likelihood_given_clustering(list(range(1+int(max(temp_particle_set[i].assignment)))),[0],lambda_)
        return -1*(invgamma.pdf(lambda_, self.alpha, self.beta) + np.sum([temp_particle_set[i].likelihood for i in range(self.L)]))
    
    
    def local_moves(self,particle):
        for cluster in range(particle.K):
            for tmp_assign in particle.islands(cluster,0.05):
                try:
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
                    split_merge_part = Particle(self.potentials[-1][0],self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    
                    #Calculate likelihood and theta for merge term
                    split_merge_part.calculate_likelihood_given_clustering(changed,particle.terms,self.lambda_)
                    split_merge_part.point_estimate_theta_and_theta_k()
                    
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
        temp_particle = Particle(p,particle.y,particle.X,particle.W,k_0=particle.k_0,mu=particle.mu,nu=particle.nu,rho=0.9,prior_alpha = particle.prior_alpha,prior_theta = particle.prior_theta,lambda_=particle.lambda_)
        
        #Calculate the likelihood and inherit the particle terms from before
        temp_particle.calculate_likelihood_given_clustering(changed,particle.terms,particle.lambda_)
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
        return temp_objective
    
    def max_finder(self,index,particle,pairwise_assignment_hashes,objective,iter_count=-1):
        objectives = []
        for i,p in enumerate(self.potentials):
            objectives.append((p[0],self.potential_check(i,p[0],pairwise_assignment_hashes,index,particle,iter_count)))
        
        max_objectives = max(zip(range(len(objectives)),objectives), key=lambda item: item[1][1])
        
        #Loop over potentials
        max_objectives = max(zip(range(len(objectives)),objectives), key=lambda item: item[1][1])
        
        max_objective = max_objectives[1][1]
        max_potential = max_objectives[1][0]
        index_of_max = max_objectives[0]
        # #After finding the max potenetial check if it actually improves on the objective
        if max_objective > objective:
            self.moves_accepted[index].append(str(self.potentials[index_of_max][0]))
            self.moves_accepted_key[index].append(str(self.potentials[index_of_max][1]))
            print(str(self.potentials[index_of_max][1]))
            
            pair_2 = self.pair_wise_ass_2(max_potential)
            self.moves_rand[index].append(self.rand(pair_2)/comb(self.n,2))
            self.Particle_set[index] = Particle(max_potential,self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
            self.Particle_set[index].calculate_likelihood_given_clustering(list(range(self.Particle_set[index].K)),[0],self.lambda_)
            self.Particle_set[index].point_estimate_theta_and_theta_k()
            self.Particle_set[index].pairwise_assignment()
            

            self.w = self.update_weights(self.Particle_set)
            objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,iter_count)

        return objective
    
    def last_islands(self, particle, objective):
        self.potentials = []
        for index, particle in enumerate(self.Particle_set):
            k = 0
            while k <= particle.K:
                for tmp_assign in particle.islands(k,2,no_borders=True):
                    try:
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
    
    # def prop_to_optimise(self,temp=False,zeal=False):
    #     pair_1 = self.pair_wise_ass_2(self.actual_assignment)
    #     self.pair_wise_1 = pair_1
    #     if temp:
    #         objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,0)
    #     else:
    #         objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,-1)
        
    #     previous_objective = objective-1
        
    #     iter_count = 0
        
    #     for index, particle in enumerate(self.Particle_set):
    #         self.moves_accepted[index].append(particle.assignment)
    #         self.moves_accepted_key[index].append("start")
            
    #         pair_2 = self.pair_wise_ass_2(particle.assignment)
            
    #         self.moves_rand[index].append(self.rand(pair_2)/comb(self.n,2))
        
    #     while iter_count < 150:
    #         iter_count += 1
    #         previous_objective = objective
    #         self.potentials = []
            
    #         #Loop over particle set
    #         for index, particle in enumerate(self.Particle_set):
    #             self.potentials = []
    #             #For each particle generate a hash
    #             hash_key = tuple(self.pair_wise_ass_2(particle.assignment))
                
    #             if zeal == True:
    #                 self.zelous_moves(particle)
    #             self.local_moves(particle)
    #             self.global_moves(particle)
                
                
    #             assert len(self.potentials) > 0, "Proposals is empty, critical failure."
                
    #             pairwise_assignment_hashes = [tuple(self.pair_wise_ass_2(p[0])) for p in self.potentials]
    #             for i in pairwise_assignment_hashes:
    #                 if i in self.visited.keys():
    #                     self.visited[i] += 1
    #                 else:
    #                     self.visited[i] = 1
                
    #             self.potentials = [p for i,p in enumerate(self.potentials) if pairwise_assignment_hashes[i] != hash_key]
                    
    #             objective = self.prop_to_finder(index,particle,pairwise_assignment_hashes,objective,iter_count) 
    #     self.iter_count = iter_count
    #     self.last_islands(index,particle,objective)
    #     return
    
    # def prop_to_finder(self,index,particle,pairwise_assignment_hashes,objective,iter_count):
    #     objectives = []
    #     for i,p in enumerate(self.potentials):
    #         objectives.append((p[0],self.potential_check(i,p[0],pairwise_assignment_hashes,index,particle,-1)))
        
    #     annealing_schedule = np.logspace(100,0,num=75)
    #     if iter_count>74:
    #         iter_count = -1
        
    #     #objectives = [(o[0],1) if  o[1] > objective else (o[0],np.exp((o[1]-objective))/annealing_schedule[iter_count]) for o in objectives]
    #     objectives = [(o[0],o[1]/annealing_schedule[iter_count]) for o in objectives]
        
    #     if len(objectives) == 0:
    #         return objective
    #     else:
    #         probs = np.array([o[1] for o in objectives])
    #         norm = sum(probs)
    #         probs = [p/norm for p in probs]
    #         index_of_max = np.random.choice(len(objectives), 1, p=probs)[0]
    #         max_potential = objectives[index_of_max][0]

    #     # #After finding the max potenetial check if it actually improves on the objective

    #         self.moves_accepted[index].append(str(self.potentials[index_of_max][0]))
    #         self.moves_accepted_key[index].append(str(self.potentials[index_of_max][1]))
            
    #         pair_2 = self.pair_wise_ass_2(max_potential)
            
    #         self.moves_rand[index].append(self.rand(pair_2)/comb(self.n,2))
    #         self.Particle_set[index] = Particle(max_potential,self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
    #         self.Particle_set[index].calculate_likelihood_given_clustering(list(range(self.Particle_set[index].K)),[0],self.lambda_)
    #         self.Particle_set[index].point_estimate_theta_and_theta_k()
    #         self.Particle_set[index].pairwise_assignment()
            

    #         self.w = self.update_weights(self.Particle_set)
            
    #         objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,-1)

    #     return objective
    
    # def simulated_annealing(self,zeal=False):
    #     pair_1 = self.pair_wise_ass_2(self.actual_assignment)
    #     self.pair_wise_1 = pair_1
    #     objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,-1)
        
    #     previous_objective = 0
        
    #     iter_count = 0
            
        
    #     for index, particle in enumerate(self.Particle_set):
    #         self.moves_accepted[index].append(particle.assignment)
    #         self.moves_accepted_key[index].append("start")
            
    #         pair_2 = self.pair_wise_ass_2(particle.assignment)
            
    #         self.moves_rand[index].append(self.rand(pair_2)/comb(self.n,2))
        

    #     while iter_count < 100 or previous_objective!=objective:
    #         print('Iter Count: '+str(iter_count))
    #         iter_count += 1
    #         previous_objective = objective
            
            
    #         #Loop over particle set
    #         for index, particle in enumerate(self.Particle_set):
    #             self.potentials = []
    #             #For each particle generate a hash
    #             hash_key = tuple(self.pair_wise_ass_2(particle.assignment))

    #             self.local_moves(particle)
    #             if iter_count>75:
    #                 if zeal==True:
                        
    #                     self.zelous_moves(particle)
    #                 self.global_moves(particle)

                
    #             assert len(self.potentials) > 0, "Proposals is empty, critical failure."
                
    #             if iter_count <= 75:
                    
    #                 self.potentials = [self.potentials[np.random.choice(len(self.potentials),1)[0]]]
    
    #                 for i in [tuple(self.pair_wise_ass_2(self.potentials[0][0]))]:
    #                     if i in self.visited.keys():
    #                         self.visited[i] += 1
    #                     else:
    #                         self.visited[i] = 1
                    
                    
    #                 objective = self.simulated_M_H(objective, [tuple(self.pair_wise_ass_2(self.potentials[0][0]))], index, particle, iter_count)
    #             else:
    #                 pairwise_assignment_hashes = [tuple(self.pair_wise_ass_2(p[0])) for p in self.potentials]

    #                 for i in pairwise_assignment_hashes:
    #                     if i in self.visited.keys():
    #                         self.visited[i] += 1
    #                     else:
    #                         self.visited[i] = 1
                    
    #                 self.potentials = [p for i,p in enumerate(self.potentials) if pairwise_assignment_hashes[i] != hash_key]
                    
    #                 objective = self.max_finder(index,particle,pairwise_assignment_hashes,objective)
                    
                
    #     self.iter_count = iter_count
    #     self.last_islands(index,particle,objective)
    #     return
    
    # def simulated_M_H(self,objective,pairwise_assignment_hashes,index,particle,iter_count):
    #     objectives = []
    #     for i,p in enumerate(self.potentials):
    #         objectives.append((p[0],self.potential_check(i,p[0],pairwise_assignment_hashes,index,particle,-1)))
        
    #     annealing_schedule = np.logspace(1,0,num=75)
    #     if iter_count>74:
    #         iter_count = -1
    #     if objectives[0][1] > objective:
    #         v = 1
    #     else:
    #         v = np.exp((objectives[0][1]-objective)/annealing_schedule[iter_count])
    #     if v >= np.random.uniform(0,1,1):
    #         self.moves_accepted[index].append(str(self.potentials[0][0]))
    #         self.moves_accepted_key[index].append(str(self.potentials[0][1]))
            
    #         pair_2 = self.pair_wise_ass_2(self.potentials[0][0])
            
    #         self.moves_rand[index].append(self.rand(pair_2)/comb(self.n,2))
    #         self.Particle_set[index] = Particle(self.potentials[0][0],self.Y,self.X,self.W,k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
    #         self.Particle_set[index].calculate_likelihood_given_clustering(list(range(self.Particle_set[index].K)),[0],self.lambda_)
    #         self.Particle_set[index].point_estimate_theta_and_theta_k()
    #         self.Particle_set[index].pairwise_assignment()
            
    #         self.w = self.update_weights(self.Particle_set)
            
    #         objective = self.objective(self.w, self.weight_entropy(self.w),self.Particle_set,-1)
    #     return objective
    
    def zelous_moves(self,particle,cluster):
        with open("intermediate.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow('\n')
            wr.writerow('\n')
            wr.writerow('\n')
        indicies = [i for i,x in enumerate(particle.assignment) if x == cluster]
        not_indicies = [i for i,x in enumerate(particle.assignment) if x != cluster]
        if len(indicies)<= 1:
            return
        #Generate random start
        random_start = np.random.choice(indicies)
        #Initilaise queue with neighbours
        W_mat = nx.adjacency_matrix(self.W).toarray()
        Q = list(set(np.nonzero(W_mat[:,random_start])[0].tolist())-set(not_indicies))+indicies
        #Visited = queue.copy()
        V = not_indicies+[random_start]
        #Assign first element to cluster 0
        assignment = particle.assignment[not_indicies+[random_start]].tolist()
        order = not_indicies+[random_start]

        while len(Q) > 0:
            index = Q[0]
            Q = Q[1:]

            if index in V:
                continue
            V.append(index)
            neighbours = np.nonzero(W_mat[:,index])[0].tolist()
            Q = list(set(neighbours)-set(V))+Q
            #
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
                    temp_parts = Particle(temp_parts_a,self.Y[order],self.X[order],self.W.subgraph(order),k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    temp_parts.calculate_likelihood_given_clustering(list(range(temp_parts.K)),[0],self.lambda_)
                    temp_parts.pairwise_assignment()
                    temp_particle_set.append(temp_parts)
            for j in range(0,max(assignment)+2):
                indicies = [i[1] for i in zip(assignment,order) if i[0]==j]
                
                if sum(W_mat[:,index][indicies]) >= 1 or j == max(assignment)+1:
                    temp_assign = np.array(assignment.copy()+[j])
                    uniques = np.unique(temp_assign).tolist()
                    keys = dict(zip(uniques, range(len(uniques))))
                    temp_assign = np.array([keys[i] for i in temp_assign])
                    temp_particle = Particle(temp_assign,self.Y[order],self.X[order],self.W.subgraph(order),k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    temp_particle.calculate_likelihood_given_clustering(list(range(temp_particle.K)),[0],self.lambda_)
                    temp_particle.pairwise_assignment()

                    temp_weights = self.update_weights(temp_particle_set+[temp_particle])
                    obj_temp = self.objective(temp_weights, self.weight_entropy(temp_weights),temp_particle_set+[temp_particle],-1)
                    temp_assignments.append((obj_temp,temp_assign))
            assignment = list(max(temp_assignments, key=lambda item: item[0])[1])
            with open("intermediate.csv", "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(assignment)
                wr.writerow(order)
            

        assignment = [i[0] for i in sorted(zip(assignment, order), key=lambda x:x[1])]

        self.potentials.append((np.array(assignment),'z'))
        return #print(assignment)
    
    def zealous_start(self,particle):
        with open("intermediate.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow('\n')
            wr.writerow('\n')
            wr.writerow('\n')
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
                    temp_parts = Particle(temp_parts_a,self.Y[order],self.X[order],self.W.subgraph(order),k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    temp_parts.calculate_likelihood_given_clustering(list(range(temp_parts.K)),[0],self.lambda_)
                    temp_parts.pairwise_assignment()
                    temp_particle_set.append(temp_parts)
            for j in range(0,max(assignment)+2):
                indicies = [i[1] for i in zip(assignment,order) if i[0]==j]
                
                if sum(W_mat[:,index][indicies]) >= 1 or j == max(assignment)+1:
                    temp_assign = np.array(assignment.copy()+[j])
                    uniques = np.unique(temp_assign).tolist()
                    keys = dict(zip(uniques, range(len(uniques))))
                    temp_assign = np.array([keys[i] for i in temp_assign])
                    temp_particle = Particle(temp_assign,self.Y[order],self.X[order],self.W.subgraph(order),k_0=self.k_0,mu=self.alpha,nu=self.beta,rho=self.rho,prior_alpha = self.prior_alpha,prior_theta = self.prior_theta,lambda_=self.lambda_)
                    temp_particle.calculate_likelihood_given_clustering(list(range(temp_particle.K)),[0],self.lambda_)
                    temp_particle.pairwise_assignment()

                    temp_weights = self.update_weights(temp_particle_set+[temp_particle])
                    obj_temp = self.objective(temp_weights, self.weight_entropy(temp_weights),temp_particle_set+[temp_particle],-1)
                    temp_assignments.append((obj_temp,temp_assign))
            assignment = list(max(temp_assignments, key=lambda item: item[0])[1])
            with open("intermediate.csv", "a") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(assignment)
                wr.writerow(order)
            
        #for j in 0 to max(assignment)+1
        #Find indicies of cluster
        #See if connected
        #Define particles of potential
        #Calculate potential
        #choose best
        #add neighbours such that not in visited
        assignment = [i[0] for i in sorted(zip(assignment, order), key=lambda x:x[1])]
        return assignment
