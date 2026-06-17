#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:39:10 2024

@author: andrew
"""

import math
import numpy as np
import networkx as nx
from partopt_plus.Particle import Particle
from sklearn.cluster import AgglomerativeClustering, SpectralClustering


class connected_Particle(Particle):
    def __init__(self,assignment,y,X,W,prior="EP",k_0=0.1,mu=1,nu=1,rho=0.9,prior_alpha = 0.1,prior_theta = 0.1,lambda_=1):
        
        super().__init__(assignment,y,X,W,prior,k_0,mu,nu,rho,prior_alpha,prior_theta,lambda_)

    def islands(self,cluster,percentage,no_borders=False):
        '''
        Returns
        -------
        tmp_assignment: list of list
            A list of possible assignments with islands taken out

        '''
        #Initiliase temporary variables
        tmp_assignment = self.assignment.copy()

        indicies = [i for i, x in enumerate(self.assignment) if x == cluster]
        #Check if the cluster is already an island
        if len(indicies)<=1:
            return [tmp_assignment]

        #Get adjacency matrix to check connectedness
        W = nx.from_numpy_array(self.W_mat[indicies,:][:,indicies])

        #Get island indicies and their new potential label
        island_indicies = self.find_islands(indicies,percentage)
        island_indicies = list(zip(island_indicies,[self.K]*len(island_indicies)))
        #print("Island_index+ "+str(island_indicies))
        

        #Check if islands are on border, if they are also propose merge island to border
        if no_borders==False:
            island_indicies = self.find_border_islands(indicies, island_indicies)
        tmp_assignment = [tmp_assignment.copy() for i in range(len(island_indicies))]
        #for each island_index in island_indicies, generate a new assignement of the cluster
        #with the island and the connected components of the remainder
        for index,island_index in enumerate(island_indicies):
            #Set the temporary assignment's island index to the number of clutsers
            tmp_assignment[index][island_index[0]] = island_index[1]

            #Get a graph of the not island indices in the cluster
            index_of_island = indicies.index(island_index[0])
            remainder_indicies = indicies[:index_of_island]+indicies[index_of_island+1:]
            remainder_graph = self.W.subgraph(remainder_indicies)

            #Check if they are connected
            assert len(remainder_indicies)>=1, "EMPTY REMAINDER GRAPH! FATAL ERROR"
            count = 0
            if nx.is_empty(remainder_graph):
                for remainder in remainder_indicies[1:]:
                    if island_index[1] == self.K:
                        tmp_assignment[index][remainder] = self.K+count+1
                    else:
                        tmp_assignment[index][remainder] = self.K+count
                    count +=1
            elif not nx.is_connected(remainder_graph):
                #find connected components
                
                for component in nx.connected_components(remainder_graph):
                    
                    if count != 0:
                    #For each value in each component set tmp_assignments value to
                    #its new cluster value
                        for comp in component:
                            #print(comp)
                            #print(self.K+count)
                            if island_index[1]!=self.K:
                                tmp_assignment[index][comp] = self.K+count-1
                            else:
                                tmp_assignment[index][comp] = self.K+count
                    count += 1
            
            #tmp_assignment, count = self.find_island_connectivity(tmp_assignment[index].copy(),island_index,remainder_indicies,remainder_graph)
            unique_lables = np.unique(tmp_assignment[index])
            # if island_index[1]!=self.K:
            #     if len(unique_lables) != self.K+count:
            #         print(self.assignment)
            #         print(self.K)
            #         print(count)
            #         print(tmp_assignment[index])
            #         print(unique_lables)
            #     assert len(unique_lables) == self.K+count, "ERROR IN ISLANDS MERGE"
            #     # if len(unique_lables) != self.K+count:
            #     #     tmp_assignment[index][island_index[0]] = self.assignment.copy()
            # else:
            #     if len(unique_lables) != self.K+count+1 or :
            #         print(self.assignment)
            #         print(self.K)
            #         print(count)
            #         print(tmp_assignment[index])
            #         print(unique_lables)
                
                
            #assert len(unique_lables) == self.K+count+1 or len(unique_lables) == self.K+count, "ERROR IN ISLANDS CONNECT"
                # if len(unique_lables) != self.K+count+1:
                #     tmp_assignment[index][island_index[0]] = self.assignment.copy()
            assert (unique_lables == np.array(range(len(unique_lables)))).all(), "ERROR IN LABELLING"
            for k in range(len(unique_lables)):
                check_indicies = [i for i, x in enumerate(tmp_assignment[index]) if x == k]
                if len(check_indicies)>1:
                    #if not nx.is_connected(self.W.subgraph(check_indicies)):
                        #print(check_indicies)
                        #print(k)
                    assert nx.is_connected(self.W.subgraph(check_indicies)), "ERROR IN CONNECTIVITY"
        return tmp_assignment
    def find_island_connectivity(self,tmp_assignment,island_index,remainder_indicies,remainder_graph):
        '''
        Function to find the connected clustering after islands has been run.  

        Parameters
        ----------
        tmp_assignment : NUMPY ARRAY
            Temprorary assignment.
        island_index : INT
            Index of island cluster.
        remainder_indicies : LIST
            List of other indicies that we need to find the connected sub-components.
        remainder_graph : NX GRAPH
            Graph of remainder indicies.

        Returns
        -------
        tmp_assignment : NUMPY ARRAY
            New temp assignment which is connected.
        count : INT
            The number of new clusters generaterd by reconnecting.

        '''
        count = 0
        if nx.is_empty(remainder_graph):
            for remainder in remainder_indicies[1:]:
                if island_index[1] == self.K:
                    tmp_assignment[remainder] = self.K+count+1
                else:
                    tmp_assignment[remainder] = self.K+count
                count +=1
        elif not nx.is_connected(remainder_graph):
            #find connected components
            for component in nx.connected_components(remainder_graph):
                if count == 0:
                    continue
                #For each value in each component set tmp_assignments value to
                #its new cluster value
                for comp in component:
                    tmp_assignment[comp] = self.K+count
                count += 1
        return tmp_assignment, count
    def find_border_islands(self,indicies,island_indicies):
        '''
        Method to find the clusters that islands border.  If the border another cluster
        we pair the island index with that cluster label.

        Parameters
        ----------
        indicies : LIST
            List of indicies in the choosen cluster.
        island_indicies : LIST OF TUPLE
            List of island indicies in the choosen cluster and their label.

        Returns
        -------
        island_indicies : LIST OF TUPLES
            List of island indicies in the choosen cluster and their label, now including
            the border merges.

        '''
        adj = []
        for i in island_indicies:
            adjacent = self.W_mat[i[0]]
            adjacent_clusters = []
            for j in range(self.n_obs):
                if j not in indicies:
                    if adjacent[j] == 1:
                        adjacent_clusters.append(self.assignment[j])
            adj.append(np.unique(adjacent_clusters))
        for a in range(len(adj)):
            for index in adj[a]:
                island_indicies.append((island_indicies[a][0],index))
        return island_indicies

    def merge(self,cluster):
        '''
        Method to merge two clusters

        Returns
        -------
        tmp_assignment : LIST
            List of new labels after a merge.
        '''
        tmp_assignment = self.assignment.copy()

        indicies = [i for i, x in enumerate(self.assignment) if x == cluster]
        #W = nx.adjacency_matrix(self.W).toarray()
        if len(indicies) == self.n_obs:
            return tmp_assignment

        clusters = self.find_cluster_neighbours(indicies)

        closest_cluster = self.find_closest_connected_cluster(cluster, clusters)

        indicies_2 = [i for i, x in enumerate(self.assignment) if x == closest_cluster]

        #Re-assign indicies, has to be ordered labels so also reduce the other indicies by 1
        if closest_cluster < cluster:
            for i in indicies:
                tmp_assignment[i] = closest_cluster
            for i in range(self.n_obs):
                if tmp_assignment[i]>cluster:
                    tmp_assignment[i] = tmp_assignment[i]-1
        else:
            for i in indicies_2:
                tmp_assignment[i] = cluster
            for i in range(self.n_obs):
                if tmp_assignment[i]>closest_cluster:
                    tmp_assignment[i] = tmp_assignment[i]-1
        #Asserts
        unique_lables = np.unique(tmp_assignment)
        assert len(unique_lables) == self.K-1, "ERROR IN MERGE"
        assert (unique_lables == np.array(range(len(unique_lables)))).all(), "ERROR IN LABELLING"
        for k in range(len(unique_lables)):
            indicies = [i for i, x in enumerate(self.assignment) if x == k]
            if len(indicies)>1:
                assert nx.is_connected(self.W.subgraph(indicies)), "ERROR IN CONNECTIVITY"
        return tmp_assignment

    def find_cluster_neighbours(self,indicies):
        '''
        Method to find the clusters that neighbour a given cluster

        Parameters
        ----------
        indicies : LIST
            List of indicies in choosen cluster.

        Returns
        -------
        clusters : LIST
            List of cluster labels that border the choosen cluster.

        '''
        W = nx.adjacency_matrix(self.W).toarray().astype('int')

        #find clutser neighbours
        indicies_of_neightbours = []
        for area in indicies:
            neighbours = W[area,:]
            indicies_of_neightbours = indicies_of_neightbours+(np.argwhere(neighbours==1).tolist())

        indicies_of_neightbours = [x for xs in indicies_of_neightbours for x in xs]

        clusters = []
        for i,x in enumerate(self.assignment):
            if i in indicies_of_neightbours:
                clusters.append(x)
        clusters = np.unique(clusters)

        return clusters

    def find_closest_connected_cluster(self,cluster,clusters):
        '''
        Method to find the closest connected cluster to the choosen one.

        Parameters
        ----------
        cluster : INT
            Numeric label of choosen cluster.
        clusters : LIST
            List of neighbouring clusters.

        Returns
        -------
        INT
            Numeric label of closest cluster in terms of expected theta_k.

        '''
        #find closest in terms of difference in expected value of mean
        distances = []
        for c in clusters:
            if c == cluster:
                distances.append(np.inf)
            else:
                distances.append(np.linalg.norm(self.theta_k[c]-self.theta_k[cluster]))
        return clusters[np.argmin(distances)]


    def split(self,cluster):
        '''
        Returns
        -------
        tmp_assignment : list
            a label assignment that splits a cluster into two connected parts.
        '''
        return self.K_split(2,cluster)

    def K_split(self,K,cluster):
        '''
        

        Parameters
        ----------
        K : 2<=int
            DESCRIPTION.

        Returns
        -------
        tmp_assignment : list
            a label assignment of the indicies into K connected subclusters

        '''

        tmp_assignment = self.assignment.copy()


        indicies = [i for i, x in enumerate(self.assignment) if x == cluster]

        if K >= 0.5*len(indicies) or len(indicies) == 1:
            return tmp_assignment

        K_split_indicies = self.find_K_split(cluster,indicies,K)

        for i in range(1,K):
            for c in K_split_indicies[i]:
                tmp_assignment[c] = self.K+i-1
        #Assert the correct labels
        unique_lables = np.unique(tmp_assignment)
        assert len(unique_lables) == self.K+K-1, "ERROR IN K SPLIT"
        assert (unique_lables == np.array(range(len(unique_lables)))).all(), "ERROR IN LABELLING"
        #Assert each subgraph is connected
        for k in range(len(unique_lables)):
            indicies = [i for i, x in enumerate(self.assignment) if x == k]
            if len(indicies)>1:
                assert nx.is_connected(self.W.subgraph(indicies)), "ERROR IN CONNECTIVITY"
        return tmp_assignment

    def find_islands(self,indicies,percentage):
        '''
        

        Parameters
        ----------
        cluster : 0 <= int <= self.K
            The numerical label of the cluster in which we are trying to find islands
        indicies : list
            List of indicies in self.assignment/self.X in the current cluster

        Returns
        -------
        islands : list
            List of indicies which could be islands.

        '''
        #find top and bottom 2.5% points in cluster, at least 1
        if percentage<1:
            number_of_islands = math.ceil(len(indicies)*percentage)
        else:
            number_of_islands = percentage
            if 2*number_of_islands > len(indicies):
                number_of_islands = int(len(indicies)/2)
        #Take norms of thetas
        theta_cluster = np.linalg.norm(self.theta[indicies,:],axis=1)
        #print(self.theta[indicies,:])
        #print(theta_cluster)
        islands = []
        if (number_of_islands < len(theta_cluster)):
            bottom_indicies = np.argpartition(theta_cluster, number_of_islands)[:number_of_islands]
            islands = islands+[indicies[i] for i in bottom_indicies]
            #print(bottom_indicies)
        #Get top and bottom indicies
        top_indicies = np.argpartition(theta_cluster, -number_of_islands)[-number_of_islands:]
        #print(top_indicies)

        islands = islands+[indicies[i] for i in top_indicies]
        return islands

    def find_split(self,cluster,indicies):
        '''
        Method to run K_split for K=2.

        Parameters
        ----------
        cluster : int
            numerical label of cluster.
        indicies : list
            current indicies of cluster in self.X/self.y.

        Returns
        -------
        clusters : list of lists
            list of list of indicies where each sub list is a subclutser.

        '''
        return self.find_K_split(2,cluster,indicies)

    def find_K_split_2(self,cluster,indicies,K):
        '''
        Parameters
        ----------
        cluster : int
            numerical label of cluster.
        indicies : list
            current indicies of cluster in self.X/self.y.
        K : int
            number of desired connected subclusters.

        Returns
        -------
        clusters : list of lists
            list of list of indicies where each sub list is a subclutser.

        '''
        distance_matrix = []
        for i in indicies:
            distance_1 = []
            for j in indicies:
                distance_1.append((1+np.linalg.norm(self.theta[i]-self.theta[j]))**(-1))
            distance_matrix.append(distance_1)
        W_k = nx.adjacency_matrix(self.W).toarray()[indicies,:][:,indicies]
        matrix = np.multiply(W_k,distance_matrix)
        subClustering = SpectralClustering(n_clusters=5,random_state=0,affinity="precomputed").fit(matrix)
        #subClustering = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(self.theta[indicies,:])
        #print(subClustering.labels_)
        clusters = []
        for label in range(K):
            indicies_new = [i for i, x in enumerate(subClustering.labels_) if x == label]
            cluster = []
            for index in indicies_new:
                cluster.append(indicies[index])
            clusters.append(cluster)
        for cluster in clusters:
            remainder_graph = self.W.subgraph(cluster)
            connected_components = nx.connected_components(remainder_graph)
            #print(list(connected_components))
            #input()
        return
    def find_K_split(self,cluster,indicies,K):
        '''
        Parameters
        ----------
        cluster : int
            numerical label of cluster.
        indicies : list
            current indicies of cluster in self.X/self.y.
        K : int
            number of desired connected subclusters.

        Returns
        -------
        clusters : list of lists
            list of list of indicies where each sub list is a subclutser.

        '''
        #Get adjacency matrix of the current clutser
        W = self.W_mat[indicies,:][:,indicies]
        
        subClustering = AgglomerativeClustering(K,connectivity=W).fit(self.theta[indicies,:])
        clusters = []
        for label in range(K):
            indicies_new = [i for i, x in enumerate(subClustering.labels_) if x == label]
            cluster = []
            for index in indicies_new:
                cluster.append(indicies[index])
            clusters.append(cluster)
        return clusters

    def find_distances_from_centres(self,indicies,v,connected_to):
        '''
        Method to find distance from area units to a given centre, if they are connected.

        Parameters
        ----------
        indicies : LIST
            List of indicies of current cluster that we are splitting.
        v : INT
            Current area unit in indicies which we are assigning to new centre.
        connected_to : LIST
            List of new clusters that v is connected to.

        Returns
        -------
        distances : LIST
            List of distances of v to new clusters.

        '''
        distances = []
        theta_1 = self.theta[indicies[v]]
        for index, connected_areas in enumerate(connected_to):
            if connected_to[index]>0:
                theta_2 = self.theta[indicies[index]]
                theta_distance = np.linalg.norm(theta_1-theta_2)
                distances.append(theta_distance)
            else:
                distances.append(np.inf)
        return distances

    def find_minimum_centres(self,centres,clusters,indicies):
        '''
        Method to find new centres by minimum intracluster distance

        Parameters
        ----------
        centres : LIST
            List of current centres.
        clusters : LIST
            List of current new clusters.
        indicies : LIST
            List of indicies in the choosen cluster we are trying to split.

        Returns
        -------
        centres : LIST
            New centres which each minimise their respective intracluster distance.

        '''
        #For each cluster choose point in cluster that minimies the intracluster distance
        centres = []
        for cluster in clusters:
            distances = []
            for point in cluster:
                distances2 = []
                theta_point = self.theta[indicies[point]]
                for other_point in cluster:
                    theta_other_point = self.theta[indicies[other_point]]
                    distances2.append(np.linalg.norm(theta_point-theta_other_point))
                distances.append(sum(distances2))
            centres.append(cluster[np.argmin(distances)])
        return centres

