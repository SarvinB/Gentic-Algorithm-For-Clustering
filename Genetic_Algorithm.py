import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import copy
import math

class GA:
    
    def __init__(self, iris, chromosomes, n, chromosome_n, parrent_number):
        self.chromosomes = chromosomes
        self.iris = iris
        self.n = n
        self.chromosome_n = chromosome_n
        self.parrent_number = parrent_number
            
    def parrent_choosing(self):
        # Randomly
        self.parrents_index =  np.random.randint(0,self.chromosome_n, self.parrent_number)
    
    # Cross-over for each pair of parrents    
    def uniform_cross_over(self, p1, p2, cuts):
        c_p1 = self.chromosomes[p1]
        c_p2 = self.chromosomes[p2]
        child = np.zeros((2, self.n+1))
        r = np.random.randint(1, self.n, cuts)
        r = np.append(0, r)
        r = np.append(r, self.n+1)
        r.sort()
        for i in range(1, len(r)):
            if i%2 == 0:
                child[0][r[i-1]:r[i]] = self.chromosomes[p1][r[i-1]:r[i]]
                child[1][r[i-1]:r[i]] = self.chromosomes[p2][r[i-1]:r[i]]
            else:
                child[0][r[i-1]:r[i]] = self.chromosomes[p2][r[i-1]:r[i]]
                child[1][r[i-1]:r[i]] = self.chromosomes[p1][r[i-1]:r[i]]
        return child
    
    def cross_over(self, cuts):
        child_chromosomes = []
        self.population = []
        i = 0
        while(i < self.parrent_number):
            childs = self.uniform_cross_over(self.parrents_index[i], self.parrents_index[i+1], cuts)
            child_chromosomes.append(childs[0])
            child_chromosomes.append(childs[1])
            i += 2
        child_chromosomes = np.array(child_chromosomes)
        self.population = np.concatenate((self.chromosomes, child_chromosomes), axis=0)

    def mutation(self, x):
        
        s = self.chromosome_n+self.parrent_number
        
        for i in range(s):
            
            cluster0 = [i for i, c in enumerate(self.population[i]) if c == 0]
            cluster1 = [i for i, c in enumerate(self.population[i]) if c == 1]
            cluster2 = [i for i, c in enumerate(self.population[i]) if c == 2]
            clusters = [cluster0, cluster1, cluster2]
            
            r = np.random.randint(0, self.n, 1)[0]
            mini = math.inf
            for c in range(3):
                e = self.group_dist(r, clusters[c], x)
                if e < mini:
                    self.population[i][r] = c
        
    # Calculate mean distance between data point to all points of a cluster    
    def group_dist(self, index, cluster, x):
        dist = 0
        k = 0
        for i in range(len(cluster)):
            if i != index:
                indexi = cluster[i]
                dist += np.linalg.norm(np.array(x[indexi])-np.array(x[index]))
                k += 1
        
        if k == 0:
            return 0       
        return dist/k
    
    def simularity(self, clusters, x):
        a = []
        for c in clusters:
            for i in c:
                a.append(self.group_dist(i, c, x))
                
        return a
                
    def dissimularity(self, clusters, x):
        b = []
        
        for i in clusters[0]:
            d1 = self.group_dist(i, clusters[1], x)
            d2 = self.group_dist(i, clusters[2], x)
            b.append(min(d1, d2))
            
        for i in clusters[1]:
            d1 = self.group_dist(i, clusters[0], x)
            d2 = self.group_dist(i, clusters[2], x)
            b.append(min(d1, d2))
            
        for i in clusters[2]:
            d1 = self.group_dist(i, clusters[0], x)
            d2 = self.group_dist(i, clusters[1], x)
            b.append(min(d1, d2))
            
        return b
    
    def objective_function_simularity(self, x, index):
        chromosome = self.population[index]
        cluster0 = [i for i, c in enumerate(chromosome) if c == 0]
        cluster1 = [i for i, c in enumerate(chromosome) if c == 1]
        cluster2 = [i for i, c in enumerate(chromosome) if c == 2]
        clusters = [cluster0, cluster1, cluster2]
        
        a = self.simularity(clusters, x)
        b = self.dissimularity(clusters, x)
        
        s = []
        for i in range(len(b)):
            s.append( (b[i]-a[i])/(max(a[i], b[i])) )
        
        return np.mean(np.array(s))
        
    
    def next_generation_choosing(self, x, y):
        scores = []
        p_size = self.chromosome_n+self.parrent_number

        for i in range(p_size):
            scores.append((self.objective_function_simularity(x, i), i))
        scores.sort(key=lambda tup: tup[0], reverse=True)
        
        winners_index = [c[1] for c in scores[:self.chromosome_n]]
        self.chromosomes = []
        for i in (winners_index):
            self.chromosomes.append(self.population[i])
            
    def winner(self, x, y):
        best_score = -1*math.inf
        index = -1

        for i in range(self.chromosome_n):
            s = self.objective_function_simularity(x, i)
            if s > best_score:
                best_score = s
                index = i
        
        accuracy = 0       
        for i in range(len(y)):
            if y[i] == (self.chromosomes[index][i]+1)%3:
                accuracy += 1
        accuracy /= len(y)
        
        
        return self.chromosomes[index], accuracy
                
        
            

        


        