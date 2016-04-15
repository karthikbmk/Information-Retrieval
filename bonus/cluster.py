"""
Assignment 5: K-Means. See the instructions to complete the methods below.
"""

from collections import Counter
import gzip
import math
from collections import defaultdict
import numpy as np
import sys
import time
import operator

class KMeans(object):
    
    def reverse_dict(self,dictionary):
        '''Revers k_v to v_k'''
        v_k = defaultdict(lambda : [])

        for k,v in dictionary.items():
            v_k[v].append(k)

        return v_k

    def __init__(self, k=2):
        """ Initialize a k-means clusterer. Should not have to change this."""
        self.k = k

    def cluster(self, documents, iters=10):
        """
        Cluster a list of unlabeled documents, using iters iterations of k-means.
        Initialize the k mean vectors to be the first k documents provided.
        After each iteration, print:
        - the number of documents in each cluster
        - the error rate (the total Euclidean distance between each document and its assigned mean vector), rounded to 2 decimal places.
        See Log.txt for expected output.
        The order of operations is:
        1) initialize means
        2) Loop
          2a) compute_clusters
          2b) compute_means
          2c) print sizes and error
        """
        ###TODO
        
        self.docs = documents        
        self.mean_vecs = documents[0:self.k]
        self.doc_cluster = defaultdict(lambda : int)
        
        
        for i in range(0,iters):                                        
            self.mean_norms = [self.sqnorm(mean) for mean in self.mean_vecs]   
            self.compute_clusters(documents)
            self.clust_docs = self.reverse_dict(self.doc_cluster) 
            self.mean_vecs = self.compute_means()
  
            print ([len(v) for v in self.clust_docs.values()])
            print (self.error(documents))

    def compute_means(self):
        """ Compute the mean vectors for each cluster (results stored in an
        instance variable of your choosing)."""
        ###TODO
        all_mean_vecs = []
        
        for doc_list in self.clust_docs.values():
            mean_vec = defaultdict(float)            
            for doc_id in doc_list:
                for k,v in self.docs[doc_id].items():
                    mean_vec[k] += v            
            total_docs_in_cluster = len(doc_list)
                                    
            g = lambda x : float(x)/total_docs_in_cluster            
            mean_vec = Counter({k: g(v) for k,v in mean_vec.items()})            
            all_mean_vecs.append(mean_vec)
        
        return all_mean_vecs
        
        
                    

    def compute_clusters(self, documents):
        """ Assign each document to a cluster. (Results stored in an instance
        variable of your choosing). """
        ###TODO
        for doc_id,doc in enumerate(documents):
            min_dist = sys.maxsize
            best_clust_id = -1
            for clust_id, mean in enumerate(self.mean_vecs):                      
                temp_dist = self.distance(doc,mean,self.mean_norms[clust_id])
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    best_clust = clust_id
            self.doc_cluster[doc_id] = best_clust                

    def sqnorm(self, d):
        """ Return the vector length of a dictionary d, defined as the sum of
        the squared values in this dict. """
        ###TODO
        sum = 0
        for x in d.values():
            sum += x**2        
        return sum

    def distance(self, doc, mean, mean_norm):
        """ Return the Euclidean distance between a document and a mean vector.
        See here for a more efficient way to compute:
        http://en.wikipedia.org/wiki/Cosine_similarity#Properties"""
        ###TODO

        asquare = self.sqnorm(doc)
        bsquare = mean_norm
     
        dist = 0.0
        
        for term in doc.keys():
            dist += (doc[term] * mean[term])
        
        return math.sqrt(asquare + bsquare - (2*dist))
        
    def error(self, documents):
        """ Return the error of the current clustering, defined as the total
        Euclidean distance between each document and its assigned mean vector."""
        ###TODO
        self.mean_vecs
        self.clust_docs
        
        total_dist = 0.0
        for clust_id,docs_in_clust in self.clust_docs.items():      
            
            mean_norm = self.sqnorm(self.mean_vecs[clust_id])
            
            for doc in docs_in_clust:                
                total_dist += self.distance(self.docs[doc],self.mean_vecs[clust_id],mean_norm)
                
        return round(total_dist,2)
        
            

    def print_top_docs(self, n=10):
        """ Print the top n documents from each cluster. These are the
        documents that are the closest to the mean vector of each cluster.
        Since we store each document as a Counter object, just print the keys
        for each Counter (sorted alphabetically).
        Note: To make the output more interesting, only print documents with more than 3 distinct terms.
        See Log.txt for an example."""
        ###TODO
        
        for clust_id, docs_in_clust in self.clust_docs.items():
            
            mean = self.mean_vecs[clust_id]
            mean_norm = self.sqnorm(mean)
            
            distances = defaultdict(float)
            
            for doc_id in docs_in_clust:
                if len(self.docs[doc_id]) > 3:
                    distances[doc_id] = self.distance(self.docs[doc_id],mean,mean_norm)
            
            sorted_docs = [tuple[0] for tuple in sorted(distances.items(), key=operator.itemgetter(1))][0:n]
            
            print ('CLUSTER ', clust_id)
            for d_id in sorted_docs:                   
                x = ' '.join(self.docs[d_id].keys())                                 
                print (x)
                    
            
            

def prune_terms(docs, min_df=3):
    """ Remove terms that don't occur in at least min_df different
    documents. Return a list of Counters. Omit documents that are empty after
    pruning words.
    >>> prune_terms([{'a': 1, 'b': 10}, {'a': 1}, {'c': 1}], min_df=2)
    [Counter({'a': 1}), Counter({'a': 1})]
    """
    ###TODO    
    
    #Term and its doc freqs are stored in this dict
    term_df = defaultdict(lambda:0)
    
    for profile in docs:
        for term in profile.keys():
            term_df[term] += 1

    final_lst = []
    
    for profile in docs:
        temp_cntr = Counter()
        for term in profile.keys():            
            if term_df[term] >= min_df:
                temp_cntr.update({term : profile[term]})
        if len(temp_cntr) > 0:
            final_lst.append(temp_cntr)
    
    return final_lst

def read_profiles(filename):
    """ Read profiles into a list of Counter objects.
    DO NOT MODIFY"""
    profiles = []
    with gzip.open(filename, mode='rt', encoding='utf8') as infile:    
        for line in infile:
            profiles.append(Counter(line.split()))
    return profiles


def main():
    profiles = read_profiles('profiles.txt.gz')
    print ('read', len(profiles), 'profiles.')
    profiles = prune_terms(profiles, min_df=2)    
    km = KMeans(k=10)    
    km.cluster(profiles, iters=20)
    km.print_top_docs()

if __name__ == '__main__':
    main()
