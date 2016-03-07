""" Assignment 2
"""
import abc
from collections import defaultdict
import math

import index


def idf(term, index):
    """ Compute the inverse document frequency of a term according to the
    index. IDF(T) = log10(N / df_t), where N is the total number of documents
    in the index and df_t is the total number of documents that contain term
    t.

    Params:
      terms....A string representing a term.
      index....A Index object.
    Returns:
      The idf value.

    >>> idx = index.Index(['a b c a', 'c d e', 'c e f'])
    >>> idf('a', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('d', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('e', idx) # doctest:+ELLIPSIS
    0.176...
    """
    ###TODO
    
    return math.log10((len(index.documents)*1.0)/index.doc_freqs[term])



class ScoringFunction:
    """ An Abstract Base Class for ranking documents by relevance to a
    query. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def score(self, query_vector, index):
        """
        Do not modify.

        Params:
          query_vector...dict mapping query term to weight.
          index..........Index object.
        """
        return


class RSV(ScoringFunction):
    """
    See lecture notes for definition of RSV.

    idf(a) = log10(3/1)
    idf(d) = log10(3/1)
    idf(e) = log10(3/2)
    >>> idx = index.Index(['a b c', 'c d e', 'c e f'])
    >>> rsv = RSV()
    >>> rsv.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.4771...
    """

    def score(self, query_vector, index):
        ###TODO

        doc_rsv = defaultdict(lambda : 0.0)
        
        for _id,doc in enumerate(index.documents):
            q_d_matches = set(query_vector.keys()) &  set(doc)

            rsv = 0

            for term in q_d_matches:
                rsv += idf(term,index)

            doc_rsv[_id+1] = rsv

        return doc_rsv          
        
        
        

    def __repr__(self):
        return 'RSV'


class BM25(ScoringFunction):
    """
    See lecture notes for definition of BM25.

    log10(3) * (2*2) / (1(.5 + .5(4/3.333)) + 2) = log10(3) * 4 / 3.1 = .6156...
    >>> idx = index.Index(['a a b c', 'c d e', 'c e f'])
    >>> bm = BM25(k=1, b=.5)
    >>> bm.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.61564032...
    """
    def __init__(self, k=1, b=.5):
        self.k = k
        self.b = b

    def score(self, query_vector, index):
        ###TODO
        
        doc_bm25 = defaultdict(lambda : 0.0)
        
        for _id,doc in enumerate(index.documents):
            q_d_matches = set(query_vector.keys()) &  set(doc)

            bm25 = 0

            for term in q_d_matches:
                
                tf_in_doc = -1

                for d_id ,tf in index.index[term]:
                    if d_id == _id + 1:
                        tf_in_doc = tf
                        break
                    
                if tf_in_doc != -1:
                    num = (self.k + 1) * tf_in_doc * idf(term,index)
                    x2 = float(len(doc))/index.mean_doc_length
                    x2 = self.b*x2
                    den = 1-self.b
                    den = den + x2
                    den = den*self.k
                    den = den + tf_in_doc
                    

                    bm25 += float(num)/den
                else:
                    return 'ERRROR !!!!!!!!!!'
                

            doc_bm25[_id+1] = bm25

        return doc_bm25          
        
        
        

    def __repr__(self):
        return 'BM25 k=%d b=%.2f' % (self.k, self.b)


class Cosine(ScoringFunction):
    """
    See lecture notes for definition of Cosine similarity.  Be sure to use the
    precomputed document norms (in index), rather than recomputing them for
    each query.

    >>> idx = index.Index(['a a b c', 'c d e', 'c e f'])
    >>> cos = Cosine()
    >>> cos.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.792857...
    """
    def score(self, query_vector, index):
        ###TODO
        
        sim_dict = defaultdict(lambda : 0)

        N = len(index.documents)

        for _id, doc in enumerate(index.documents):
            numerator = 0

            for term, freq in query_vector.items():
                tf_in_doc = -1

                for d_id ,tf in index.index[term]:
                    if d_id == _id + 1:
                        tf_in_doc = tf
                        break
                    
                if tf_in_doc != -1:                    
                    doc_tf_idf = (1 + math.log10(tf_in_doc)) * math.log10(float(N)/ index.doc_freqs[term])
                    numerator += (freq * doc_tf_idf)

            sim_dict[_id+1] = float(numerator)/ index.doc_norms[_id + 1]

        return sim_dict

    def __repr__(self):
        return 'Cosine'
