""" Assignment 2
"""
import abc

import numpy as np


class EvaluatorFunction:
    """
    An Abstract Base Class for evaluating search results.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, hits, relevant):
        """
        Do not modify.
        Params:
          hits...A list of document ids returned by the search engine, sorted
                 in descending order of relevance.
          relevant...A list of document ids that are known to be
                     relevant. Order is insignificant.
        Returns:
          A float indicating the quality of the search results, higher is better.
        """
        return


class Precision(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute precision.

        >>> Precision().evaluate([1, 2, 3, 4], [2, 4])
        0.5
        """
        ###TODO
        tp = len( set(hits) & set(relevant))

        return float(tp) / len(hits)

    def __repr__(self):
        return 'Precision'


class Recall(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute recall.

        >>> Recall().evaluate([1, 2, 3, 4], [2, 5])
        0.5
        """
        ###TODO
        tp = len( set(hits) & set(relevant))
        fn = len(relevant) - tp

        denom = tp + fn

        return float(tp) / denom

    def __repr__(self):
        return 'Recall'


class F1(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute F1.

        >>> F1().evaluate([1, 2, 3, 4], [2, 5])  # doctest:+ELLIPSIS
        0.333...
        """
        ###TODO
        prec = Precision().evaluate(hits,relevant)
        rec =  Recall().evaluate(hits,relevant)

        num = 2 * prec * rec
        denom = prec + rec

        if denom == 0:
            return 0
        return float(num) /denom
    
    def __repr__(self):
        return 'F1'


class MAP(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute Mean Average Precision.

        >>> MAP().evaluate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 4, 6, 11, 12, 13, 14, 15, 16, 17])
        0.2
        """
        ###TODO        

        tp_cnt = 0
        numerator = 0

        set_rel = set(relevant)# to make things faster
        
        for _id,hit in enumerate(hits):            
            if hit in set_rel:
                tp_cnt += 1
                numerator += (float(tp_cnt) / (_id + 1))
            

        return float(numerator)/len(relevant)
        

    def __repr__(self):
        return 'MAP'

