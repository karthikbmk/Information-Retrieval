"""
Assignment 3. Implement a Multinomial Naive Bayes classifier for spam filtering.

You'll only have to implement 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

"""

from collections import defaultdict
import glob
import math
import os



class Document(object):
    """ A Document. Do not modify.
    The instance variables are:

    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename=None, label=None, tokens=None):
        """ Initialize a document either from a file, in which case the label
        comes from the file name, or from specified label and tokens, but not
        both.
        """
        if label: # specify from label/tokens, for testing.
            self.label = label
            self.tokens = tokens
        else: # specify from file.
            self.filename = filename
            self.label = 'spam' if 'spmsg' in filename else 'ham'
            self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):

    def get_word_probability(self, label, term):
        """
        Return Pr(term|label). This is only valid after .train has been called.

        Params:
          label: class label.
          term: the term
        Returns:
          A float representing the probability of this term for the specified class.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_word_probability('spam', 'a')
        0.25
        >>> nb.get_word_probability('spam', 'b')
        0.375
        """
        ###TODO
        return self.word_class_prob[label][term]

    def get_top_words(self, label, n):
        """ Return the top n words for the specified class, using the odds ratio.
        The score for term t in class c is: p(t|c) / p(t|c'), where c'!=c.

        Params:
          labels...Class label.
          n........Number of values to return.
        Returns:
          A list of (float, string) tuples, where each float is the odds ratio
          defined above, and the string is the corresponding term.  This list
          should be sorted in descending order of odds ratio.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_top_words('spam', 2)
        [(2.25, 'b'), (1.5, 'a')]
        """
        ###TODO
        all_labels = {'spam','ham'}
        other_label = list(all_labels - {label})[0]
        
        
        score_term = []
        
        for word in self.word_class_prob[label]:
            if self.word_class_prob[other_label][word]  !=  0:
                score = float(self.word_class_prob[label][word])/ (self.word_class_prob[other_label][word] )
                score_term.append((score,word))
        
        
        return sorted(score_term, key=lambda x: x[0], reverse=True)[0:n]
    
    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your
        book. Store these as instance variables, to be used by the classify
        method subsequently.
        Params:
          documents...A list of training Documents.
        Returns:
          Nothing.
        """
        ###TODO
                
        self.class_prior = defaultdict(lambda : 0.0)
        self.word_class_prob = defaultdict(lambda : defaultdict(lambda : 0.0))
        
        self.total_docs = 0.0
        self.class_no_of_docs = defaultdict(lambda : 0)
        
        all_labels = {'spam','ham'}
        class_total_toks = defaultdict(int)
        
        
        for doc in documents :
            
            #Prior class probs
            self.class_no_of_docs[doc.label] +=1
            self.total_docs += 1
            
            #Word conditional probs
            for token in doc.tokens:
                
                #+1 for correct class
                class_total_toks[doc.label] += 1
                self.word_class_prob[doc.label][token] +=1
                other_lbl = all_labels - {doc.label}
                
                #0 assignment for other class
                self.word_class_prob[list(other_lbl)[0]][token]
        
        #Get the total number of words in the vocabulary
        total_vocab_len = len(self.word_class_prob['spam'])
       
        
        for label in self.class_no_of_docs.keys():
            #Compute final priors
            self.class_prior[label] = float(self.class_no_of_docs[label]) / self.total_docs
        
            #Compute final word cond probs
            for token in self.word_class_prob[label].keys():
                self.word_class_prob[label][token] = float((self.word_class_prob[label][token] + 1))/ ( class_total_toks[label] + total_vocab_len)
        

    def classify(self, documents):
        """ Return a list of strings, either 'spam' or 'ham', for each document.
        Params:
          documents....A list of Document objects to be classified.
        Returns:
          A list of label strings corresponding to the predictions for each document.
        """
        ###TODO
                   
        
        predictions = []
        
        for doc in documents :
            spam_doc_score = math.log10(self.class_prior['spam'])  
            ham_doc_score = math.log10(self.class_prior['ham'])  
            
            for token in doc.tokens:

                spam_cond_prob = self.word_class_prob['spam'][token]
                ham_cond_prob = self.word_class_prob['ham'][token]
                
                if spam_cond_prob > 0:
                    spam_doc_score += math.log10(spam_cond_prob)
                
                if ham_cond_prob > 0 :
                    ham_doc_score += math.log10(ham_cond_prob)
                    
            if spam_doc_score > ham_doc_score:
                cls = 'spam'
            else:
                cls = 'ham'
            
            predictions.append(cls)
        
        return predictions
                
                

def evaluate(predictions, documents):
    """ Evaluate the accuracy of a set of predictions.
    Return a tuple of three values (X, Y, Z) where
    X = percent of documents classified correctly
    Y = number of ham documents incorrectly classified as spam
    X = number of spam documents incorrectly classified as ham

    Params:
      predictions....list of document labels predicted by a classifier.
      documents......list of Document objects, with known labels.
    Returns:
      Tuple of three floats, defined above.
    """
    ###TODO
    
    X = 0
    Y = 0.0
    Z = 0.0
    
    for d_id, doc in enumerate(documents):
        
        if doc.label == predictions[d_id]:
            X += 1
        
        elif doc.label == 'ham' and predictions[d_id] == 'spam':
            Y += 1
            
        elif doc.label == 'spam' and predictions[d_id] == 'ham':
            Z += 1
    
    X = float(X) / len(predictions)
    
    return (X,Y,Z)

def main():
    """ Do not modify. """
    if not os.path.exists('train'):  # download data        
        from urllib.request import urlretrieve
        import tarfile
        urlretrieve('http://cs.iit.edu/~culotta/cs429/lingspam.tgz', 'lingspam.tgz')
        tar = tarfile.open('lingspam.tgz')
        tar.extractall()
        tar.close()
    train_docs = [Document(filename=f) for f in glob.glob("train/*.txt")]
    print('read', len(train_docs), 'training documents.')
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(filename=f) for f in glob.glob("test/*.txt")]
    print('read', len(test_docs), 'testing documents.')
    predictions = nb.classify(test_docs)
    results = evaluate(predictions, test_docs)
    print('accuracy=%.3f, %d false spam, %d missed spam' % (results[0], results[1], results[2]))
    print('top ham terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('ham', 10)))
    print('top spam terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('spam', 10)))

if __name__ == '__main__':
    main()
