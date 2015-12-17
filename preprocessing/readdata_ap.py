# EECS 545 Final Project - LDA
# Read Sample AP Dataset (txt file) and convert it the format usable by the LDA API
# 12/10/2015
# Xiang Li

import pdb

import numpy as np
import pandas as pd
import scipy.io as sio
import re
from scipy.sparse import coo_matrix, csr_matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

ENGLISH_STOP_WORDS = stopwords.words("english")
for w in ['would', 'also', 'said', 're']:
    ENGLISH_STOP_WORDS.append(w)

def string_to_token(s):
    """Convert a string to a list of tokens.
    Input: string
    Output: string
    The input string is converted to lowercase, non-letters and stop words are removed.
    """
    # Remove non-letters
    s = re.sub("[^a-zA-Z]", " ", s)
    # # Convert to lower case, split into individual words
    # words = s.lower().split()
    # # Remove stop words
    # words = [w for w in words if not w in set(ENGLISH_STOP_WORDS)]
    # Join words with spaces in between and return the string
    # return " ".join(words)
    return s


def ap_to_tokens(filename):
    """Convert AP dataset to a tokens
    Input: filename of the AP dataset (.txt)
    Output: list of strings
    """
    num_line = 0
    num_document = 0
    tokens = []
    with open(filename, 'r') as infile:
        for line in infile:
            num_line = num_line + 1
            if num_line % 6 == 4 and line.split() != []:   # the line with the document which is not empty
                num_document = num_document + 1
                # if len(line.split()) < 3:  # peek the short lines
                print 'Document ', num_document
                token = string_to_token(line)
                print token
                tokens.append(token)
    return tokens


def vectorize(tokens, dirname):
    """ vectorize the tokens
    Input: list of strings
    Output: count vector
    The volcabulary is built and saved in .mat
    """
    vectorizer = CountVectorizer(max_df=0.9,
                                min_df=20,
                                stop_words=ENGLISH_STOP_WORDS)
    X = vectorizer.fit_transform(tokens)
    # WO is a W x 1 cell array of strings where WO{k} contains the kth vocabulary
    # item and W is the number of distinct vocabulary items. Not needed for running
    # the Gibbs sampler but becomes necessary when writing the resulting word-topic
    # distributions to a file using the writetopics matlab function.
    WO = vectorizer.get_feature_names()
    sio.savemat(dirname + 'WO.mat',
                mdict={'WO': WO},
                oned_as='column')
    # # save as txt
    # np.savetxt('data/ap/LDA_input/WO.txt', WO, '%s')
    return X


def split_data(X):
    """split the data into training set and test set
    """
    Xtrain, Xtest = train_test_split(X, train_size = 0.8)
    return Xtrain, Xtest


def save_data(X, appendix, dirname):
    """save X, WS, DS
    """
    # save X
    Xdense = X.todense()                            # convert X to dense matrix
    Xrow_sum = np.ravel(np.sum(Xdense, axis = 1))   # sum of each row
    nonzero_index = np.ravel(np.nonzero(Xrow_sum))  # find index of nonzeros
    Xdense_nonzero = Xdense[nonzero_index, :]       # get nonzero rows
    sio.savemat(dirname + 'W' + appendix + '.mat',
                mdict={'W'+appendix: Xdense_nonzero})
    # save as sparse matrix not working
    # sio.savemat(dirname + 'W' + appendix + '.mat',
    #             mdict={'W'+appendix: X})

    # WS is a N x 1 vector where WS(k) contains the vocabulary index of the kth
    # word token, and N is the number of word tokens. The word indices are not zero
    # based, i.e., min( WS )=1 and max( WS ) = W = number of distinct words in
    # vocabulary
    # DS is a N x 1 vector where DS(k) contains the document index of the kth word
    # token. The document indices are not zero based, i.e., min( DS )=1 and max( DS )
    # = D = number of documents
    DS = []
    WS = []
    Xcoo = coo_matrix(X)
    for i, j, v in zip(Xcoo.row, Xcoo.col, Xcoo.data):
        # i is document index, j is word index, v is word count
        for iv in np.arange(v):
            DS.append(i+1)
            WS.append(j+1)
    sio.savemat(dirname + 'DS' + appendix + '.mat',
                mdict={'DS'+appendix: DS},
                oned_as='column')
    sio.savemat(dirname + 'WS' + appendix + '.mat',
                mdict={'WS'+appendix: WS},
                oned_as='column')

    # # Generate matrix used by variational inference LDA - ZWD
    # N = max(X.sum(axis = 1))[0][0]   # maximum number of tokens a document contains
    # D = X.shape[0]   # number of documents
    # ZWD = np.zeros([D, N])
    # DS = np.array(DS)
    # WS = np.array(WS)
    # for i in np.arange(D):
    #     row = WS[(DS == i+1).nonzero()]
    #     row.resize((1, N))
    #     ZWD[i,:] = row
    # sio.savemat(dirname + 'ZWD' + appendix + '.mat',
    #             mdict={'ZWD'+appendix: ZWD},
    #             oned_as='column')


if __name__ == "__main__":
    tokens = ap_to_tokens('data/ap/ap.txt')
    X = vectorize(tokens, 'data/ap/LDA_input/')
    Xtrain, Xtest = split_data(X)
    save_data(Xtrain, '', 'data/ap/LDA_input/')
    save_data(Xtest, '_test', 'data/ap/LDA_input/')
