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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def string_to_token(s):
    """Convert a string to a list of tokens.
    Input: string
    Output: string
    The input string is converted to lowercase, non-letters and stop words are removed.
    """
    # Remove non-letters
    s = re.sub("[^a-zA-Z]", " ", s)

    # Convert to lower case, split into individual words
    words = s.lower().split()

    # Convert stopwords list to a set for fast searching
    ENGLISH_TOP_WORDS = set(stopwords.words("english"))

    # Remove stop words
    words = [w for w in words if not w in ENGLISH_TOP_WORDS]

    # Join words with spaces in between and return the string
    return " ".join(words)



num_line = 0
num_document = 0
tokens = []
with open('data/ap/ap.txt', 'r') as infile:
    for line in infile:
        num_line = num_line + 1
        if num_line % 6 == 4 and line.split() != []:   # the line with the document which is not empty
            num_document = num_document + 1
            # if len(line.split()) < 3:  # peek the short lines
            print 'Document ', num_document
            token = string_to_token(line)
            print token
            tokens.append(token)

vectorizer = CountVectorizer(max_df=0.95, min_df=20, stop_words='english')
vectorizer.fit(tokens)
X = vectorizer.transform(tokens)
sio.savemat('data/ap/LDA_input/W.mat',
            mdict={'W': X.todense()})

# Generate 3 matrices for LDA input

# WO is a W x 1 cell array of strings where WO{k} contains the kth vocabulary
# item and W is the number of distinct vocabulary items. Not needed for running
# the Gibbs sampler but becomes necessary when writing the resulting word-topic
# distributions to a file using the writetopics matlab function.
WO = vectorizer.get_feature_names()
# # save as txt
# np.savetxt('data/ap/LDA_input/WO.txt', WO, '%s')
sio.savemat('data/ap/LDA_input/WO.mat',
            mdict={'MO': WO},
            oned_as='column')


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

# save as txt
# np.savetxt('data/ap/LDA_input/DS.txt', DS, '%d')
# np.savetxt('data/ap/LDA_input/WS.txt', WS, '%d')
sio.savemat('data/ap/LDA_input/DS.mat',
            mdict={'DS': DS},
            oned_as='column')
sio.savemat('data/ap/LDA_input/WS.mat',
            mdict={'WS': WS},
            oned_as='column')


# Generate matrix used by variational inference LDA - ZWD
N = max(X.sum(axis = 1))[0][0]   # maximum number of tokens a document contains
D = X.shape[0]   # number of documents
ZWD = np.zeros([D, N])
DS = np.array(DS)
WS = np.array(WS)
for i in np.arange(D):
    # pdb.set_trace()
    row = WS[(DS == i+1).nonzero()]
    row.resize((1, N))
    ZWD[i,:] = row

sio.savemat('data/ap/LDA_input/ZWD.mat',
            mdict={'ZWD': ZWD},
            oned_as='column')
