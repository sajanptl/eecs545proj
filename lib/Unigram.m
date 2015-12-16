% EECS 545 Final Project
% Sajan Patel (sajanptl)
% December 2015

function [Ptrain, Ptest, PPlexTrain, PPlexTest, WplexTrain, WplexTest] = Unigram(Wtrain, Wtest, lambda)
% Implements a unigram model on the documents
%
% Input:
% Wtrain - a D1 x V matrix of counts that each word w in the vocabulary of size 
%          V appears in the ith document in the set of D1 training documents.
%
% Wtrain - a D2 x V matrix of counts that each word w in the vocabulary of size 
%          V appears in the ith document in the set of D2 testing documents.
%
% lambda - smoothing term
%
% Output:
% Ptrain - a D1 by V matrix of the probabilities of each word appearing in the 
%          D1 training documents. Note that it's the same row of V probabilities 
%          (one for each word) repeated D1 times for convenience of mapping 
%          documents to their words' probabilities.
%
% Ptest -  a D2 by V matrix of the probabilities of each word appearing in the 
%          D2 testing documents, taking into account the smoothing term lambda.
%          Let I(x) = 1 if x > 0, or 0 otherwise. Ptest is calculated as follows:
%          Ptest(i,j) = (lambda*I(Wtest(i,j))*Ptrain(i,j)) + ((1-lambda)/V).
% 
% PPlexTrain - the total perplexity of the entire training set (a scalar)
%
% PPlexTest  - the total perplexity of the entire testing set (a scalar)
% 
% WplexTrain - a vector of the perplexities of each word in the training set
%
% WplexTest - a vector of the perplexities of each word in the testing set

%% train the model
disp('Training Model');
[ntrain, v] = size(Wtrain);
wCnt = sum(Wtrain); % create a row vector of sums for each row down the column of W
tot = sum(wCnt); % get the total counts for the training set
pword_row = wCnt / tot; % each word's probability is its column count over the total cout

%% training perplexity calculation
Ptrain = repmat(pword_row, ntrain, 1);
log2ptrain = -log2(Ptrain);
Htrain = sum(sum(log2ptrain));
%PPlexTrain = 2^(Htrain / v);
%WplexTrain = 2.^(sum(log2ptrain) / v);
PPlexTrain = Htrain / v;
WplexTrain = sum(log2ptrain) / v;

%% test the model
disp('Testing Model');
[ntest, vt] = size(Wtest);
PwordTest = repmat(pword_row, ntest, 1);
Ptest = (lambda * (double(Wtest > 0).*PwordTest)) + ((1 - lambda) / v);

%% training perplexity calculation
log2ptest = -log2(Ptest);
Htest = sum(sum(log2ptest));
%PPlexTest = 2^(Htest / v);
%WplexTest = 2.^(sum(log2ptest) / v);
PPlexTest = Htest / v;
WplexTest = sum(log2ptest) / v;

end
