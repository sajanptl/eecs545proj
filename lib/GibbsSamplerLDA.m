% Gibbs sampler for LDA
% EECS 545 Final Project
% Author: Xiang Li
% 12/10/2015

function [ WP,DP,Z ] = GibbsSamplerLDA( WS , DS , T , N , ALPHA , BETA , SEED , OUTPUT )
% Input:
%
% WS - N x 1 vector where WS(k) contains the vocabulary index of the kth
% word token, and N is the number of word tokens. The word indices are not zero
% based, i.e., min( WS )=1 and max( WS ) = W = number of distinct words in
% vocabulary
% 
% DS - N x 1 vector where DS(k) contains the document index of the kth word
% token. The document indices are not zero based, i.e., min( DS )=1 and max( DS )
% = D = number of documents
%
% T - number of topics
% 
% N - number of iterations
% 
% ALPHA - Parameter that defines the symmetric Dirichlet distribution for topic assignment
% 
% BETA - Parameter that defines the symmetric Dirichlet distribution for
% word generation


% Randomly assign each token to one topic
TS = randi([1, T], 1, N);

for i = 1:N
    reassign( TS, WS, DS, T);
end

end



function reassign( TS, WS, DS, T)

n_tokens = len(TS)
for w = 1:n_tokens
    p = zeros(T, 1);
    for j = 1:T
        % number of tokens in document d
        nd = sum(DS == d(w));

        % number of token w in document d
        nwd = sum(WS == w && DS == d(w));

        % number of all tokens in topic j
        nj = sum(TS == j);

        % number of token w in topic j
        nwj = sum(WS == w && TS == j);

        p[j] = nwd / nd * nwj / nj;
    end
    p = p / sum(p);
    TS(w) = randsample(1:T, 1, true, p);
end

end
