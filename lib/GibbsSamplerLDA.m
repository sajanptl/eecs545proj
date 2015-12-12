% Gibbs sampler for LDA
% EECS 545 Final Project
% Author: Xiang Li
% 12/10/2015

function [ WP,DP,Z ] = GibbsSamplerLDA( WS , DS , ZWD, T , N_round , ALPHA , BETA , SEED)
% Input:
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
% N_round - number of iterations
% 
% ALPHA - Parameter that defines the symmetric Dirichlet distribution for topic assignment
% 
% BETA - Parameter that defines the symmetric Dirichlet distribution for
% word generation
% 
% Output:
% WP - sparse matrix of size W x T, where WP(i,j) contains the number of 
% times word i has been assigned to topic j. (related with beta_k)   
% 
% DP - a sparse matrix of size D x T matrix, where DP(i,j) contains the 
% number of times a word in document d has been assigned to topic j. 
% 
% Z - topic assignment for token k.  

N = length(WS);     % N - total number of words 
W = max(WS);        % W - number of distinct words in volcabulary
D = length(ZWD);    % D - total number of documents

ND = sum(ZWD, 2);   % number of words in each document
NWD = zeros(W, D);  % times of word w occuring in document d (except current word)
for w = 1:W
    for d = 1:D
%         NWD(w, d) = sum( (WS == w) .* (DS == d) ) - 1;
        NWD(w, d) = sum(ZWD(d,:) == w);
    end
end

% randomly assign each token to one topic
Z = randi([1, T], 1, N)';

WP = zeros(W, T);
for i = 1:N         % for every word i
    w = WS(i);      % token represented by word i
    WP(w, Z(i)) = WP(w, Z(i)) + 1;
end

% update the Gibbs sampler
for i = 1:N_round
    Per = reassign(Z, WS, DS, ND, NWD, WP, T, ALPHA, BETA);
end

DP = zeros(D, T);

end



function Per = reassign( Z, WS, DS, ND, NWD, WP, T, ALPHA, BETA )

N = length(Z);
for i = 1:N   % for every word
    if (rem(i, 100) == 0)
        disp(sprintf('Reassigning word %d of %d', i, N))
    end
    p = zeros(T, 1);   % probability distribution for reassigning
    
    w = WS(i);    % token represented by word i
    d = DS(i);    % document d is the document containing word i
    
%     nd = sum(DS == d) - 1;   % number of words in document d (except current word) 
    nd = ND(d) - 1;   % number of words in document d (except current word) 
    
%     nwd = sum( (WS == w) .* (DS == d) ) - 1;   % times of word w occuring in document d (except current word)
    nwd = NWD(w,d) - 1;   % times of word w occuring in document d (except current word)

    for j = 1:T  % for every topic
%         nj = sum(Z == j) - 1;   % total number of words in topic j (except current word)
        nj = sum(WP(:, j));

%         nwj = sum( (WS == w) .* (Z == j) ) - 1;   % times of word w assigned to topic j (except current word)
        nwj = WP(w, j);

        p(j) = (nwd + BETA) / (nd + N*BETA) * (nwj + ALPHA) / (nj + T*ALPHA);
    end
    
    p = p / sum(p);
    Z(w) = randsample(1:T, 1, true, p);
end

% update WP
for i = 1:N         % for every word i
    w = WS(i);      % token represented by word i
    WP(w, Z(i)) = WP(w, Z(i)) + 1;
end

% calculate perplexity
Per = 0;

end
