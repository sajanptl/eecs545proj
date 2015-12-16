% Gibbs sampler for LDA
% EECS 545 Final Project
% Author: Xiang Li
% 12/10/2015

function [ WP,DP,Z,Per ] = GibbsSamplerLDA( WS , DS , ZWD, WO, WS_test, ...
    DS_test, ZWD_test, T , N_round , ALPHA , BETA)
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
% ALPHA - Parameter that defines the symmetric Dirichlet distribution for 
%         topic assignment
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

[N, W, D] = prepare_data(WS, ZWD, WO);
[N_test, W_test, D_test] = prepare_data(WS_test, ZWD_test, WO);


% randomly assign each token to one topic
Z = randi([1, T], 1, N)';

WP = zeros(W, T);
DP = zeros(D, T);
for i = 1:N         % for every word i
    w = WS(i);      % token represented by word i
    WP(w, Z(i)) = WP(w, Z(i)) + 1;
    DP(DS(i), Z(i)) = DP(DS(i), Z(i));
end

Per = zeros(N_round, 1);
for i = 1:N_round
    disp(sprintf('N_round = %d', i))
    % update the Gibbs sampler
    [Z, WP, DP] = reassign(Z, WS, DS, WP, DP, T, ALPHA, BETA);  
 
    % Calculate perplexity
%     Per(i) = perplexity(WS_test, DS_test, ND_test, NWD_test, WP, T, ALPHA , BETA);   
    
%     disp(sprintf('Perplexity = %f', Per(i)))
end


end

%%
function [N, W, D] = prepare_data(WS, ZWD, WO)

N = length(WS);     % N - total number of words
W = length(WO);     % W - number of distinct words in volcabulary
D = size(ZWD, 1);    % D - total number of documents

% % number of words (tokens) in each document
% ND = sum(ZWD ~= 0, 2);   
% % times of word w occuring in document d
% NWD = zeros(K, D);  
% for w = 1:W
%     for d = 1:D
%         NWD(w, d) = sum(ZWD(d,:) == w);
%     end
% end

end

%%
function [Z_new, WP_new, DP_new] = reassign( Z, WS, DS, WP, DP, T, ALPHA, BETA )

N = length(Z);
Z_new = zeros(N, 1);
for i = 1:N   % for every word
    p = zeros(T, 1);   % probability distribution for reassigning

    w = WS(i);    % token represented by word i
    d = DS(i);    % document d is the document containing word i
%     nd = ND(d) - 1;   % number of words in document d (except current word)
%     % times of token w occuring in document d (except current word)
%     nwd = NWD(w,d) - 1;   
    
%     for j = 1:T  % for every topic
%         % total number of words in topic j (except current word)
%         nj = sum(WP(:, j)) - (Z(i)==j);  
%         % times of token w assigned to topic j (except current word)
%         nwj = WP(w, j) - (Z(i)==j);   
%         p(j) = (nwd + BETA) / (nd + N*BETA) * (nwj + ALPHA) / (nj + T*ALPHA);
%     end
    
    % vectorized version
    Zi = Z(i) == (1:T);
    % total number of words in topic j (except current word)
    nj = sum(WP) - Zi;
    % times of token w assigned to topic j (except current word)
    nwj = WP(w,:) - Zi;
%     p = (nwd + BETA) / (nd + N*BETA) * (nwj + ALPHA) ./ (nj + T*ALPHA);
    dp = DP(d,:)- Zi;
    p = (dp + ALPHA) .* (nwj + BETA) ./ (nj + T*BETA);
    
    p = p / sum(p);
    if (min(p) < 0)
        p
    end
    Z_new(i) = randsample(1:T, 1, true, p);
end

% update WP
WP_new = zeros(size(WP));
DP_new = zeros(size(DP));
for i = 1:N         % for every word i
    w = WS(i);      % token represented by word i
    WP_new(w, Z_new(i)) = WP_new(w, Z_new(i)) + 1;
    DP_new(DS(i), Z_new(i)) = DP_new(DS(i), Z_new(i)) + 1;
end

end

%%
function per = perplexity( WS_test, DS_test, WP, DP, T, ALPHA , BETA)

N = length(WS_test);
per = 0;
for i = 1:N            % for every word
    w = WS_test(i);    % token represented by word i
    d = DS_test(i);    % document d is the document containing word i
    
    % vectorize
    nj = sum(WP);
    nwj = WP(w, :);
    dp = DP(d,:);
    ndp = 
    p = log10( (dp + T*ALPHA) ./ (dp + ALPHA).* ...
        (nwj + BETA) ./ (nj + T*BETA) );

    per = per + sum(p);
end
per = 10^(-per/N);

end
