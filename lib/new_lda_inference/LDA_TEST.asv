function [topic,perp]=LDA_TEST(Wtest,alpha,beta,thresh,iter)

% This function applies the learned posterior distribution to the test
% documents and discover the their topics.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   V -- vocabulary size
%
% Input:
%   Wtest -- test documents (D-by-V)
%   alpha -- learned alpha (K-by-1)
%   beta -- learned beta (K-by-V)
%   thresh -- threshold of convergence for estimation
%   iter -- maximum number of iterations for estimation
%
% Output:
%   topic -- topic distribution for each test document (K-by-D)
%   perp -- perplexity for the test documents (scalar)
%
% Author: Z. Luo
% Date: December 2015

% Estimate phi and gamma using the learned alpha and beta.
[phi,gamma]=Estep(Wtest,alpha,beta,thresh,iter);

% Obtain the perplexity and topic distribution.
[~,perp]=ELB(Wtest,gamma,phi,alpha,beta);
topic=phi;

end