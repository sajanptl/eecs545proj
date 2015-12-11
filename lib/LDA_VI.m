function [theta,z,beta,L,i]=LDA_VI(lambda,gamma,phi,W,alpha,eta,beta,...
    thresh,iter)

% This function applies LDA (latent Dirichlet allocation) with mean-field
% variational inference.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   N -- number of words in a document
%   V -- vocabulary size
%
% Input (can specify initial values):
%   lambda -- attached parameter to topics (K-by-V)
%   gamma -- attached parameter to per-document topic propotions (D-by-K)
%   phi -- attached parameter to per-word topic assignment (D-by-N-by-K)
%   W -- observed documents (D-by-N)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   eta -- parameter for K-dimensional Dirichlet distribution (1-by-1)
%   beta -- topic distribution among words (K-by-V)
%   thresh -- program stops when improvement on the evidence lower bound is
%             less than this value
%   iter -- program stops when the maximum number of iterations achieves
%           this value
%
% Output:
%   beta -- topic distribution among words (K-by-V)
%   theta -- per-document topic propotions (D-by-K)
%   z -- per-word topic assignment (D-by-N-by-K)
%   L -- the evidence lower bound w.r.t. iterations
%   i -- number of iterations
%
% Author: Zheng Luo
% Date: December 2015

% Run iterations with mean-field variational inference.
change=inf;
i=0;
L=ELB(gamma,phi,alpha,beta,W);
while(change>thresh)
    if(i>=iter)
        break;
    end
    [lambda,gamma,phi]=MFVar(lambda,gamma,phi,W,alpha,eta);
    % The (n)th entry is the lower bound after the (n-1)th iteration.
    L(i+2)=ELB(gamma,phi,alpha,beta,W);
    change=L(i+2)-L(i+1);
    i=i+1;
end

% Obtain the estimated posterior distribution.
theta=bsxfun(@times,gamma,1./sum(gamma,2));
z=phi;
beta=bsxfun(@times,lambda,1./sum(lambda,2));

end