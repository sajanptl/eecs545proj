function [alpha_new,z,beta_new,L,i]=LDA_VI(lambda,gamma,phi,W,alpha,eta,...
    beta,thresh,iter)

% This function applies LDA (latent Dirichlet allocation) with mean-field
% variational inference.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   V -- vocabulary size
%
% Input (can specify initial values):
%   lambda -- attached parameter to topics (K-by-V)
%   gamma -- attached parameter to per-document topic propotions (D-by-K)
%   phi -- attached parameter to per-word topic assignment (D-by-V-by-K)
%   W -- observed documents (D-by-V)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   eta -- parameter for K-dimensional Dirichlet distribution (1-by-1)
%   beta -- topic distribution among words (K-by-V)
%   thresh -- program stops when improvement on the evidence lower bound is
%             less than this value
%   iter -- program stops when the maximum number of iterations achieves
%           this value
%
% Output:
%   alpha_new -- updated alpha (K-by-1)
%   z -- per-word topic assignment (D-by-V)
%   beta_new -- topic distribution among words (K-by-V)
%   L -- the evidence lower bound vs. iterations (including intial value)
%   i -- number of iterations
%
% Author: Z. Luo
% Date: December 2015

% Obtain the number of documents
D=size(W,1);
V=size(beta,2);

% Run iterations with mean-field variational inference.
change=inf;
i=0;
L=ELB(gamma,phi,alpha,beta);
while(change>thresh)
    if(i>=iter)
        break;
    end
    [lambda,gamma,phi]=MFVar(lambda,gamma,phi,W,alpha,eta);
    % The (n)th entry is the lower bound after the (n-1)th iteration.
    L(i+2)=ELB(gamma,phi,alpha,beta);
    change=abs(L(i+2)-L(i+1));
    i=i+1;
end

% Obtain posterior distributions.
alpha_new=sum(gamma,1)';
beta_new=bsxfun(@times,lambda,1./sum(lambda,2));
z=zeros(D,V);
for d=1:D
    idx=find(W(d,:));
    len=length(idx);
    for i=1:len
       [~,z_value]=max(phi(d,idx(i),:));
       z(d,i)=z_value;
    end
end

end