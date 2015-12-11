function L=ELB(gamma,phi,alpha,beta,W)

% This function is used to calculate the evidence lower bound for LDA with
% variational inference.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   N -- number of words in a document
%   V -- vocabulary size
%
% Input:
%   gamma -- attached parameter to per-document topic propotions (D-by-K)
%   phi -- attached parameter to per-word topic assignment (D-by-N-by-K)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   beta -- topic distribution among words (K-by-V)
%   W -- observed documents (D-by-N)
%
% Output:
%   L -- evidence lower bound for documents (D-by-1)
%
% Author: Zheng Luo
% Date: December 2015

% Obtain basic parameter information.
D=size(W,1);
V=size(beta,2);
phi=permute(phi,[2 3 1]);
my_buffer=bsxfun(@plus,psi(gamma),-psi(sum(gamma,2)));

% Part 1 of the evidence lower bound.
buffer=my_buffer;
buffer=sum(buffer*(alpah-1));
L1=D*(gammaln(sum(alpha))-sum(gammaln(alpha)))+buffer;

% Part 2 of the evidence lower bound.
buffer=my_buffer;
L2=0;
for d=1:D
    L2=L2+sum(phi(:,:,d).*buffer(d,:)');
end

% Part 3 of the evidence lower bound.
L3=0;
beta=log(beta);
for v=1:V
    [idx1,idx2]=find(W==v);
    len=length(idx1);
    for d=1:len
        L3=L3+sum(phi(idx2,:,idx1(d))*beta(:,v));
    end
end

% Part 4 of the evidence lower bound.
buffer=sum(sum((gamma-1).*my_buffer));
L4=-sum(gammaln(sum(gamma,2)))+sum(sum(gammaln(gamma)))-buffer;

% Part 5 of the evidence lower bound.
L5=-sum(sum(sum(phi.*log(phi))));

% Obtain the evidence lower bound.
L=L1+L2+L3+L4+L5;

end