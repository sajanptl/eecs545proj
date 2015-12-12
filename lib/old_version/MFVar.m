function [lambda,gamma,phi]=MFVar(lambda,gamma,phi,W,alpha,eta)

% This function applies one iteration of mean-field variational inference
% for LDA.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   N -- number of words in a document
%   V -- vocabulary size
%
% Input:
%   lambda -- attached parameter to topics (K-by-V)
%   gamma -- attached parameter to per-document topic propotions (D-by-K)
%   phi -- attached parameter to per-word topic assignment (D-by-N-by-K)
%   W -- observed documents (D-by-N)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   eta -- parameter for K-dimensional Dirichlet distribution (1-by-1)
%
% Output:
%   lambda -- attached parameter to topics (K-by-V)
%   gamma -- attached parameter to per-document topic propotions (D-by-K)
%   phi -- attached parameter to per-word topic assignment (D-by-N-by-K)
%
% Author: Zheng Luo
% Date: December 2015

% Obtain basic parameter information.
K=length(alpha);
D=size(W,1);
N=size(W,2);
V=size(lambda,2);

% Update the variable lambda.
for k=1:K
    for v=1:V
        indicator=zeros(D,N);
        indicator(W==v)=1;
        lambda(k,v)=eta+sum(sum(indicator.*phi(:,:,k)));
    end
end

% Update the variables gamma and phi.
for d=1:D
    % Update gamma.
    for k=1:K
        gamma(d,k)=alpha(k)+sum(phi(d,:,k));
    end
end
for d=1:D
    % Update phi.
    for n=1:N
        if(W(d,n)==0)
            continue;
        else
            idx=W(d,n);
        end
        phi(d,n,:)=exp(psi(gamma(d,:)')+psi(lambda(:,idx))-...
            psi(sum(lambda,2)));
        phi_sum=sum(phi(d,n,:));
        if(phi_sum>0)
            phi(d,n,:)=phi(d,n,:)/phi_sum;
        end
    end
end

end