function [lambda_new,gamma_new,phi_new]=MFVar(lambda,gamma,phi,W,alpha,eta)

% This function applies one iteration of mean-field variational inference
% for LDA.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   V -- vocabulary size
%
% Input:
%   lambda -- attached parameter to topics (K-by-V)
%   gamma -- attached parameter to per-document topic propotions (D-by-K)
%   phi -- attached parameter to per-word topic assignment (D-by-V-by-K)
%   W -- observed documents in the form of bag of words (D-by-V)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   eta -- parameter for K-dimensional Dirichlet distribution (1-by-1)
%
% Output:
%   lambda_new -- updated lambda (K-by-V)
%   gamma_new -- updated gamma (D-by-K)
%   phi_new -- updated phi (D-by-V-by-K)
%
% Author: Z. Luo
% Date: December 2015

% Obtain basic parameter information.
K=length(alpha);
D=size(W,1);
V=size(W,2);

% Update gamma and phi.
indicator=zeros(D,V);
indicator(W~=0)=1;
gamma_new=gamma;
phi_new=phi;
for d=1:D
    % update gamma
    for k=1:K
        gamma_new(d,k)=alpha(k)+indicator(d,:)*phi(d,:,k)';
    end
    % update phi
    idx=find(W(d,:));
    for i=1:length(idx)
        phi_new(d,i,:)=exp(psi(gamma_new(d,:)')+psi(lambda(:,i))-...
            psi(sum(lambda,2)));
        % normalize phi
        phi_sum=sum(phi_new(d,i,:));
        if (phi_sum>0)
            phi_new(d,i,:)=phi_new(d,i,:)/phi_sum;
        end
    end
end

% Update lambda.
lambda_new=lambda;
for k=1:K
    lambda_new(k,:)=eta+sum(indicator.*phi_new(:,:,k),1);
end

end