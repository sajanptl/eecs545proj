function [alpha,beta]=MStep(W,alpha,phi,gamma,smooth,thresh,iter)

% This function is the maximization step of variational inference for LDA.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   V -- vocabulary size
%
% Input:
%   W -- observed documents (D-by-V)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   phi -- attached parameter to per-word topic assignment (K-by-D)
%   gamma -- attached parameter to per-document topic propotions (K-by-D)
%   smooth -- smoothing parameter (scalar)
%   thresh -- threshold for convergence
%   iter -- maximum number of iterations
%
% Output:
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   beta -- topic distributions among words in the vocabulary (K-by-V)
%
% Author: Z. Luo
% Date: December 2015

% Obtain basic parameter information.
[D,V]=size(W);
K=length(alpha);

% Estimate the value of beta.
beta=zeros(K,V);
for i=1:K
    beta(i,:)=sum(repmat(phi(i,:)',1,V).*W,1);
end
beta=beta+smooth;
beta=bsxfun(@times,beta,1./sum(beta,2));

% Newton-Raphson algorithm to update alpha.
i=0;
err=inf;
alpha_old=alpha;
while(err>thresh)
    if(i>=iter)
        break;
    end
    % Calculate the gradient and Hessian.
    temp=bsxfun(@plus,psi(gamma),-psi(sum(gamma,1)));
    gradient=D*(psi(sum(alpha_old))-psi(alpha_old))+sum(temp,2);
    Hessian=D*psi(1,sum(alpha_old))-D*diag(psi(1,alpha_old));
    % Update alpha.
    decrease=Hessian\gradient;
    alpha_new=alpha_old-decrease;
    j=0;
    while(~isempty(find(alpha_new<=0,1)))
        j=j+1;
        alpha_new=alpha_old-(0.9^j)*decrease;
    end
    % Calculate the error term.
    err=sum(abs(alpha_new-alpha_old))/sum(alpha_old);
    % Assign new values to old values.
    i=i+1;
    alpha_old=alpha_new;
end

% Assign the final estimated value.
alpha=alpha_new;

end