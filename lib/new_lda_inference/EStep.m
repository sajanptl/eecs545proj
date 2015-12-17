function [phi,gamma]=EStep(W,alpha,beta,thresh,iter)

% This function is the estimation step of variational inference for LDA.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   V -- vocabulary size
%   N -- number of words for each document (D-by-1)
%
% Input:
%   W -- observed documents (D-by-V)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   beta -- topic distributions among words in the vocabulary (K-by-V)
%   thresh -- threshold for convergence
%   iter -- maximum number of iterations
%
% Output:
%   phi -- attached parameter to per-word topic assignment (K-by-D)
%   gamma -- attached parameter to per-document topic propotions (K-by-D)
%
% Author: Z. Luo
% Date: December 2015

% Obtain basic parameter information.
[D,~]=size(W);
K=length(alpha);
N=sum(W,2);
if(~isempty(find(N==0,1)))
    error(['At leaset one document has no words in the user-defined '...
        'vocabulary (i.e. represented as zeros). Please check and '...
        'remove it/them.']);
end

% Initialize parameters.
phi_old=ones(K,D)/K;
gamma_old=repmat(alpha,1,D)+repmat(N',K,1)/K;

% Iterations until convergence (or maximum number of iterations reached).
i=0;
err=inf;
while(err>thresh)
    if(i>=iter)
        break;
    end
    % Update phi.
    temp1=bsxfun(@plus,gamma_old,-sum(gamma_old,1));
    if(length(find(beta==0))>=1)
        temp2=W*log(beta'+realmin)./repmat(N,1,K);
    else
        temp2=W*log(beta')./repmat(N,1,K);
    end
    temp=temp1+temp2';
    temp3=repmat(abs(max(temp,[],1)),K,1);
    phi_new=exp(temp+temp3);
    phi_new=bsxfun(@times,phi_new,1./sum(phi_new,1));
    % Update gamma.
    gamma_new=repmat(alpha,1,D)+phi_new.*repmat(N',K,1);
    % Calculate the error term.
    err1=sum(sum(abs(phi_new-phi_old)))/sum(sum(phi_old));
    err2=sum(sum(abs(gamma_new-gamma_old)))/sum(sum(gamma_old));
    err=max(err1,err2);
    % Assign the new values to old values.
    phi_old=phi_new;
    gamma_old=gamma_new;
    i=i+1;
end

% Assign the final estimated values.
phi=phi_new;
gamma=gamma_new;

end