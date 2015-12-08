function [gamma,phi,lambda]=var_infer(alpha,beta,lambda_ini,thresh,iter)

% This function applys variational inference.
%
% Terminology:
%   K = number of topics
%   N = number of words in a document
%   V = vocabulary size
%   M = corpus size
%
% Input:
%   alpah -- proportions parameter (K-by-1)
%   eta -- topic parameter
%   lambda_ini -- the initial value of lambda
%   thresh -- threshold of loss to stop the optimization
%   iter -- maximum number of loops
%
% Output:
%   gamma -- attached parameter to per-document topic propotions
%   phi -- attached parameter to per-word topic assignment (K-by-M)
%   lambda -- attached parameter to topics (K-by-M)
%
% Author: Z. Luo
% Date: December 2015

% Calculate the initial loss.


% Start the iteration.



%% E Step


[K,V]=size(beta);
Ns=sum(X,2); 
M=size(X,1);

phi_t=ones(K,M)/K;
gama_t=alpha*ones(1,M)+ones(K,1)*Ns'/K;


epsilon=0.001;
time=500;

e=100;
t=1;

while e>epsilon && t<time
    % phi
    temp=gama_t-ones(K,1)*sum(gama_t,1)+(X*log(beta'+realmin)./(Ns*ones(1,K)))';
    maxtemp=max(temp,[],1);
    phi_tt=exp(temp+ones(K,1)*abs(maxtemp));
    clear temp maxtemp;
    phi_tt=phi_tt./(ones(K,1)*sum(phi_tt,1));
    
    % gama
    gama_tt=alpha*ones(1,M)+phi_tt.*(ones(K,1)*Ns');
   
    e1=sum(sum(abs(phi_tt-phi_t)))/sum(sum(phi_t));
    e2=sum(sum(abs(gama_tt-gama_t)))/sum(sum(gama_t));
    e=max(e1,e2);
    
    phi_t=phi_tt;
    gama_t=gama_tt;

    t=t+1;
end









%% M Step




[K,M]=size(phi);
[M,V]=size(X);

% get beta
for i=1:K
    tempphi=phi(i,:)'*ones(1,V);
    beta(i,:)=sum(tempphi.*X,1);
end
beta=beta+lap;
beta=beta./(sum(beta,2)*ones(1,V));


%get alpha
alpha_t=alpha;
epsilon=0.001;
time=500;

t=0;
e=100;
psiGama=psi(gama);
psiSumGama=psi(sum(gama,1));
while e>epsilon&&t<time
    g=sum((psiGama-ones(K,1)*psiSumGama),2)+M*(psi(sum(alpha_t))-psi(alpha_t));
    h=-M*psi(1,alpha_t);
    z=M*psi(1,sum(alpha_t));
    c=sum(g./h)/(1/z+sum(1./h));
    delta=(g-c)./h;

    eta=1;
    alpha_tt=alpha_t-delta;
    while (length(find(alpha_tt<=0))>0)
        eta=eta/2;
        alpha_tt=alpha_t-eta*delta;
    end
    e=sum(abs(alpha_tt-alpha_t))/sum(alpha_t);
    
    alpha_t=alpha_tt;

    t=t+1;
end
alpha=alpha_t;






