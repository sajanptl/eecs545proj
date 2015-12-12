function L=ELB(gamma,phi,alpha,beta)

% This function is used to calculate the evidence lower bound for LDA with
% variational inference.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   V -- vocabulary size
%
% Input:
%   gamma -- attached parameter to per-document topic propotions (D-by-K)
%   phi -- attached parameter to per-word topic assignment (D-by-V-by-K)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   beta -- topic distribution among words (K-by-V)
%
% Output:
%   L -- evidence lower bound for documents (1-by-1)
%
% Author: Z. Luo
% Date: December 2015

L1=gammaln(sum(alpha))-sum(gammaln(alpha))+...
     psi(sum(gamma,1)-psi(sum(sum(gamma))))*(alpha-1);

L2=(psi(sum(gamma,2))-psi(sum(sum(gamma))))*sum(sum(phi,2),3);    

%phi_kn=sum(phi,3)';
phi_kn=sum(phi,1)';    
L3=sum(beta.*phi_kn,1);

L4=-gammaln(sum(sum(gamma)))+sum(gammaln(sum(gamma)))+...
      (sum(gamma,1)-1)*psi(sum(gamma,1)-psi(sum(sum(gamma))))';

L5=-sum(sum(sum(phi.*log(phi))));

L=L1+L2+L3+L4+L5;

end