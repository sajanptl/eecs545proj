function [alpha_new,beta_new,phi_new,gamma_new,perp,L,i]=LDA_VI(W,...
    alpha,beta,smooth,thresh1,iter1,thresh2,iter2,thresh,iter)

% This function applies LDA (latent Dirichlet allocation) with mean-field
% variational inference.
%
% Terminology:
%   K -- number of topics
%   D -- number of documents
%   V -- vocabulary size
%
% Input:
%   W -- observed documents (D-by-V)
%   alpha -- parameter for V-dimensional Dirichlet distribution (K-by-1)
%   beta -- topic distributions among words in the vocabulary (K-by-V)
%   smooth -- smoothing parameter for estimating beta at Mstep (scalar)
%   thresh1 -- stopping threshold for EStep (scalar)
%   iter1 -- maximum number of iterations for EStep (integer)
%   thresh2 -- stopping threshold for MStep (scalar)
%   iter2 -- maximum number of iterations for MStep (integer)
%   thresh -- stopping threshold for combinations of EM (scalar)
%   iter -- maximum number of iterations for EM (integer)
%
% Output:
%   alpha_new -- updated alpha (K-by-1)
%   beta_new -- updated beta (K-by-V)
%   phi_new -- updated phi (K-by-D)
%   gamma_new -- updated gamma (K-by-D)
%   perp -- the perplexity vs. iterations (including intial value)
%   L -- the evidence lower bound vs. iterations (including intial value)
%   i -- number of iterations of combinations of EM (integer)
%
% Author: Z. Luo
% Date: December 2015

% Run EM algorithms for multiple iterations.
err=inf;
i=0;
L=[];
perp=[];
L_old=1; % a pseudo initial value
while(err>thresh)
    if(i>=iter)
        break;
    end
    % Estimation
    [phi,gamma]=EStep(W,alpha,beta,thresh1,iter1);
    % Maximization
    [alpha,beta]=MStep(W,alpha,phi,gamma,smooth,thresh2,iter2);
    % Evidence Lower Bound
    [L_new,perp_new]=ELB(W,gamma,phi,alpha,beta);
    % Store the output values.
    L=[L,L_new];
    perp=[perp,perp_new];
    % Calculate the error term.
    err=abs(L_new-L_old)/abs(L_old);
    L_old=L_new;
    i=i+1;
end

% Assign final values.
alpha_new=alpha;
beta_new=beta;
phi_new=phi;
gamma_new=gamma;

end