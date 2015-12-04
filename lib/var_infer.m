function [gamma,phi,lambda]=var_infer(alpha,eta,lambda_ini,thresh,iter)

% This function applys variational inference.
%
% Input:
%   alpah -- proportions parameter
%   eta -- topic parameter
%   lambda_ini -- the initial value of lambda
%   thresh -- threshold of loss to stop the optimization
%   iter -- maximum number of loops
%
% Output:
%   gamma -- attached parameter to per-document topic propotions
%   phi -- attached parameter to per-word topic assignment
%   lambda -- attached parameter to topics
%
% Author: Z. Luo
% Date: December 2015

% Calculate the initial loss.


% Start the iteration.