 function [f, df] = dcmldalikevec(x,theta,phi,deriv)
% negative Log-likelihood and its derivative for the DCMLDA model
% assumes uniformity of alpha but not of beta
% [f, df] = dcmldalikevec(x,theta,phi,deriv)
% x - alpha or beta
% theta, phi - self-explanatory (phi should be a vector)
% deriv - 'alpha' if derivative wrt alpha is desired
%       - 'beta' if derivative wrt beta desired


sumtheta = sum(sum(theta));

D = size(theta,2);      %Number of docs
K = size(theta,1);      %Number of topics
V = size(phi);        %Vocabulary size (number of types, not tokens)

%Calculating overall log-likelihood.  See (8) in writeup
%Note that we omit the final terms because they are invariant wrt alpha &
%gamma

% Also, I discard all terms that are invariant wrt to the relevant 

if strcmp(deriv,'beta')
    beta =x;
    f = D*(gammaln(sum(beta))-sum(gammaln(beta)))+sum((beta-1).*phi);
else
    alpha=x;
    f = D*gammaln(K*alpha)-D*K*gammaln(alpha)+(alpha-1)*sumtheta;
end

f = -f;

if nargout > 1
    if strcmp(deriv,'beta')
        %calculate derivative wrt beta
        df = D*(psi(sum(beta))-psi(beta))+phi;
    else
        %calculate derivative wrt alpha
        df = D*K*psi(K*alpha)-D*K*psi(alpha)+sumtheta;
    end
    df = -df;
end
