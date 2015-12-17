function beta = traindir(phi)
% train Dirichlet
% phi is a matrix with D columns, each being one vector of word counts
% beta is a column vector with W rows

[W D] = size(phi);

phi = full(phi) + 1;
for d = 1:D
    phi(:,d) = phi(:,d) ./ sum(phi(:,d));
end

b = ones(W,1);

%bdisc = checkgrad('dcmldbeta', b, 1e-5, phi);
%sprintf('Max relative discrepancy %g in beta gradient.',bdisc)

beta = minimize(b,'dcmldbeta',100,phi); 