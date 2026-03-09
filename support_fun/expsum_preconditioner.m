function E = expsum_preconditioner(q, R, V)
%--------------------------------------------------------------------------   
% Construct the discrete inverse of the D-dimensional Laplacian as 
% sum_{k=-q}^{q} c_k exp(k\eta\Delta_{1,d}) \otimes ....\otimes exp(k\eta\Delta_{1,d})
%  where \Delta_{1,h} is the (minus) discretization of the Laplacian operator
%  in 1D along mode h for h=1, .., d, and eta = pi/sqrt(q).
% 
% INPUT:
%   q       - [integer] extremum of the sum  
%   R       - [cell array] of d vectors of the eigenvalues of \Delta_1
%   V       - [cell array] of d matrices of the eigenvectors of \Delta_1
%               s.t. \Delta_{1,h} = V{h}diag(R{h})V{h}' for h = 1, ..., d
% 
% OUTPUT:
%   E       - [cell array] of (2q+1 x d) matrices such that
%                E{k, h} = exp(k\eta\Delta_{1,h}) 
%--------------------------------------------------------------------------   

% Coefficients of the exponential sum
eta = pi / sqrt(q);
k = q : -1 : -q;
beta = exp(k * eta);
% Dimensionality of the tensor
d = length(R);
%Enonzero = zeros(2*q+1, d);
E = cell(2*q + 1, d);
for h = 1 : d
    kk = 1;
    for j = 1 : 2*q + 1
        E{kk,h} = V{h}*diag(exp(-beta(j) * R{h}))*V{h}';
        kk = kk +1;
    end
end

end

