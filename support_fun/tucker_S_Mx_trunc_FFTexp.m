function [Ycore, Yfact] = tucker_S_Mx_trunc_FFTexp(M, Xcore, Xfact, maxrank, tol_tr)
%--------------------------------------------------------------------------   
% This function compute the product between the inverse of the discrete
% D-dimensional laplacian
%   sum_{k=-q}^{q} c_k exp(k\eta\Delta_{1,d}) \otimes ....\otimes exp(k\eta\Delta_{1,d})
% and a tensor X of core Xcore and Tucker factors
% Xfact, using the FFT-based approach.
% ALL THE COMPUTATIONS are done in SINGLE PRECISION.
% THE OUTPUT IS CAST BACK TO DOUBLE PRECISION.
%
% INPUT:
%   M           - [structure] where
%                 M.filter is a [cell array] of 2q+1 vectors s.t.
%                   M{j} = exp(-j*eta*lambda) with lambda the eigenvalues
%                   of \Delta_1 and eta = pi/sqrt(q)
%                 M.c is a [numeric vector] of 2q+1 elements s.t.
%                 M.c(h) = eta*exp(eta*h) for h=-q,.., q
%                 M.fftprec is a [boolean] set to true;
%   Xcore       - [numeric array] for the core tensor of size
%                 r_1 x r_2 x ... x r_d
%   Xfactors    - [cell array] of the Tucker factors of size (m_k x r_k)
%   maxrank     - [integer] maximum rank for the truncated Tucker 
%                 representation of the output
%   tol_tr      - [float] tolerance for the truncation of the Tucker 
%                 representation of the output
%
% OUTPUT:
%   Ycore       - [numeric array] for the core tensor of M(X)
%   Yfactor     - [cell array] of the Tucker factors of M(X)
%
%  If you use this code, please cite the following paper:
%
% M. Iannacito, L. Piccinini, and V. Simoncini, "Subspace gradient descent for linear tensor equations", arXiv preprint arXiv:2602.21974, 2026
%
% @article{iannacito2026subspace,
%   title={Subspace gradient descent method for linear tensor equations},
%   author={Iannacito, Martina and Piccinini, Lorenzo and Simoncini, Valeria},
%   journal={arXiv preprint arXiv:2602.21974},
%   year={2026}
% }
%
% https://arxiv.org/pdf/2602.21974
%--------------------------------------------------------------------------   


d = length(Xfact);
if d ~= ndims(Xcore)
    r = zeros(1,d);
    for k=1:d, r(k) = size(Xfact{k},2);end
else
    r = size(Xcore);
end
% Eigenvalues (Dirichlet/DST)
diag_filter = M.filter;
ck = M.c;
nterms = length(diag_filter);
% Transform the factors
tmpV = cell(1,d);
for k=1:d,tmpV{k} = zeros(size(Xfact{k}, 1), nterms*r(k));  end
hat_Xfact = cell(1,d);
for k = 1:d
    hat_Xfact{k} = my_dst(single(Xfact{k})); % FFT-based fast transform on columns
    Xfact{k} = [];
end
for j = 1:nterms
    for k=1:d
        idx = (j-1)*r(k) + 1 : j*r(k);
        tmpV{k}(:, idx) = hat_Xfact{k}.*single(diag_filter{j});
    end
end
tmpQ = cell(1,d);
R = cell(1,d);
new_r = zeros(1,d);
for k = 1:d
    [tmpQ{k}, R{k}] = qr(tmpV{k}, 0); % Standard "Economy" QR
    tmpV{k} = [];
    new_r(k) = size(tmpQ{k}, 2); % Store the new rank after QR decomposition
end

newXcore = zeros(new_r);
for j = 1:nterms
    update = single(Xcore);
    for k = 1:d
        idx = (j-1)*r(k) + 1 : j*r(k);
        tmpR = R{k}(:, idx);
        update = tucker_times_matrix(update, tmpR, k);
    end
    newXcore = newXcore + ck(j)*update;
end

[Ycore, newYfact] = tucker_trunc_dense(newXcore, maxrank, tol_tr);
Yfact = cell(1,d);
for k = 1:d
    Yfact{k} = double(my_idst(tmpQ{k}*newYfact{k}));
    tmpQ{k} = [];
    newYfact{k} = [];
end
Ycore = double(Ycore);
