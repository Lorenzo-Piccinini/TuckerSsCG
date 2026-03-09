function [Ycore, Yfact] = tucker_Mx_trunc_exp(M, Xcore, Xfact, maxrank, tol_tr)
%--------------------------------------------------------------------------
% This function compute the product between the inverse of the discrete
% D-dimensional laplacian
%   sum_{k=-q}^{q} c_k exp(k\eta\Delta_{1,d}) \otimes ....\otimes exp(k\eta\Delta_{1,d})
% and a tensor X of core Xcore and Tucker factors
% Xfact.
% ALL THE COMPUTATIONS are done in DOUBLE PRECISION.
%
% INPUT:
%   M           - [structure] where
%                 M.E is a [cell array] of (2q+1 x d) matrices s.t.
%                   M{k, h} = exp(k\eta\Delta_{1,h})
%                 M.c is a [numeric vector] of 2q+1 elements s.t.
%                 M.c(h) = eta*exp(eta*h) for h=-q,.., q
%                 M.fftprec is a [boolean] set to false;
%   Xcore       - [numeric array] for the core tensor of size
%                 r_1 x r_2 x ... x r_d
%   Xfactors    - [cell array] of the Tucker factors of size (m_k x r_k)
%   maxrank     - [integer] maximum rank for the truncated Tucker 
%                 representation of the output
%   tol_tr      - [float] tolerance for the truncation of the Tucker 
%                 representation of the output
% OUTPUT:
%  Ycore, Yfact - [structure] Tucker representation of the output tensor
%                 after applying the preconditioner, i.e., M(X)
%
% If you use this code, please cite the following paper:
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

c = M.c;
E = M.E;
nterms = length(c); % Length of the exponential sum
d = size(E,2);

Ycore = Xcore;
Yfact = cell(1,d);
for k = 1:d
    Yfact{k} =  c(1)*E{1, k}*(Xfact{k});
end

for l = 2:nterms
    Ycore = tucker_blkdiag(Ycore, Xcore);
    for k = 1:d
        Yfact{k} = [Yfact{k}, c(l)*E{l, k}*Xfact{k}];
        [Uy, Sy, Vy] = svd(Yfact{k}, 'econ');
        sy = diag(Sy)./sum(diag(Sy));
        ry = sum(sy > tol_tr);
        Yfact{k} = Uy(:, 1:ry);
        Ycore = tucker_times_matrix(Ycore, (Sy(1:ry, 1:ry)*(Vy(:, 1:ry)')), k);
    end
    if ndims(Ycore) == d
        [Ycore, newYfact] = tucker_trunc_dense(Ycore, maxrank, tol_tr);
        for k=1:d
            Yfact{k} = Yfact{k}*newYfact{k};
            newYfact{k} = [];
        end
    end
end

