function y = tucker_Ax_trunc(A, x, maxrank, tol, single_flag)
%--------------------------------------------------------------------------
% Compute the product between a tensor operator A and a tensor X in
% Tucker-format truncating at multilinear rank maxrank and accuracy tol
%
% INPUT:
%   A           - [cell array] of factor matrices forming the operator A. 
%                   The operator represented is a sum of Kronecker products: 
%                   A = sum_{l=1..nterms} kron(A{l,1}, A{l,2}, ..., A{l,d}).
%   x           - [structure] for Tucker tensor, where 
%                   X.core is a [numeric array] for the core tensor of size 
%                   r_1 x r_2 x ... x r_d
%                   X.factors is a [cell array] of the Tucker factor 
%                   matrices of size (m_k x r_k)
%   maxrank     - [integer] maximum rank per mode
%   tol         - [float] truncation tolerance
%   single_flag - [boolean] if true, computations are done in single
%                   precision, otherwise in double
% OUTPUT:
%   y           - [numeric array] obtained as A(X) 
%
% If you use this code, please cite the following paper:
%
% M. Iannacito, L. Piccinini, V. Simoncini, "Subspace gradient descent method for linear tensor equations", arXiv preprint arXiv:2602.21974, 2026.
%
% @article{iannacito2026subspace,
%   title={Subspace gradient descent method for linear tensor equations},
%   author={Iannacito, Martina and Piccinini, Lorenzo and Simoncini, Valeria},
%   journal={arXiv preprint arXiv:2602.21974},
%   year={2026}
% }
% 
% https://arxiv.org/abs/2602.21974
%--------------------------------------------------------------------------

nterms = size(A, 1);
d = size(A,2);
n = zeros(1,d);

for k = 1:d, n(k) = size(A{1, k}, 2); end

[Xcore, Xfact] = tucker_trunc_dense(reshape(x, n), maxrank, tol);
Yfact = cell(nterms);
Yfact{1} = cell(1, d);
if single_flag
    for k = 1:d
       Yfact{1}{k} = single(A{1, k})*single(Xfact{k});   
    end
    Y = tucker_times_matrix(Xcore, Yfact{1});
    for l = 2:nterms
        Yfact{l} = cell(1, d);
        for k = 1:d
           Yfact{l}{k} = single(A{l, k})*single(Xfact{k});
        end
        Y = Y + single(tucker_times_matrix(single(Xcore), Yfact{l}));
    end
    [Ycore, Yfact] = tucker_trunc_dense(Y, maxrank, tol);
    y = reshape(single(tucker_times_matrix(single(Ycore), Yfact)), [], 1);
else
    for k = 1:d
       Yfact{1}{k} = (A{1, k})*(Xfact{k});   
    end
    Y = tucker_times_matrix(Xcore, Yfact{1});
    for l = 2:nterms
        Yfact{l} = cell(1, d);
        for k = 1:d
           Yfact{l}{k} = (A{l, k})*(Xfact{k});
        end
        Y = Y + (tucker_times_matrix(single(Xcore), Yfact{l}));
    end
    [Ycore, Yfact] = tucker_trunc_dense(Y, maxrank, tol);
    y = reshape(tucker_times_matrix(single(Ycore), Yfact), [], 1);
end
