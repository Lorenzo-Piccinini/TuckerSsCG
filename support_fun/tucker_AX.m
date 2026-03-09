function Y = tucker_AX(A, X)
%--------------------------------------------------------------------------
% Compute the product between a tensor operator A and a tensor X in
% Tucker-format
%
% INPUT:
%   A       - [cell array] of factor matrices forming the operator A. 
%               The operator represented is a sum of Kronecker products: 
%               A = sum_{l=1..nterms} kron(A{l,1}, A{l,2}, ..., A{l,d}).
%   X       - [structure] for Tucker tensor, where 
%               X.core is a [numeric array] for the core tensor of size 
%               r_1 x r_2 x ... x r_d
%               X.factors is a [cell array] of the Tucker factor matrices of
%               size (m_k x r_k)
% OUTPUT:
%   Y       - [numeric array] obtained as A(X) 
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
Xfact = X.factors;
Xcore = X.core;
Yfact = cell(nterms);
Yfact{1} = cell(1, d);
for k = 1:d
   Yfact{1}{k} = A{1, k}*Xfact{k};
end
Y = tucker_times_matrix(Xcore, Yfact{1});
for l = 2:nterms
    Yfact{l} = cell(1, d);
    for k = 1:d
       Yfact{l}{k} = A{l, k}*Xfact{k};
    end
    Y = Y + tucker_times_matrix(Xcore, Yfact{l});
end
end