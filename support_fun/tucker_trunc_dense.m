function [Xcore, Xfact] = tucker_trunc_dense(X, maxrank, tol)
%--------------------------------------------------------------------------   
% This function computes the ST-HOSVD on a dense tensor X.
% 
% INPUT:
%   X           - [numeric array] input dense tensor
%   maxrank     - [integer] or [numeric vector] maximum rank for the 
%                 truncated Tucker representation of the output
%   tol_tr      - [float] tolerance for the truncation of the Tucker 
%                 representation of the output
%
% OUTPUT:
%   Xcore       - [numeric array] for the core tensor of ST-HOSVD(X)
%   Xfactor     - [cell array] of the Tucker factors of ST-HOSVD(X)
%
% Notes:
%   - If maxrank is a scalar it is expanded to all modes.
%   - tol is applied to normalized singular values (sum to 1) to decide truncation.
%--------------------------------------------------------------------------   

d = ndims(X);
n = size(X);
Xfact = cell(1,d);
Xcore = X;
if isscalar(maxrank), maxrank = maxrank*ones(1,d); end 
for k = 1:d
    Xmat = reshape(permute(Xcore, [k 1:k-1 k+1:d]), n(k), []);
    [U, S, ~] = svd(Xmat, "econ");
    s = diag(S)./sum(diag(S));
    r = min(length(s(s > tol)), maxrank(k));
    Xfact{k} = U(:, 1:r);
    Xcore = tucker_times_matrix(Xcore, Xfact{k}', k);
end
