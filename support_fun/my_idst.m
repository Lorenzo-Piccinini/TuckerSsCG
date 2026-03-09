function x = my_idst(y)
%--------------------------------------------------------------------------
% Compute the inverse Discrete Sine Transform (DST) of type I. For DST-I 
% the inverse transform is identical to the forward transform up to a 
% scaling factor. This implementation calls my_dst to perform the forward 
% DST-I and then applies the appropriate normalization.
%
% INPUT:
%   y  - [numeric array] Input vector or matrix whose columns contain DST-I 
%        coefficients. If y is a matrix, the transform is applied columnwise.
% 
% OUTPUT:
%   x  - [numeric array] Reconstructed signal(s) after applying the inverse
%        DST-I. The output has the same size as y.
%--------------------------------------------------------------------------

[n, ~] = size(y);

% Apply the forward transform
y_raw = my_dst(y);

% Apply scaling factor
x = y_raw * (2 / (n + 1));
end
