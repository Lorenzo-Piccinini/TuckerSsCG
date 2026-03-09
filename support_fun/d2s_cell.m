function y = d2s_cell(x)
%--------------------------------------------------------------------------
% convert cell array from double to single 
%
% INPUT:
%   x       - [cell array] in DOUBLE precision
%
% OUTPUT:
%   y       - [cell array] in SINGLE precision
%--------------------------------------------------------------------------
y  = cell(size(x));
[n,m] = size(x);
for i = 1:n
    for j = 1:m
        y{i, j} = single(x{i, j});
    end
end
