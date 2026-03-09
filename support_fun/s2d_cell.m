function y = s2d_cell(x)
%--------------------------------------------------------------------------
% convert cell array from double to single 
%
% INPUT:
%   x       - [cell array] in SINGLE precision
%
% OUTPUT:
%   y       - [cell array] in DOUBLE precision
%--------------------------------------------------------------------------
y  = cell(size(x));
[n,m] = size(x);
for i = 1:n
    for j = 1:m
        y{i, j} = double(x{i, j});
    end
end