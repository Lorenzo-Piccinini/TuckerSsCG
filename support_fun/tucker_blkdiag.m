function C = tucker_blkdiag(A,B)
%--------------------------------------------------------------------------   
% Create a block-diagonal concatenation of two tensors.
% C = tucker_blkdiag(A,B) returns a tensor C whose size is the element-wise
% sum of the sizes of A and B. The input arrays A and B may be 2-D matrices
% or 3-D tensors. If an input is 2-D, it is treated as a 3-D array with a
% singleton third dimension.
% The result C places A in the "upper-left-front" block and B in the
% "lower-right-back" block, with all other elements set to zero. More
% precisely:
% - If A is size [a1 a2 a3] and B is size [b1 b2 b3], then C is size
%   [a1+b1, a2+b2, a3+b3].
% - C(1:a1,1:a2,1:a3) = A
% - C(a1+1:end,a2+1:end,a3+1:end) = B
% 
% INPUT:
%   A   - [numeric array] input dense tensor of size (n1,..,nd)
%   B   - [numeric array] input dense tensor of size (m1, .., md)
%
% OUTPUT:
%   C   - [numeric array] input dense tensor of size (m1+ n1,..,md+nd)
%
% Examples:
% % Two matrices (treated as third dimension = 1)
% A = rand(2,3); B = rand(4,2);
% C = tucker_blkdiag(A,B);   % size(C) = [6,5,1]
% % Two 3-D tensors
% A = rand(2,2,2); B = rand(3,1,4);
% C = tucker_blkdiag(A,B);   % size(C) = [5,3,6]
%
% Notes:
% - The function does not attempt to align modes other than by simple
%   concatenation of sizes, so it is suitable when constructing block
%   diagonal Tucker cores or similarly structured tensors.
% - Inputs are not modified.
%--------------------------------------------------------------------------   

sizeA=size(A);
sizeB=size(B);
if length(sizeA) == 2, sizeA = [sizeA, 1]; end
if length(sizeB) == 2, sizeB = [sizeB, 1]; end

C = zeros(sizeA + sizeB);
C(1:sizeA(1), 1:sizeA(2), 1:sizeA(3)) = A;
C(1+sizeA(1):end, 1+sizeA(2):end, 1+sizeA(3):end) = B;

end
