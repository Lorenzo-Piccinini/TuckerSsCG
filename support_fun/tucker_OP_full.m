function sumAkron = tucker_OP_full(A)
%--------------------------------------------------------------------------   
% Compute the Kronecker product of the matrices forming the tensor operator
% as sum_{l=1,..., nterms} of A_{l,d} \kron ... \kron A_{l,1}
% 
% INPUT
%   A           - [cell array] operator, terms on 1st mode, dimension on 
%                2nd mode
% OUTPUT
%   sumAkron    - [matrix] sum_{l=1,..., nterms} of A_{l,d} \kron ... \kron A_{l,1}
%--------------------------------------------------------------------------   
d = size(A,2);
nterms = size(A,1);
sumAkron = kron(A{1, 2}, A{1, 1});
for k = 3:d
    sumAkron = kron(A{1, k}, sumAkron);
end
for l = 2:nterms
    Akron = kron(A{l,2}, A{l,1});
    for k = 3:d
        Akron = kron(A{l,k}, Akron);
    end
    sumAkron = sumAkron + Akron;
end
end