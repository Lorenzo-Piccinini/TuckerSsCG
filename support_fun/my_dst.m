function y = my_dst(x)
%--------------------------------------------------------------------------
% Computes the Discrete Sine Transform (Type I) of input data using 
% an FFT-based method.  It forms an odd extension of the input and using 
% the FFT. The FFT of this odd sequence is purely imaginary, and the DST-I 
% coefficients are  obtained from the imaginary parts of the appropriate 
% FFT bins with a scaling factor.
%
% INPUT:
%   x   - [numeric array] Input vector or matrix. If y is a matrix, the 
%         transform is applied columnwise.
% OUTPUT:
%   y   - [numeric array] DST-I coefficients. If x is n-by-m, y is n-by-m 
%         containing the DST of each column.
%--------------------------------------------------------------------------   

[n, m] = size(x);
% Odd-extend the input to length 2n + 2
x_ext = [zeros(1, m); x; zeros(1, m); -flipud(x)];

% Compute FFT
y_ext = fft(x_ext);
% Extract the sine coefficients
y = -0.5 * imag(y_ext(2:n+1, :));
end