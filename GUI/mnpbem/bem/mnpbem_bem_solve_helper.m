function [x_r, x_i] = mnpbem_bem_solve_helper(M_r, M_i, b_r, b_i)
%  MNPBEM_BEM_SOLVE_HELPER - LU factorization and solve via MATLAB.
%    Used by BEMRetLayer (Wave 66) to delegate the linear solve
%    of the 2n x 2n block matrix to MATLAB for bit-identical results
%    against the MATLAB reference implementation.
%
%  Inputs
%    M_r, M_i : real and imaginary parts of M (2n x 2n)
%    b_r, b_i : real and imaginary parts of RHS (2n,) or (2n, npol)
%
%  Outputs
%    x_r, x_i : real and imaginary parts of x = M \ b
%
M = complex(M_r, M_i);
b = complex(b_r, b_i);
x = M \ b;
x_r = real(x);
x_i = imag(x);
end
