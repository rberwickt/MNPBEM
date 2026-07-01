function out = mnpbem_bem_init_helper(in)
%  MNPBEM_BEM_INIT_HELPER - Build BEMRetLayer auxiliary matrices in MATLAB.
%    Mirror of MATLAB @bemretlayer/private/initmat.m, but takes the
%    Green-function matrices as input from Python (computed via
%    CompGreenRetLayer).
%
%  Input struct fields (all complex matrices unless noted):
%    G11, G21         (n x n)
%    H11, H21         (n x n)
%    G22_ss, G22_hh, G22_p, G22_sh, G22_hs   (n x n)
%    H22_ss, H22_hh, H22_p, H22_sh, H22_hs   (n x n)
%    G12, H12          (n x n)
%    eps1_diag, eps2_diag (n vectors, real or complex; if scalar, length 1)
%    k                 (scalar)
%    nvec              (n x 3 real)
%
%  Output struct fields:
%    G1, G2_ss, G2_hh, G2_p, G2_sh, G2_hs
%    G2e_ss, G2e_hh, G2e_p, G2e_sh, G2e_hs
%    H2_ss, H2_hh, H2_p, H2_sh, H2_hs
%    H2e_ss, H2e_hh, H2e_p, H2e_sh, H2e_hs
%    G1i, G2pi
%    Sigma1, Sigma1e, Sigma2p
%    L1, L2p
%    Gamma, Gammapar
%    m_full   (2n x 2n)

eps1_diag = in.eps1_diag(:);
eps2_diag = in.eps2_diag(:);

% MATLAB initmat.m unique-eps optimization
if numel(unique(eps1_diag)) == 1 && numel(unique(eps2_diag)) == 1
    eps1 = eps1_diag(1);
    eps2 = eps2_diag(1);
else
    eps1 = spdiags(eps1_diag, 0, numel(eps1_diag), numel(eps1_diag));
    eps2 = spdiags(eps2_diag, 0, numel(eps2_diag), numel(eps2_diag));
end

% Inner-surface mixed contributions
G1  = in.G11 - in.G21;
G1e = eps1 * in.G11 - eps2 * in.G21;
H1  = in.H11 - in.H21;
H1e = eps1 * in.H11 - eps2 * in.H21;

% Outer-surface mixed contributions (struct components)
G2.ss = in.G22_ss - in.G12;  G2e.ss = eps2 * in.G22_ss - eps1 * in.G12;
G2.hh = in.G22_hh - in.G12;  G2e.hh = eps2 * in.G22_hh - eps1 * in.G12;
G2.p  = in.G22_p  - in.G12;  G2e.p  = eps2 * in.G22_p  - eps1 * in.G12;
G2.sh = in.G22_sh;           G2e.sh = eps2 * in.G22_sh;
G2.hs = in.G22_hs;           G2e.hs = eps2 * in.G22_hs;

H2.ss = in.H22_ss - in.H12;  H2e.ss = eps2 * in.H22_ss - eps1 * in.H12;
H2.hh = in.H22_hh - in.H12;  H2e.hh = eps2 * in.H22_hh - eps1 * in.H12;
H2.p  = in.H22_p  - in.H12;  H2e.p  = eps2 * in.H22_p  - eps1 * in.H12;
H2.sh = in.H22_sh;           H2e.sh = eps2 * in.H22_sh;
H2.hs = in.H22_hs;           H2e.hs = eps2 * in.H22_hs;

% Auxiliary matrices
G1i  = inv(G1);
G2pi = inv(G2.p);

% Sigma matrices
Sigma1  = H1   * G1i;
Sigma1e = H1e  * G1i;
Sigma2p = H2.p * G2pi;

% Auxiliary dielectric function matrices
L1  = G1e   * G1i;
L2p = G2e.p * G2pi;

% Normal-vector parts
nvec = in.nvec;
nperp = nvec(:, 3);
npar  = nvec - nperp * [0, 0, 1];

% Gamma
Gamma = inv(Sigma1 - Sigma2p);
Gammapar = 1i * in.k * (L1 - L2p) * Gamma .* (npar * npar.');

% 2x2 block response matrix (Eq. 10)
k = in.k;
m11 = Sigma1e * G2.ss - H2e.ss - 1i * k * ...
    (Gammapar * (L1 * G2.ss - G2e.ss) + bsxfun(@times, L1 * G2.sh - G2e.sh, nperp));
m12 = Sigma1e * G2.sh - H2e.sh - 1i * k * ...
    (Gammapar * (L1 * G2.sh - G2e.sh) + bsxfun(@times, L1 * G2.hh - G2e.hh, nperp));
m21 = Sigma1  * G2.hs - H2.hs  - 1i * k * bsxfun(@times, L1 * G2.ss - G2e.ss, nperp);
m22 = Sigma1  * G2.hh - H2.hh  - 1i * k * bsxfun(@times, L1 * G2.sh - G2e.sh, nperp);

m_full = [m11, m12; m21, m22];

% Pack output
out.G1 = G1;
out.G2_ss = G2.ss; out.G2_hh = G2.hh; out.G2_p = G2.p; out.G2_sh = G2.sh; out.G2_hs = G2.hs;
out.G2e_ss = G2e.ss; out.G2e_hh = G2e.hh; out.G2e_p = G2e.p; out.G2e_sh = G2e.sh; out.G2e_hs = G2e.hs;
out.H2_ss = H2.ss; out.H2_hh = H2.hh; out.H2_p = H2.p; out.H2_sh = H2.sh; out.H2_hs = H2.hs;
out.H2e_ss = H2e.ss; out.H2e_hh = H2e.hh; out.H2e_p = H2e.p; out.H2e_sh = H2e.sh; out.H2e_hs = H2e.hs;
out.G1i = G1i;
out.G2pi = G2pi;
out.Sigma1 = Sigma1;
out.Sigma1e = Sigma1e;
out.Sigma2p = Sigma2p;
out.L1 = L1;
out.L2p = L2p;
out.Gamma = Gamma;
out.Gammapar = Gammapar;
out.m_full = m_full;
end
