function [G_r, G_i, Fr_r, Fr_i, Fz_r, Fz_i, names] = mnpbem_layer_green_helper(eps_specs, layer_inds, ztab, opts, enei, r, z1, z2)
%  _LAYER_GREEN_HELPER - Build layerstructure and call green().
%    Designed to be called from MATLAB Engine API via Python.
%
%  Inputs
%    eps_specs   :  cell array; each cell {'const', val} or {'table', filename}
%    layer_inds  :  vector indices into eps_specs (1-based)
%    ztab        :  z-positions of interfaces (vector)
%    opts        :  struct with fields ztol, rmin, zmin, semi, ratio, atol, rtol, initial_step
%    enei        :  scalar wavelength
%    r,z1,z2     :  vectors (same shape as required by green.m)
%
%  Outputs
%    G_r/G_i etc :  struct fields for each reflection name (real and imag parts)
%    names       :  cell of reflection names

%  Build epstab
n_eps = length(eps_specs);
epstab = cell(1, n_eps);
for k = 1:n_eps
    spec = eps_specs{k};
    kind = spec{1};
    val = spec{2};
    if strcmp(kind, 'const')
        epstab{k} = epsconst(val);
    elseif strcmp(kind, 'table')
        epstab{k} = epstable(val);
    else
        error('unknown eps kind: %s', kind);
    end
end

%  Build layerstructure with options
op = layerstructure.options('ztol', opts.ztol, 'rmin', opts.rmin, ...
                            'zmin', opts.zmin, 'semi', opts.semi, ...
                            'ratio', opts.ratio);
%  ODE options
op.op = odeset('AbsTol', opts.atol, 'RelTol', opts.rtol, ...
               'InitialStep', opts.initial_step);

layer = layerstructure(epstab, layer_inds(:)', ztab(:)', op);

%  Compute Green
[G, Fr, Fz, ~] = green(layer, enei, r(:), z1(:), z2(:));

%  Marshal struct fields
names = fieldnames(G);
G_r = struct();
G_i = struct();
Fr_r = struct();
Fr_i = struct();
Fz_r = struct();
Fz_i = struct();
for k = 1:length(names)
    nm = names{k};
    G_r.(nm)  = real(G.(nm));   G_i.(nm)  = imag(G.(nm));
    Fr_r.(nm) = real(Fr.(nm));  Fr_i.(nm) = imag(Fr.(nm));
    Fz_r.(nm) = real(Fz.(nm));  Fz_i.(nm) = imag(Fz.(nm));
end

end
