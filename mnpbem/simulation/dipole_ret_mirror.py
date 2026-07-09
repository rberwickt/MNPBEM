import numpy as np
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..geometry.comparticle_mirror import CompStructMirror
from .dipole_ret import DipoleRet


class DipoleRetMirror(object):
    """Excitation of an oscillating dipole with mirror symmetry (retarded).

    MATLAB: @dipoleretmirror

    Parameters
    ----------
    pt : ComPoint
        Compound of points for dipole positions
    dip : ndarray, optional
        Directions of dipole moments
    medium : int, optional
        Embedding medium (default: 1)
    pinfty : Particle, optional
        Unit sphere at infinity
    """

    name = 'dipole'
    needs = {'sim': 'ret', 'sym': True}

    def __init__(self,
            pt: Any,
            dip: Any = None,
            full: bool = False,
            medium: int = 1,
            pinfty: Any = None,
            **options: Any) -> None:
        self.dip = DipoleRet(pt, dip, full = full, medium = medium,
                             pinfty = pinfty, **options)
        self.sym = None  # type: Optional[str]
        self.mirror = []  # type: List[DipoleRet]

    def _init_mirror(self, p: Any) -> None:
        """Initialize mirror dipoles.

        MATLAB: @dipoleretmirror/init.m
        """
        if self.sym is not None and self.sym == p.sym:
            return

        self.sym = p.sym

        pt = self.dip.pt

        # identity dipole: x, y, z unit vectors
        dip_eye = np.eye(3)
        mirror = [DipoleRet(pt, dip_eye)]

        if self.sym == 'x':
            # flip x: negate x-component of dipole
            mirror.append(DipoleRet(
                pt.flip(1),
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = np.float64)))
        elif self.sym == 'y':
            # flip y: negate y-component of dipole
            mirror.append(DipoleRet(
                pt.flip(2),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype = np.float64)))
        elif self.sym == 'xy':
            # flip x
            mirror.append(DipoleRet(
                pt.flip(1),
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = np.float64)))
            # flip y
            mirror.append(DipoleRet(
                pt.flip(2),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype = np.float64)))
            # flip x and y
            mirror.append(DipoleRet(
                pt.flip([1, 2]),
                np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype = np.float64)))
        else:
            raise ValueError('[error] Unknown symmetry: {}'.format(self.sym))

        self.mirror = mirror

    def field(self,
            p: Any,
            enei: float,
            inout: int = 1) -> CompStruct:
        """Electromagnetic fields for dipole excitation.

        MATLAB: @dipoleretmirror/field.m
        """
        return self.dip.field(p, enei, inout)

    def potential(self,
            p: Any,
            enei: float) -> CompStructMirror:
        """Potential of dipole excitation for use in BEMRetMirror.

        MATLAB: @dipoleretmirror/potential.m

        Parameters
        ----------
        p : ComParticleMirror
            Particle with mirror symmetry
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        exc : CompStructMirror
            Potential with symmetry values
        """
        # initialize mirror dipoles
        self._init_mirror(p)

        # compute potentials for each mirror dipole
        mirror_vals = [m(p, enei) for m in self.mirror]

        # initialize CompStructMirror
        exc = CompStructMirror(p, enei, lambda x: self.full(x))

        if self.sym == 'x':
            val1 = mirror_vals[0] + mirror_vals[1]
            val1.symval = p.symvalue(['-', '+', '+'])
            val2 = mirror_vals[0] - mirror_vals[1]
            val2.symval = p.symvalue(['+', '-', '-'])
            exc.val.append(val1)
            exc.val.append(val2)

        elif self.sym == 'y':
            val1 = mirror_vals[0] + mirror_vals[1]
            val1.symval = p.symvalue(['+', '-', '+'])
            val2 = mirror_vals[0] - mirror_vals[1]
            val2.symval = p.symvalue(['-', '+', '-'])
            exc.val.append(val1)
            exc.val.append(val2)

        elif self.sym == 'xy':
            # m2 + m1 + m4 + m3
            val1 = mirror_vals[1] + mirror_vals[0] + mirror_vals[3] + mirror_vals[2]
            val1.symval = p.symvalue(['-+', '+-', '++'])
            # -m2 + m1 - m4 + m3
            val2 = -mirror_vals[1] + mirror_vals[0] - mirror_vals[3] + mirror_vals[2]
            val2.symval = p.symvalue(['++', '--', '-+'])
            # m2 + m1 - m4 - m3
            val3 = mirror_vals[1] + mirror_vals[0] - mirror_vals[3] - mirror_vals[2]
            val3.symval = p.symvalue(['--', '++', '+-'])
            # -m2 + m1 + m4 - m3
            val4 = -mirror_vals[1] + mirror_vals[0] + mirror_vals[3] - mirror_vals[2]
            val4.symval = p.symvalue(['+-', '-+', '--'])
            exc.val.append(val1)
            exc.val.append(val2)
            exc.val.append(val3)
            exc.val.append(val4)

        return exc

    def full(self,
            val: CompStructMirror) -> Tuple[CompStruct, Any]:
        """Expand surface charges/potentials/fields for full particle.

        MATLAB: @dipoleretmirror/full.m

        Parameters
        ----------
        val : CompStructMirror
            Mirror symmetry values

        Returns
        -------
        result : CompStruct
            Values expanded for full particle
        p : ComParticle
            Expanded particle
        """
        expanded = val.expand()
        if len(expanded) == 0:
            return val, val.p.full()

        # average expanded values
        if val.p.sym in ('x', 'y'):
            val1, val2 = expanded[0], expanded[1]
            vmean = 0.5 * (val1 + val2)
        elif val.p.sym == 'xy':
            val1, val2, val3, val4 = expanded[0], expanded[1], expanded[2], expanded[3]
            vmean = 0.25 * (val1 + val2 + val3 + val4)
        else:
            raise ValueError('[error] Unknown symmetry: {}'.format(val.p.sym))

        p_full = expanded[0].p

        result = CompStruct(p_full, val.enei)

        # requested dipole moments
        dip = self.dip.dip  # (npt, 3, ndip)

        npt = dip.shape[0]
        ndip = dip.shape[2]

        # transform scalars
        for name in ('sig1', 'sig2', 'phi1', 'phi1p', 'phi2', 'phi2p'):
            v = getattr(vmean, name, None)
            if v is None:
                continue

            # DipoleRet.potential() flattens (n, npt, 3) -> (n, npt*3) for BEM.
            # Reshape back to (n, npt, 3) before transforming.
            n = v.shape[0]
            if v.ndim == 2 and v.shape[1] == npt * 3:
                v = v.reshape(n, npt, 3)
            elif v.ndim == 2 and v.shape[1] == 3 and npt == 1:
                v = v.reshape(n, 1, 3)

            vi = np.zeros((n, npt, ndip), dtype = complex)

            for i in range(npt):
                for j in range(ndip):
                    d = dip[i, :, j]  # (3,)
                    vi[:, i, j] = (d[0] * v[:, i, 0]
                                 + d[1] * v[:, i, 1]
                                 + d[2] * v[:, i, 2])

            # Flatten (n, npt, ndip) -> (n, npt*ndip) to match downstream
            # (DipoleRet.scattering / spectrum_ret.farfield) expectation.
            setattr(result, name, vi.reshape(n, npt * ndip))

        # transform vectors
        for name in ('e', 'h', 'h1', 'h2', 'a1', 'a1p', 'a2', 'a2p'):
            v = getattr(vmean, name, None)
            if v is None:
                continue

            # DipoleRet.potential() flattens (n, 3, npt, 3) -> (n, 3, npt*3).
            # Reshape back to (n, 3, npt, 3) before transforming.
            n = v.shape[0]
            if v.ndim == 3 and v.shape[2] == npt * 3:
                v = v.reshape(n, 3, npt, 3)
            elif v.ndim == 3 and v.shape[2] == 3 and npt == 1:
                v = v.reshape(n, 3, 1, 3)

            vi = np.zeros((n, 3, npt, ndip), dtype = complex)

            for i in range(npt):
                for j in range(ndip):
                    d = dip[i, :, j]  # (3,)
                    vi[:, :, i, j] = (d[0] * v[:, :, i, 0]
                                    + d[1] * v[:, :, i, 1]
                                    + d[2] * v[:, :, i, 2])

            # Flatten (n, 3, npt, ndip) -> (n, 3, npt*ndip) to match downstream.
            setattr(result, name, vi.reshape(n, 3, npt * ndip))

        return result, p_full

    def scattering(self,
            sig: CompStructMirror) -> Any:
        """Scattering cross section for dipole excitation.

        MATLAB: @dipoleretmirror/scattering.m
        """
        full_sig, _ = self.full(sig)
        return self.dip.scattering(full_sig)

    def farfield(self,
            spec: Any,
            enei: float) -> Any:
        """Electromagnetic fields of dipoles in the far-field limit.

        MATLAB: @dipoleretmirror/farfield.m
        """
        return self.dip.farfield(spec, enei)

    def decayrate(self,
            sig: CompStructMirror) -> Any:
        """Total and radiative decay rate for oscillating dipole.

        MATLAB: @dipoleretmirror/decayrate.m
        """
        full_sig, _ = self.full(sig)
        return self.dip.decayrate(full_sig)

    def radiative(self,
            field: Any) -> Tuple[Any, Any]:
        """Radiative decay rate for oscillating dipole.

        MATLAB: @dipoleretmirror/radiative.m
        """
        # dielectric function
        epsb = self.dip.pt.eps1(field.enei).reshape(-1, 1)
        nb = np.sqrt(epsb)

        if np.any(np.imag(nb) != 0):
            import warnings
            warnings.warn('Dipole embedded in medium with complex dielectric function')

        # wavenumber of light in vacuum
        k0 = 2 * np.pi / field.enei
        # Wigner-Weisskopf decay rate in free space
        rad0 = nb * 4 / 3 * k0 ** 3

        # power emitted by oscillating dipole
        from .planewave_ret import PlaneWaveRet
        p = _scattering_power(field)
        # wavenumber in medium
        k = nb * k0
        # transform from emitted power to scattering rate
        ndip_out = field.e.shape[3] if field.e.ndim == 4 else 1
        rad = p / np.tile(2 * np.pi * k * epsb * rad0, (1, ndip_out))

        return rad, rad0

    def __call__(self,
            p: Any,
            enei: float) -> CompStructMirror:
        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'DipoleRetMirror(dip={}, sym={})'.format(self.dip, self.sym)


def _scattering_power(field: Any) -> Any:
    """Compute scattering power from far-field.

    Helper for radiative decay rate calculation.
    """
    if hasattr(field, 'p') and hasattr(field.p, 'area'):
        area = field.p.area
        e = field.e
        h = field.h

        if e.ndim == 4:
            ndir = e.shape[0]
            npt = e.shape[2]
            ndip = e.shape[3]
            p = np.zeros((npt, ndip))

            for i in range(npt):
                for j in range(ndip):
                    # Poynting vector integration
                    s = 0.5 * np.real(
                        np.cross(e[:, :, i, j], np.conj(h[:, :, i, j]), axis = 1))
                    nvec = field.p.nvec if hasattr(field.p, 'nvec') else np.zeros_like(s)
                    p[i, j] = np.sum(np.sum(s * nvec, axis = 1) * area)
            return p

    return 0
