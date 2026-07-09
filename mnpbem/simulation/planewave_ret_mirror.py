import numpy as np
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..geometry.comparticle_mirror import CompStructMirror
from .planewave_ret import PlaneWaveRet


class PlaneWaveRetMirror(object):
    """Plane wave excitation for full Maxwell equations using mirror symmetry.

    MATLAB: @planewaveretmirror

    Parameters
    ----------
    pol : ndarray
        Light polarization, shape (npol, 3). Must have pol[:, 2] == 0.
    dir : ndarray
        Light propagation direction, shape (npol, 3)
    medium : int, optional
        Excitation medium (default: 1)
    pinfty : Particle, optional
        Unit sphere at infinity
    """

    name = 'planewave'
    needs = {'sim': 'ret', 'sym': True}

    def __init__(self,
            pol: np.ndarray,
            dir: np.ndarray,
            medium: int = 1,
            **options: Any) -> None:
        pol = np.atleast_2d(pol)
        dir = np.atleast_2d(dir)
        assert np.allclose(pol[:, 2], 0), '[error] pol[:, 2] must be 0 for mirror symmetry'

        self.pol = pol
        self.dir = dir
        self.exc = PlaneWaveRet(pol, dir, medium = medium, **options)

    def field(self,
            p: Any,
            enei: float,
            inout: int = 1) -> CompStruct:
        """Electric and magnetic field for plane wave excitation.

        MATLAB: @planewaveretmirror/field.m
        """
        return self.exc.field(p, enei, inout)

    def potential(self,
            p: Any,
            enei: float) -> CompStructMirror:
        """Potential of plane wave excitation for use in BEMRetMirror.

        MATLAB: @planewaveretmirror/potential.m

        Parameters
        ----------
        p : ComParticleMirror
            Particle with mirror symmetry
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        pot : CompStructMirror
            Potential with symmetry values
        """
        pot = CompStructMirror(p, enei, lambda x: self.full(x))

        # external excitation for x-pol and y-pol basis
        for i in range(2):
            pol_basis = np.zeros((2, 3))
            pol_basis[:, i] = 1.0
            dir_basis = np.array([[0, 0, 1], [0, 0, -1]], dtype = np.float64)

            exc_copy = PlaneWaveRet(pol_basis, dir_basis,
                                    medium = self.exc.medium)
            val = exc_copy(p, enei)
            pot.val.append(val)

        # add symmetry values
        if p.sym == 'x':
            pot.val[0].symval = p.symvalue(['+', '-', '-'])
            pot.val[1].symval = p.symvalue(['-', '+', '+'])
        elif p.sym == 'y':
            pot.val[0].symval = p.symvalue(['+', '-', '+'])
            pot.val[1].symval = p.symvalue(['-', '+', '-'])
        elif p.sym == 'xy':
            pot.val[0].symval = p.symvalue(['++', '--', '-+'])
            pot.val[1].symval = p.symvalue(['--', '++', '+-'])

        return pot

    def full(self,
            val: CompStructMirror) -> Tuple[CompStruct, Any]:
        """Expand surface charges/potentials/fields for full particle.

        MATLAB: @planewaveretmirror/full.m

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
        pol = self.pol
        dir = self.dir

        expanded = val.expand()
        if len(expanded) < 2:
            return val, val.p.full()

        val1, val2 = expanded[0], expanded[1]
        p_full = val1.p

        result = CompStruct(p_full, val1.enei)

        # transform scalars
        for name in ('sig1', 'sig2', 'phi1', 'phi1p', 'phi2', 'phi2p'):
            v1 = getattr(val1, name, None)
            v2 = getattr(val2, name, None)
            if v1 is not None and v2 is not None:
                n = v1.shape[0]
                npol_out = pol.shape[0]
                v = np.zeros((n, npol_out), dtype = complex)

                for ip in range(npol_out):
                    j = 0 if dir[ip, 2] > 0 else 1
                    v1_j = v1[:, j] if v1.ndim > 1 else v1
                    v2_j = v2[:, j] if v2.ndim > 1 else v2
                    v[:, ip] = pol[ip, 0] * v1_j + pol[ip, 1] * v2_j

                setattr(result, name, v)

        # transform vectors
        for name in ('h1', 'h2', 'a1', 'a1p', 'a2', 'a2p', 'e', 'h'):
            v1 = getattr(val1, name, None)
            v2 = getattr(val2, name, None)
            if v1 is not None and v2 is not None:
                n = v1.shape[0]
                npol_out = pol.shape[0]
                v = np.zeros((n, 3, npol_out), dtype = complex)

                for ip in range(npol_out):
                    j = 0 if dir[ip, 2] > 0 else 1
                    if v1.ndim == 3:
                        v1_j = v1[:, :, j]
                        v2_j = v2[:, :, j]
                    else:
                        v1_j = v1
                        v2_j = v2
                    v[:, :, ip] = pol[ip, 0] * v1_j + pol[ip, 1] * v2_j

                setattr(result, name, v)

        return result, p_full

    def scattering(self, sig: CompStructMirror) -> Any:
        """Scattering cross section.

        MATLAB: @planewaveretmirror/subsref.m -> exc.sca
        """
        full_sig, _ = self.full(sig)
        return self.exc.scattering(full_sig)

    def extinction(self, sig: CompStructMirror) -> Any:
        """Extinction cross section.

        MATLAB: @planewaveretmirror/subsref.m -> exc.ext
        """
        full_sig, _ = self.full(sig)
        return self.exc.extinction(full_sig)

    def absorption(self, sig: CompStructMirror) -> Any:
        """Absorption cross section.

        MATLAB: @planewaveretmirror/subsref.m -> exc.abs
        """
        full_sig, _ = self.full(sig)
        return self.exc.absorption(full_sig)

    def __call__(self,
            p: Any,
            enei: float) -> CompStructMirror:
        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'PlaneWaveRetMirror(pol={}, dir={})'.format(
            self.pol.tolist(), self.dir.tolist())
