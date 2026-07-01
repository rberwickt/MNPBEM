import numpy as np
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..geometry.comparticle_mirror import CompStructMirror
from .planewave_stat import PlaneWaveStat


class PlaneWaveStatMirror(object):
    """Plane wave excitation within quasistatic approximation for particle
    with mirror symmetry.

    MATLAB: @planewavestatmirror

    Parameters
    ----------
    pol : ndarray
        Light polarization, shape (npol, 3)
    medium : int, optional
        Excitation medium (default: 1)
    """

    name = 'planewave'
    needs = {'sim': 'stat', 'sym': True}

    def __init__(self,
            pol: np.ndarray,
            medium: int = 1,
            **options: Any) -> None:
        self.pol = np.atleast_2d(pol)
        self.exc = PlaneWaveStat(pol, medium = medium, **options)

    def field(self,
            p: Any,
            enei: float) -> CompStruct:
        """Electric field for plane wave excitation.

        MATLAB: @planewavestatmirror/field.m
        """
        return self.exc.field(p, enei)

    def potential(self,
            p: Any,
            enei: float) -> CompStructMirror:
        """Potential of plane wave excitation with mirror symmetry.

        MATLAB: @planewavestatmirror/potential.m

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
        exc = CompStructMirror(p, enei, lambda x: self.full(x))

        nvec = p.nvec

        # MATLAB: only x, y polarization basis (no z-basis)
        phip_x = -nvec @ np.array([1, 0, 0])
        phip_y = -nvec @ np.array([0, 1, 0])

        if p.sym == 'x':
            val1 = CompStruct(p, enei, phip = phip_x)
            val1.symval = p.symvalue(['+', '-', '-'])
            val2 = CompStruct(p, enei, phip = phip_y)
            val2.symval = p.symvalue(['-', '+', '-'])
        elif p.sym == 'y':
            val1 = CompStruct(p, enei, phip = phip_x)
            val1.symval = p.symvalue(['+', '-', '+'])
            val2 = CompStruct(p, enei, phip = phip_y)
            val2.symval = p.symvalue(['-', '+', '-'])
        elif p.sym == 'xy':
            val1 = CompStruct(p, enei, phip = phip_x)
            val1.symval = p.symvalue(['++', '--', '-+'])
            val2 = CompStruct(p, enei, phip = phip_y)
            val2.symval = p.symvalue(['--', '++', '+-'])
        else:
            raise ValueError('[error] Unknown symmetry: {}'.format(p.sym))

        exc.val.append(val1)
        exc.val.append(val2)

        return exc

    def full(self,
            val: CompStructMirror) -> Tuple[CompStruct, Any]:
        """Expand surface charge/potentials/field for full particle.

        MATLAB: @planewavestatmirror/full.m

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
        pol = self.exc.pol

        expanded = val.expand()
        if len(expanded) < 2:
            return val, val.p.full()

        p_full = expanded[0].p
        result = CompStruct(p_full, expanded[0].enei)
        n_basis = len(expanded)

        # transform scalars
        for name in ('sig', 'phi', 'phip'):
            vecs = [getattr(e, name, None) for e in expanded]
            if all(v is not None for v in vecs):
                n = vecs[0].shape[0]
                npol_out = pol.shape[0]
                v = np.zeros((n, npol_out), dtype = complex)

                for ip in range(npol_out):
                    for ib in range(n_basis):
                        v[:, ip] += pol[ip, ib] * vecs[ib]

                setattr(result, name, v)

        # transform vectors
        for name in ('e',):
            vecs = [getattr(e, name, None) for e in expanded]
            if all(v is not None for v in vecs):
                n = vecs[0].shape[0]
                npol_out = pol.shape[0]
                v = np.zeros((n, 3, npol_out), dtype = complex)

                for ip in range(npol_out):
                    for ib in range(n_basis):
                        v[:, :, ip] += pol[ip, ib] * vecs[ib]

                setattr(result, name, v)

        return result, p_full

    def scattering(self, sig: CompStructMirror) -> Any:
        """Scattering cross section.

        MATLAB: @planewavestatmirror/scattering.m
        """
        full_sig, _ = self.full(sig)
        return self.exc.scattering(full_sig)

    def absorption(self, sig: CompStructMirror) -> Any:
        """Absorption cross section.

        MATLAB: @planewavestatmirror/absorption.m
        """
        full_sig, _ = self.full(sig)
        return self.exc.absorption(full_sig)

    def extinction(self, sig: CompStructMirror) -> Any:
        """Extinction cross section.

        MATLAB: @planewavestatmirror/extinction.m
        """
        full_sig, _ = self.full(sig)
        return self.exc.extinction(full_sig)

    def __call__(self,
            p: Any,
            enei: float) -> CompStructMirror:
        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'PlaneWaveStatMirror(pol={})'.format(self.pol.tolist())
