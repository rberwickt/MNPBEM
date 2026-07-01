import numpy as np
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..geometry.comparticle_mirror import CompStructMirror
from .dipole_stat import DipoleStat


class DipoleStatMirror(object):
    """Excitation of an oscillating dipole with mirror symmetry (quasistatic).

    MATLAB: @dipolestatmirror

    Parameters
    ----------
    pt : ComPoint
        Compound of points for dipole positions
    dip : ndarray, optional
        Directions of dipole moments
    """

    name = 'dipole'
    needs = {'sim': 'stat', 'sym': True}

    def __init__(self,
            pt: Any,
            dip: Any = None,
            full: bool = False,
            **options: Any) -> None:
        self.dip = DipoleStat(pt, dip, full = full, **options)
        self.sym = None  # type: Optional[str]
        self.mirror = []  # type: List[DipoleStat]

    def _init_mirror(self, p: Any) -> None:
        """Initialize mirror dipoles.

        MATLAB: @dipolestatmirror/init.m
        """
        if self.sym is not None and self.sym == p.sym:
            return

        self.sym = p.sym

        pt = self.dip.pt

        # identity dipole: x, y, z unit vectors
        dip_eye = np.eye(3)
        mirror = [DipoleStat(pt, dip_eye)]

        if self.sym == 'x':
            # flip x: negate x-component of dipole
            mirror.append(DipoleStat(
                pt.flip(1),
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = np.float64)))
        elif self.sym == 'y':
            # flip y: negate y-component of dipole
            mirror.append(DipoleStat(
                pt.flip(2),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype = np.float64)))
        elif self.sym == 'xy':
            # flip x
            mirror.append(DipoleStat(
                pt.flip(1),
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = np.float64)))
            # flip y
            mirror.append(DipoleStat(
                pt.flip(2),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype = np.float64)))
            # flip x and y
            mirror.append(DipoleStat(
                pt.flip([1, 2]),
                np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype = np.float64)))
        else:
            raise ValueError('[error] Unknown symmetry: {}'.format(self.sym))

        self.mirror = mirror

    def field(self,
            p: Any,
            enei: float) -> CompStruct:
        """Electric field for dipole excitation.

        MATLAB: @dipolestatmirror/field.m
        """
        return self.dip.field(p, enei)

    def potential(self,
            p: Any,
            enei: float) -> CompStructMirror:
        """Potential of dipole excitation for use in BEMStatMirror.

        MATLAB: @dipolestatmirror/potential.m

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

        # compute surface derivatives (phip) for each mirror dipole
        mirror_phip = []
        for m in self.mirror:
            val = m(p, enei)
            mirror_phip.append(val.phip)

        # initialize CompStructMirror
        exc = CompStructMirror(p, enei, lambda x: self.full(x))

        if self.sym == 'x':
            val1 = CompStruct(p, enei, phip = mirror_phip[0] + mirror_phip[1])
            val1.symval = p.symvalue(['-', '+', '+'])
            val2 = CompStruct(p, enei, phip = mirror_phip[0] - mirror_phip[1])
            val2.symval = p.symvalue(['+', '-', '-'])
            exc.val.append(val1)
            exc.val.append(val2)

        elif self.sym == 'y':
            val1 = CompStruct(p, enei, phip = mirror_phip[0] + mirror_phip[1])
            val1.symval = p.symvalue(['+', '-', '+'])
            val2 = CompStruct(p, enei, phip = mirror_phip[0] - mirror_phip[1])
            val2.symval = p.symvalue(['-', '+', '-'])
            exc.val.append(val1)
            exc.val.append(val2)

        elif self.sym == 'xy':
            # m2 + m1 + m4 + m3
            val1 = CompStruct(p, enei, phip = mirror_phip[1] + mirror_phip[0] + mirror_phip[3] + mirror_phip[2])
            val1.symval = p.symvalue(['-+', '+-', '++'])
            # -m2 + m1 - m4 + m3
            val2 = CompStruct(p, enei, phip = -mirror_phip[1] + mirror_phip[0] - mirror_phip[3] + mirror_phip[2])
            val2.symval = p.symvalue(['++', '--', '-+'])
            # m2 + m1 - m4 - m3
            val3 = CompStruct(p, enei, phip = mirror_phip[1] + mirror_phip[0] - mirror_phip[3] - mirror_phip[2])
            val3.symval = p.symvalue(['--', '++', '+-'])
            # -m2 + m1 + m4 - m3
            val4 = CompStruct(p, enei, phip = -mirror_phip[1] + mirror_phip[0] + mirror_phip[3] - mirror_phip[2])
            val4.symval = p.symvalue(['+-', '-+', '--'])
            exc.val.append(val1)
            exc.val.append(val2)
            exc.val.append(val3)
            exc.val.append(val4)

        return exc

    def full(self,
            val: CompStructMirror) -> Tuple[CompStruct, Any]:
        """Expand surface charge/potentials/field for full particle.

        MATLAB: @dipolestatmirror/full.m

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
        for name in ('sig', 'phi1', 'phi1p', 'phi2', 'phi2p'):
            v = getattr(vmean, name, None)
            if v is None:
                continue

            # DipoleStat.potential() may flatten (n, npt, 3) -> (n, npt*3)
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

            # Flatten to (n, npt*ndip) for downstream consumers.
            setattr(result, name, vi.reshape(n, npt * ndip))

        # transform vectors
        for name in ('e',):
            v = getattr(vmean, name, None)
            if v is None:
                continue

            # DipoleStat.potential() may flatten (n, 3, npt, 3) -> (n, 3, npt*3).
            # Reshape back before transforming.
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

            # Flatten to (n, 3, npt*ndip) for downstream consumers.
            setattr(result, name, vi.reshape(n, 3, npt * ndip))

        return result, p_full

    def decayrate(self,
            sig: CompStructMirror) -> Any:
        """Total and radiative decay rate for oscillating dipole.

        MATLAB: @dipolestatmirror/decayrate.m
        """
        full_sig, _ = self.full(sig)
        return self.dip.decayrate(full_sig)

    def farfield(self,
            spec: Any,
            enei: float) -> Any:
        """Electromagnetic fields of dipoles in the far-field limit.

        MATLAB: @dipolestatmirror/farfield.m
        """
        return self.dip.farfield(spec, enei)

    def __call__(self,
            p: Any,
            enei: float) -> CompStructMirror:
        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'DipoleStatMirror(dip={}, sym={})'.format(self.dip, self.sym)
