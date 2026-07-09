"""
Face quadrature for boundary element integration.

MATLAB: integration/@quadface/
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .gauss_legendre import lglnodes
from .options import getbemoptions


class QuadFace(object):
    """
    Integration over triangular or quadrilateral boundary elements.

    MATLAB: integration/@quadface

    Parameters
    ----------
    rule : int
        Integration rule (1-19, see triangle_unit_set)
    refine : int, optional
        Refine surface element
    npol : list or int, optional
        Number of points for polar integration [nrad, nang]

    Methods
    -------
    adapt(verts) -> Tuple[ndarray, ndarray]
    plot() -> None
    """

    def __init__(self, **kwargs: Any) -> None:
        # get BEM options
        op = getbemoptions(kwargs)

        if 'quadface' in op and isinstance(op['quadface'], dict):
            op.update(op['quadface'])

        # number of integration points for polar integration
        if 'npol' not in op:
            op['npol'] = [7, 5]
        npol = op['npol']
        if isinstance(npol, (int, float)):
            npol = [int(npol), int(npol)]
        self.npol = npol

        # triangle integration
        if 'rule' not in op:
            op['rule'] = 18
        self.x, self.y, self.w = triangle_unit_set(op['rule'])

        # refine triangles
        if 'refine' in op:
            self.x, self.y, self.w = trisubdivide(
                self.x, self.y, self.w, op['refine'])

        # polar triangle integration
        x1, w1 = lglnodes(npol[0])
        rho_t = 0.5 * (x1 + 1 + 1e-6)
        x2, w2 = lglnodes(npol[1])
        phi_t = (270 + 60 * x2) / 180 * np.pi
        phi0_t = 120 / 180 * np.pi

        # make 2d arrays
        rho_2d, phi_2d = np.meshgrid(rho_t, phi_t, indexing = 'ij')
        rho_flat = rho_2d.ravel()
        phi_flat = phi_2d.ravel()

        # radius
        rad_t = 1.0 / np.abs(2 * np.sin(phi_flat))

        # three rotated copies
        # MATLAB uses column-major ravel (phi = [phi, phi+phi0, phi+2*phi0] then x(:)),
        # so x3/y3 must be concatenated with the three sector copies sequentially
        # in alignment with w3 = repmat(w(:) .* rho .* rad.^2, 3, 1).
        phi_3 = np.concatenate([phi_flat, phi_flat + phi0_t, phi_flat + 2 * phi0_t])
        rho_rep = np.tile(rho_flat, 3)
        rad_rep = np.tile(rad_t, 3)

        # integration points in triangle
        rho_rad = rho_rep * rad_rep
        x3 = np.cos(phi_3) * rho_rad
        y3 = np.sin(phi_3) * rho_rad

        # transform to unit triangle
        x3, y3 = ((1 - np.sqrt(3) * x3 - y3) / 3,
                   (1 + np.sqrt(3) * x3 - y3) / 3)

        # integration weights
        w_2d = np.outer(w1, w2).ravel()
        w3 = np.tile(w_2d * rho_flat * rad_t ** 2, 3)
        w3 = w3 / np.sum(w3)

        self.x3 = x3
        self.y3 = y3
        self.w3 = w3

        # polar quadrilateral integration
        x1q, w1q = lglnodes(npol[0])
        rho_q = 0.5 * (x1q + 1 + 1e-6)
        x2q, w2q = lglnodes(npol[1])
        phi_q = (90 + 45 * x2q) / 180 * np.pi
        phi0_q = np.pi / 2

        rho_2d_q, phi_2d_q = np.meshgrid(rho_q, phi_q, indexing = 'ij')
        rho_flat_q = rho_2d_q.ravel()
        phi_flat_q = phi_2d_q.ravel()

        rad_q = 1.0 / np.abs(np.sin(phi_flat_q))

        # four rotated copies (MATLAB column-major ravel convention)
        phi_4 = np.concatenate([
            phi_flat_q,
            phi_flat_q + phi0_q,
            phi_flat_q + 2 * phi0_q,
            phi_flat_q + 3 * phi0_q,
        ])
        rho_rep_q = np.tile(rho_flat_q, 4)
        rad_rep_q = np.tile(rad_q, 4)

        rho_rad_q = rho_rep_q * rad_rep_q
        x4 = np.cos(phi_4) * rho_rad_q
        y4 = np.sin(phi_4) * rho_rad_q

        w_2d_q = np.outer(w1q, w2q).ravel()
        w4 = np.tile(w_2d_q * rho_flat_q * rad_q ** 2, 4)
        w4 = 4 * w4 / np.sum(w4)

        self.x4 = x4
        self.y4 = y4
        self.w4 = w4

    def adapt(self, *args: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        MATLAB: @quadface/private/adapt.m

        Adapt stored integration points to boundary element.
        """
        # handle different input forms
        if len(args) == 1:
            verts = np.asarray(args[0])
        else:
            verts = np.vstack(args)

        if verts.shape[0] == 3:
            return self._adaptrule(verts, [0, 1, 2])
        else:
            # divide quadrilateral into two triangles
            pos_a, w_a = self._adaptrule(verts, [0, 1, 2])
            pos_b, w_b = self._adaptrule(verts, [2, 3, 0])
            # combine
            total = pos_a.shape[0] + pos_b.shape[0]
            pos = np.empty((total, verts.shape[1]))
            pos[:pos_a.shape[0]] = pos_a
            pos[pos_a.shape[0]:] = pos_b
            w = np.empty(w_a.shape[0] + w_b.shape[0])
            w[:w_a.shape[0]] = w_a
            w[w_a.shape[0]:] = w_b
            return pos, w

    def _adaptrule(self, verts: np.ndarray,
            tri: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        MATLAB: @quadface/private/adaptrule.m

        Adapt triangle integration.
        """
        v1 = verts[tri[0], :]
        v2 = verts[tri[1], :]
        v3 = verts[tri[2], :]

        xc = self.x.ravel()[:, np.newaxis]
        yc = self.y.ravel()[:, np.newaxis]

        # linear triangle integration
        pos = xc * v1[np.newaxis, :] + yc * v2[np.newaxis, :] + (1 - xc - yc) * v3[np.newaxis, :]

        # normal vector
        nvec = np.cross(v1 - v3, v2 - v3)
        # integration weight
        w = 0.5 * self.w * np.sqrt(np.dot(nvec, nvec))

        return pos, w

    def plot(self) -> None:
        """
        MATLAB: integration/@quadface/plot.m

        Plot integration points.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

        # triangle integration points
        ax1.plot(self.x, self.y, 'b.', markersize = 3)
        # polar triangle integration points
        ax1.plot(1 - self.x3, 1 - self.y3, 'r.', markersize = 3)
        # triangle outline
        ax1.plot([0, 1, 0, 0], [0, 0, 1, 0], 'k-')
        ax1.plot([1, 1, 0, 1], [0, 1, 1, 0], 'k-')
        ax1.set_aspect('equal')
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('nz = {} ({})'.format(len(self.x), len(self.x3)))

        # polar quadrilateral integration points
        ax2.plot(self.x4, self.y4, 'r.', markersize = 3)
        ax2.set_aspect('equal')
        ax2.set_xlim(-1.05, 1.05)
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('nz = {}'.format(len(self.x4)))

        plt.tight_layout()

    def __call__(self, *args: Any) -> Tuple[np.ndarray, np.ndarray]:
        return self.adapt(*args)

    def __repr__(self) -> str:
        return 'QuadFace(npts={}, npol={})'.format(len(self.x), self.npol)


def triangle_unit_set(rule: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB: integration/@quadface/private/triangle_unit_set.m

    Sets a quadrature rule in a unit triangle.
    Returns (x, y, w) arrays for integration points and weights.

    Rules 1-19 with increasing precision.
    """
    if rule == 1:
        # 1 point, precision 1
        a = 1.0 / 3.0
        xtab = np.array([a])
        ytab = np.array([a])
        weight = np.array([1.0])

    elif rule == 2:
        # 3 points, precision 2, Strang and Fix formula #1
        xtab = np.array([4.0, 1.0, 1.0]) / 6.0
        ytab = np.array([1.0, 4.0, 1.0]) / 6.0
        weight = np.array([1.0, 1.0, 1.0]) / 3.0

    elif rule == 3:
        # 3 points, precision 2, Strang and Fix formula #2
        a = 0.5
        c = 1.0 / 3.0
        xtab = np.array([0.0, a, a])
        ytab = np.array([a, 0.0, a])
        weight = np.array([c, c, c])

    elif rule == 4:
        # 4 points, precision 3
        xtab = np.array([10.0, 18.0, 6.0, 6.0]) / 30.0
        ytab = np.array([10.0, 6.0, 18.0, 6.0]) / 30.0
        weight = np.array([-27.0, 25.0, 25.0, 25.0]) / 48.0

    elif rule == 5:
        # 6 points, precision 3
        a = 0.659027622374092
        b = 0.231933368553031
        c = 0.109039009072877
        w = 1.0 / 6.0
        xtab = np.array([a, a, b, b, c, c])
        ytab = np.array([b, c, a, c, a, b])
        weight = np.array([w, w, w, w, w, w])

    elif rule == 6:
        # 6 points, precision 3, Stroud T2:3-1
        a = 0.0
        b = 0.5
        c = 2.0 / 3.0
        d = 1.0 / 6.0
        v = 1.0 / 30.0
        w = 3.0 / 10.0
        xtab = np.array([a, b, b, c, d, d])
        ytab = np.array([b, a, b, d, c, d])
        weight = np.array([v, v, v, w, w, w])

    elif rule == 7:
        # 6 points, precision 4
        a = 0.816847572980459
        b = 0.091576213509771
        c = 0.108103018168070
        d = 0.445948490915965
        v = 0.109951743655322
        w = 0.223381589678011
        xtab = np.array([a, b, b, c, d, d])
        ytab = np.array([b, a, b, d, c, d])
        weight = np.array([v, v, v, w, w, w])

    elif rule == 8:
        # 7 points, precision 4
        a = 1.0 / 3.0
        c = 0.736712498968435
        d = 0.237932366472434
        e = 0.025355134551932
        v = 0.375
        w = 0.104166666666667
        xtab = np.array([a, c, c, d, d, e, e])
        ytab = np.array([a, d, e, c, e, c, d])
        weight = np.array([v, w, w, w, w, w, w])

    elif rule == 9:
        # 7 points, precision 5
        a = 1.0 / 3.0
        b = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
        c = (6.0 - np.sqrt(15.0)) / 21.0
        d = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
        e = (6.0 + np.sqrt(15.0)) / 21.0
        u = 0.225
        v = (155.0 - np.sqrt(15.0)) / 1200.0
        w = (155.0 + np.sqrt(15.0)) / 1200.0
        xtab = np.array([a, b, c, c, d, e, e])
        ytab = np.array([a, c, b, c, e, d, e])
        weight = np.array([u, v, v, v, w, w, w])

    elif rule == 10:
        # 9 points, precision 6
        a = 0.124949503233232
        b = 0.437525248383384
        c = 0.797112651860071
        d = 0.165409927389841
        e = 0.037477420750088
        u = 0.205950504760887
        v = 0.063691414286223
        xtab = np.array([a, b, b, c, c, d, d, e, e])
        ytab = np.array([b, a, b, d, e, c, e, c, d])
        weight = np.array([u, u, u, v, v, v, v, v, v])

    elif rule == 11:
        # 12 points, precision 6
        a = 0.873821971016996
        b = 0.063089014491502
        c = 0.501426509658179
        d = 0.249286745170910
        e = 0.636502499121399
        f = 0.310352451033785
        g = 0.053145049844816
        u = 0.050844906370207
        v = 0.116786275726379
        w = 0.082851075618374
        xtab = np.array([a, b, b, d, c, d, e, e, f, f, g, g])
        ytab = np.array([b, a, b, c, d, d, f, g, e, g, e, f])
        weight = np.array([u, u, u, v, v, v, w, w, w, w, w, w])

    elif rule == 12:
        # 13 points, precision 7
        a = 0.479308067841923
        b = 0.260345966079038
        c = 0.869739794195568
        d = 0.065130102902216
        e = 0.638444188569809
        f = 0.312865496004875
        g = 0.048690315425316
        h = 1.0 / 3.0
        t = 0.175615257433204
        u = 0.053347235608839
        v = 0.077113760890257
        w = -0.149570044467670
        xtab = np.array([a, b, b, c, d, d, e, e, f, f, g, g, h])
        ytab = np.array([b, a, b, d, c, d, f, g, e, g, e, f, h])
        weight = np.array([t, t, t, u, u, u, v, v, v, v, v, v, w])

    elif rule == 13:
        # 7 points
        a = 1.0 / 3.0
        b = 1.0
        c = 0.5
        z = 0.0
        u = 27.0 / 60.0
        v = 3.0 / 60.0
        w = 8.0 / 60.0
        xtab = np.array([a, b, z, z, z, c, c])
        ytab = np.array([a, z, b, z, c, z, c])
        weight = np.array([u, v, v, v, w, w, w])

    elif rule == 14:
        # 16 points, conical product Gauss
        xtab1 = np.array([
            -0.861136311594052575223946488893,
            -0.339981043584856264802665759103,
            0.339981043584856264802665759103,
            0.861136311594052575223946488893])
        weight1 = np.array([
            0.347854845137453857373063949222,
            0.652145154862546142626936050778,
            0.652145154862546142626936050778,
            0.347854845137453857373063949222])
        xtab1 = 0.5 * (xtab1 + 1.0)

        weight2 = np.array([0.1355069134, 0.2034645680, 0.1298475476, 0.0311809709])
        xtab2 = np.array([0.0571041961, 0.2768430136, 0.5835904324, 0.8602401357])

        xtab = np.empty(16)
        ytab = np.empty(16)
        weight = np.empty(16)
        k = 0
        for i in range(4):
            for j in range(4):
                xtab[k] = xtab2[j]
                ytab[k] = xtab1[i] * (1.0 - xtab2[j])
                weight[k] = weight1[i] * weight2[j]
                k += 1

    elif rule == 15:
        # 64 points, precision 15
        xtab1 = np.array([
            -0.960289856497536231683560868569,
            -0.796666477413626739591553936476,
            -0.525532409916328985817739049189,
            -0.183434642495649804939476142360,
            0.183434642495649804939476142360,
            0.525532409916328985817739049189,
            0.796666477413626739591553936476,
            0.960289856497536231683560868569])
        weight1 = np.array([
            0.101228536290376259152531354310,
            0.222381034453374470544355994426,
            0.313706645877887287337962201987,
            0.362683783378361982965150449277,
            0.362683783378361982965150449277,
            0.313706645877887287337962201987,
            0.222381034453374470544355994426,
            0.101228536290376259152531354310])
        weight2 = np.array([
            0.00329519144, 0.01784290266, 0.04543931950, 0.07919959949,
            0.10604735944, 0.11250579947, 0.09111902364, 0.04455080436])
        xtab2 = np.array([
            0.04463395529, 0.14436625704, 0.28682475714, 0.45481331520,
            0.62806783542, 0.78569152060, 0.90867639210, 0.98222008485])

        xtab = np.empty(64)
        ytab = np.empty(64)
        weight = np.empty(64)
        k = 0
        for j in range(8):
            for i in range(8):
                xtab[k] = 1.0 - xtab2[j]
                ytab[k] = 0.5 * (1.0 + xtab1[i]) * xtab2[j]
                weight[k] = weight1[i] * weight2[j]
                k += 1

    elif rule == 16:
        # 19 points, precision 8
        a = 1.0 / 3.0
        b = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
        c = (6.0 - np.sqrt(15.0)) / 21.0
        d = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
        e = (6.0 + np.sqrt(15.0)) / 21.0
        f = (40.0 - 10.0 * np.sqrt(15.0) + 10.0 * np.sqrt(7.0) + 2.0 * np.sqrt(105.0)) / 90.0
        g = (25.0 + 5.0 * np.sqrt(15.0) - 5.0 * np.sqrt(7.0) - np.sqrt(105.0)) / 90.0
        p = (40.0 + 10.0 * np.sqrt(15.0) + 10.0 * np.sqrt(7.0) - 2.0 * np.sqrt(105.0)) / 90.0
        q = (25.0 - 5.0 * np.sqrt(15.0) - 5.0 * np.sqrt(7.0) + np.sqrt(105.0)) / 90.0
        r = (40.0 + 10.0 * np.sqrt(7.0)) / 90.0
        s = (25.0 + 5.0 * np.sqrt(15.0) - 5.0 * np.sqrt(7.0) - np.sqrt(105.0)) / 90.0
        t = (25.0 - 5.0 * np.sqrt(15.0) - 5.0 * np.sqrt(7.0) + np.sqrt(105.0)) / 90.0

        w1 = (7137.0 - 1800.0 * np.sqrt(7.0)) / 62720.0
        w2 = (-9301697.0 / 4695040.0 - 13517313.0 * np.sqrt(15.0) / 23475200.0 + 764885.0 * np.sqrt(7.0) / 939008.0 + 198763.0 * np.sqrt(105.0) / 939008.0) / 3.0
        w3 = (-9301697.0 / 4695040.0 + 13517313.0 * np.sqrt(15.0) / 23475200.0 + 764885.0 * np.sqrt(7.0) / 939008.0 - 198763.0 * np.sqrt(105.0) / 939008.0) / 3.0
        w4 = (102791225.0 - 23876225.0 * np.sqrt(15.0) - 34500875.0 * np.sqrt(7.0) + 9914825.0 * np.sqrt(105.0)) / 59157504.0 / 3.0
        w5 = (102791225.0 + 23876225.0 * np.sqrt(15.0) - 34500875.0 * np.sqrt(7.0) - 9914825.0 * np.sqrt(105.0)) / 59157504.0 / 3.0
        w6 = (11075.0 - 3500.0 * np.sqrt(7.0)) / 8064.0 / 6.0

        xtab = np.array([a, b, c, c, d, e, e, f, g, g, p, q, q, r, r, s, s, t, t])
        ytab = np.array([a, c, b, c, e, d, e, g, f, g, q, p, q, s, t, r, t, r, s])
        weight = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w5, w5, w5, w6, w6, w6, w6, w6, w6])

    elif rule == 17:
        # 19 points, precision 9
        a = 1.0 / 3.0
        b = 0.02063496160252593
        c = 0.4896825191987370
        d = 0.1258208170141290
        e = 0.4370895914929355
        f = 0.6235929287619356
        g = 0.1882035356190322
        r = 0.9105409732110941
        s = 0.04472951339445297
        t = 0.7411985987844980
        u = 0.03683841205473626
        v = 0.22196288916076574

        w1 = 0.09713579628279610
        w2 = 0.03133470022713983
        w3 = 0.07782754100477543
        w4 = 0.07964773892720910
        w5 = 0.02557767565869810
        w6 = 0.04328353937728940

        xtab = np.array([a, b, c, c, d, e, e, f, g, g, r, s, s, t, t, u, u, v, v])
        ytab = np.array([a, c, b, c, e, d, e, g, f, g, s, r, s, u, v, t, v, t, u])
        weight = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w5, w5, w5, w6, w6, w6, w6, w6, w6])

    elif rule == 18:
        # 28 points, precision 11
        a = 1.0 / 3.0
        b = 0.9480217181434233
        c = 0.02598914092828833
        d = 0.8114249947041546
        e = 0.09428750264792270
        f = 0.01072644996557060
        g = 0.4946367750172147
        p = 0.5853132347709715
        q = 0.2073433826145142
        r = 0.1221843885990187
        s = 0.4389078057004907
        t = 0.6779376548825902
        u = 0.04484167758913055
        v = 0.27722066752827925
        w_val = 0.8588702812826364
        x_val = 0.0
        y_val = 0.1411297187173636

        w1 = 0.08797730116222190
        w2 = 0.008744311553736190
        w3 = 0.03808157199393533
        w4 = 0.01885544805613125
        w5 = 0.07215969754474100
        w6 = 0.06932913870553720
        w7 = 0.04105631542928860
        w8 = 0.007362383783300573

        xtab = np.array([a, b, c, c, d, e, e, f, g, g, p, q, q,
                         r, s, s, t, t, u, u, v, v, w_val, w_val, x_val, x_val, y_val, y_val])
        ytab = np.array([a, c, b, c, e, d, e, g, f, g, q, p, q,
                         s, r, s, u, v, t, v, t, u, x_val, y_val, w_val, y_val, w_val, x_val])
        weight = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4, w5, w5, w5,
                           w6, w6, w6, w7, w7, w7, w7, w7, w7, w8, w8, w8, w8, w8, w8])

    elif rule == 19:
        # 37 points, precision 13
        a = 1.0 / 3.0
        b = 0.950275662924105565450352089520
        c = 0.024862168537947217274823955239
        d = 0.171614914923835347556304795551
        e = 0.414192542538082326221847602214
        f = 0.539412243677190440263092985511
        g = 0.230293878161404779868453507244

        w1 = 0.051739766065744133555179145422
        w2 = 0.008007799555564801597804123460
        w3 = 0.046868898981821644823226732071
        w4 = 0.046590940183976487960361770070
        w5 = 0.031016943313796381407646220131
        w6 = 0.010791612736631273623178240136
        w7 = 0.032195534242431618819414482205
        w8 = 0.015445834210701583817692900053
        w9 = 0.017822989923178661888748319485
        wx = 0.037038683681384627918546472190

        xtab = np.array([a, b, c, c, d, e, e, f, g, g])
        ytab = np.array([a, c, b, c, e, d, e, g, f, g])
        weight_arr = np.array([w1, w2, w2, w2, w3, w3, w3, w4, w4, w4,
                               w5, w5, w5, w6, w6, w6, w7, w7, w7,
                               w8, w8, w8, w8, w8, w8,
                               w9, w9, w9, w9, w9, w9,
                               wx, wx, wx, wx, wx, wx])

        # additional points
        a2 = 0.772160036676532561750285570113
        b2 = 0.113919981661733719124857214943
        xtab = np.append(xtab, [a2, b2, b2])
        ytab = np.append(ytab, [b2, a2, b2])

        a3 = 0.009085399949835353883572964740
        b3 = 0.495457300025082323058213517632
        xtab = np.append(xtab, [a3, b3, b3])
        ytab = np.append(ytab, [b3, a3, b3])

        a4 = 0.062277290305886993497083640527
        b4 = 0.468861354847056503251458179727
        xtab = np.append(xtab, [a4, b4, b4])
        ytab = np.append(ytab, [b4, a4, b4])

        a5 = 0.022076289653624405142446876931
        b5 = 0.851306504174348550389457672223
        c5 = 1.0 - a5 - b5
        xtab = np.append(xtab, [a5, a5, b5, b5, c5, c5])
        ytab = np.append(ytab, [b5, c5, a5, c5, a5, b5])

        a6 = 0.018620522802520968955913511549
        b6 = 0.689441970728591295496647976487
        c6 = 1.0 - a6 - b6
        xtab = np.append(xtab, [a6, a6, b6, b6, c6, c6])
        ytab = np.append(ytab, [b6, c6, a6, c6, a6, b6])

        a7 = 0.096506481292159228736516560903
        b7 = 0.635867859433872768286976979827
        c7 = 1.0 - a7 - b7
        xtab = np.append(xtab, [a7, a7, b7, b7, c7, c7])
        ytab = np.append(ytab, [b7, c7, a7, c7, a7, b7])

        weight = weight_arr

    else:
        raise ValueError('[error] Invalid <rule> = {}!'.format(rule))

    return xtab, ytab, weight


def trisubdivide(xtab: np.ndarray, ytab: np.ndarray,
        wtab: np.ndarray,
        nsub: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB: integration/@quadface/private/trisubdivide.m

    Refines triangle integration of unit triangle.
    """
    x_list = []
    y_list = []
    w_list = []

    h = 1.0 / nsub

    for i in range(nsub):
        for j in range(nsub - i):
            # triangle pointing upwards
            x_list.append(i + xtab)
            y_list.append(j + ytab)
            w_list.append(wtab.copy())

            # triangle pointing downwards
            if j != nsub - 1 - i:
                x_list.append(i + 1 - xtab)
                y_list.append(j + 1 - ytab)
                w_list.append(wtab.copy())

    x_out = np.hstack(x_list) * h
    y_out = np.hstack(y_list) * h
    w_out = np.hstack(w_list)
    n_sub_tri = len(x_out) // len(xtab)
    w_out = w_out / n_sub_tri

    return x_out, y_out, w_out
