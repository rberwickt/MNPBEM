"""
Compound particle with multiple materials.

Matches MATLAB MNPBEM @comparticle implementation exactly.
"""

import numpy as np
from .particle import Particle


class ComParticle(object):
    """
    Compound particle with multiple dielectric media.

    Combines multiple particles with different material properties,
    specifying which dielectric function applies inside and outside
    each particle surface.

    MATLAB: @comparticle (inherits from @compound)

    Parameters
    ----------
    eps : list of material objects
        List of dielectric functions (EpsConst, EpsTable, etc.)
    particles : list of Particle
        List of particle objects
    inout : list or ndarray, shape (nparticles, 2)
        For each particle: [inside_eps_index, outside_eps_index]
        Indices refer to the eps list (1-indexed like MATLAB)
    closed_args : tuple, optional
        Arguments passed to closed() method
    **kwargs : dict
        Options (interp='curv'/'flat', etc.)

    Attributes
    ----------
    eps : list
        Dielectric functions
    p : list of Particle
        Particle geometries
    inout : ndarray
        Inside/outside dielectric indices
    closed : list
        Closed surface information for each particle
    pc : Particle
        Concatenated particle (vertcat of all particles)
    nverts : int
        Total number of vertices
    nfaces : int
        Total number of faces
    np : int
        Number of unique material boundaries
    """

    def __init__(self, eps, particles, inout, *closed_args, **kwargs):
        """
        Initialize compound particle.

        MATLAB: obj = comparticle(eps, p, inout, varargin)
        """
        # Validate inputs (C1-4..C1-8)
        if eps is None:
            raise ValueError("ComParticle: 'eps' must be a list of dielectric functions, got None.")
        if particles is None:
            raise ValueError("ComParticle: 'particles' must be a list of Particle, got None.")
        if inout is None:
            raise ValueError("ComParticle: 'inout' must be an inside/outside index table, got None.")

        if not hasattr(eps, '__len__'):
            raise TypeError("ComParticle: 'eps' must be a sequence (list) of dielectric functions.")
        if len(eps) == 0:
            raise ValueError("ComParticle: 'eps' must contain at least one dielectric function.")

        self.eps = eps

        # Process input particles and options (MATLAB: getinput.m)
        particles, closed_args = self._getinput(particles, closed_args, kwargs)

        # Handle particle list (may be nested list from MATLAB style)
        if isinstance(particles, list):
            if len(particles) > 0 and isinstance(particles[0], list):
                # Flatten nested list: [[p1]] -> [p1]
                self.p = [item for sublist in particles for item in sublist]
            else:
                self.p = particles
        else:
            self.p = [particles]

        # Convert inout to numpy array (MATLAB uses 1-indexing)
        inout_arr = np.atleast_2d(np.asarray(inout))
        if inout_arr.ndim != 2 or inout_arr.shape[1] != 2:
            raise ValueError(
                "ComParticle: 'inout' must be a list of [in, out] pairs with "
                "shape (n, 2); got shape {}.".format(inout_arr.shape))
        if inout_arr.shape[0] != len(self.p):
            raise ValueError(
                "ComParticle: 'inout' must have one row per particle; got "
                "{} particles vs {} inout rows."
                .format(len(self.p), inout_arr.shape[0]))
        # eps indices in inout are 1-based (MATLAB) — must be within range.
        max_idx = int(inout_arr.max()) if inout_arr.size > 0 else 0
        if max_idx > len(eps):
            raise ValueError(
                "ComParticle: 'inout' references eps[{}] but only {} "
                "dielectric functions provided.".format(max_idx, len(eps)))
        if int(inout_arr.min()) < 1:
            raise ValueError(
                "ComParticle: 'inout' indices are 1-based; got minimum {}."
                .format(int(inout_arr.min())))
        self.inout = inout_arr

        # Mask (default: all particles active)
        self._mask = list(range(len(self.p)))

        # Initialize closed surfaces (MATLAB: init.m)
        # Note: MATLAB demos always pass per-sub-particle closed args
        # (e.g. ``comparticle(eps, {p1, p2}, [...], 1, 2, op)``). When
        # closed_args is empty here, ``compgreen_ret._initclosed`` applies a
        # default per-sub-particle closed convention so the Fuchs-Liu
        # surface-integral identity (-2*pi for closed surface) holds.
        self.closed = [None] * len(self.p)
        if len(closed_args) > 0:
            self.set_closed(*closed_args)

        # Compute auxiliary properties and create pc
        self._compute_properties()
        self._norm()

    def _getinput(self, particles, closed_args, kwargs):
        """
        Extract options for particles and get closed arguments.

        MATLAB: getinput.m
        """
        # Apply interp option to all particles if specified
        if 'interp' in kwargs:
            interp = kwargs['interp']
            for p in particles:
                if interp == 'curv':
                    p.curved()
                else:
                    p.flat()

        # MATLAB getinput.m: re-init p.quad with bemoptions refine/rule/npol
        quad_kwargs = {}
        if 'rule' in kwargs:
            quad_kwargs['rule'] = kwargs['rule']
        if 'npol' in kwargs:
            quad_kwargs['npol'] = kwargs['npol']
        if 'refine' in kwargs:
            quad_kwargs['refine'] = kwargs['refine']
        if quad_kwargs:
            from ..utils.quadface import QuadFace as _QuadFace
            _rule = quad_kwargs.get('rule', 18)
            _npol = quad_kwargs.get('npol', (7, 5))
            _refine = quad_kwargs.get('refine', None)
            for p in particles:
                p.quad = _QuadFace(rule=_rule, npol=_npol, refine=_refine)
            # Persist for later application to self.pc, which is re-built by
            # Particle.vertcat (in _norm) with a fresh default quad.
            self._quad_kwargs = dict(rule=_rule, npol=_npol, refine=_refine)

        return particles, closed_args

    def _norm(self):
        """
        Compute auxiliary information for discretized particle surface.

        MATLAB: norm(obj)
        """
        # Create concatenated particle (vertcat all particles)
        if len(self.p) > 0:
            self.pc = self.p[0]
            for p in self.p[1:]:
                self.pc = self.pc + p
        else:
            self.pc = Particle(np.array([]).reshape(0, 3),
                              np.array([]).reshape(0, 4))

        # Particle.vertcat rebuilds `pc` with a default-quad Particle. Restore
        # the bemoptions-supplied quadrature on `pc` so cover-layer / near-field
        # polar integration uses the requested npol/rule/refine.
        qk = getattr(self, '_quad_kwargs', None)
        if qk is not None and hasattr(self.pc, 'quad'):
            from ..utils.quadface import QuadFace as _QuadFace
            self.pc.quad = _QuadFace(**qk)

    def _compute_properties(self):
        """Compute derived properties."""
        # Total number of vertices and faces
        self.nverts = sum(part.nverts for part in self.p)
        self.nfaces = sum(part.nfaces for part in self.p)

        # Number of particles
        self.np = len(self.p)

        # Create index mapping for boundary elements
        self._create_index()

    def _create_index(self):
        """
        Create index array mapping faces to material boundaries.

        MATLAB: compound.index property
        """
        self.index = np.zeros(self.nfaces, dtype=int)

        # Create mapping from (eps_in, eps_out) to unique index
        unique_pairs = []
        pair_to_idx = {}

        for pair in self.inout:
            pair_tuple = tuple(pair)
            if pair_tuple not in pair_to_idx:
                pair_to_idx[pair_tuple] = len(unique_pairs)
                unique_pairs.append(pair_tuple)

        # Assign indices to each face
        offset = 0
        for i, part in enumerate(self.p):
            pair = tuple(self.inout[i])
            idx = pair_to_idx[pair]
            self.index[offset:offset + part.nfaces] = idx
            offset += part.nfaces

    # ==================== Closed surfaces ====================

    def set_closed(self, *args):
        """
        Indicate closed surfaces of particles (for use in compgreen).

        MATLAB: closed(obj, varargin)

        Usage
        -----
        obj.set_closed([i1, i2, ...])
            Closed surface of particles i1, i2, ...
        obj.set_closed({i1, p1, p2, ...})
            Closed surface of particle i1 and particles p1, p2, ...
        """
        for arg in args:
            # Input is index to particle(s) stored in obj
            if not isinstance(arg, (list, tuple)) or not isinstance(arg[0], (list, Particle)):
                # Simple list of indices
                if np.isscalar(arg):
                    indices = [arg]
                else:
                    indices = arg

                for ind in indices:
                    ind_abs = abs(ind)
                    # Set closed property if not previously set (1-indexed!).
                    # Store a copy so subsequent mutations of the input or of
                    # one entry's list do not alias other sub-particles.
                    if self.closed[ind_abs - 1] is None:
                        self.closed[ind_abs - 1] = list(indices)
            # Input is an additional particle
            else:
                idx = arg[0] if np.isscalar(arg[0]) else arg[0][0]
                # Vertcat particles
                particles_to_concat = [self.p[idx - 1]] + list(arg[1:])
                combined = particles_to_concat[0]
                for p in particles_to_concat[1:]:
                    combined = combined + p
                self.closed[idx - 1] = combined

    def closedparticle(self, ind):
        """
        Return particle with closed surface for indexed particle.

        MATLAB: [p, dir, loc] = closedparticle(obj, ind)

        Parameters
        ----------
        ind : int
            Particle index (1-indexed like MATLAB)

        Returns
        -------
        p : Particle or None
            Closed particle (None if not closed)
        dir : int or None
            Outer (dir=1) or inner (dir=-1) surface normal
        loc : ndarray or None
            If closed particle is contained in pc, loc points to the
            elements of the closed particle (None otherwise)
        """
        idx = ind - 1  # Convert to 0-indexed

        if self.closed[idx] is None:
            return None, None, None

        elif isinstance(self.closed[idx], Particle):
            return self.closed[idx], 1, None

        else:
            closed_list = self.closed[idx]
            # Find direction
            dir_val = None
            for c in closed_list:
                if abs(c) == ind:
                    dir_val = np.sign(c) if c != 0 else 1
                    break

            if dir_val is None:
                dir_val = 1

            # Put together closed particle surface
            abs_closed = [abs(c) for c in closed_list]
            sign_closed = [np.sign(c) if c != 0 else 1 for c in closed_list]

            particles = [self.p[i - 1] for i in abs_closed]

            # Flip faces where direction doesn't match
            for i, (p, sign) in enumerate(zip(particles, sign_closed)):
                if sign != dir_val:
                    particles[i] = p.flipfaces()

            # Vertcat all particles
            p_combined = particles[0]
            for p in particles[1:]:
                p_combined = p_combined + p

            # Index to closed particle
            if all(c > 0 for c in closed_list):
                # Vectorized match: for each row in p_combined.pos, find the
                # nearest row in self.pc.pos.  Replaces the previous O(n^2)
                # Python ``np.allclose`` double loop, which dominated
                # ``closedparticle`` runtime for large meshes (e.g. n=284 ->
                # 2.1s).  Use a single broadcasted ||.||_inf <= atol test.
                pos_a = np.asarray(p_combined.pos)
                pos_b = np.asarray(self.pc.pos)
                if pos_a.size and pos_b.size:
                    diff = np.abs(pos_a[:, None, :] - pos_b[None, :, :])
                    matches = np.all(diff <= 1e-8, axis = 2)
                    has_match = matches.any(axis = 1)
                    if has_match.all():
                        loc = matches.argmax(axis = 1)
                    else:
                        loc = None
                else:
                    loc = None
            else:
                loc = None

            return p_combined, dir_val, loc

    # ==================== Selection ====================

    def select(self, **kwargs):
        """
        Select faces in comparticle object.

        MATLAB: obj = select(obj, 'PropertyName', PropertyValue)

        Parameters
        ----------
        index : array_like, optional
            Index to selected elements
        carfun : callable, optional
            Function f(x, y, z) for selected elements
        polfun : callable, optional
            Function f(phi, r, z) for selected elements
        sphfun : callable, optional
            Function f(phi, theta, r) for selected elements

        Returns
        -------
        obj : ComParticle
            Selected comparticle
        """
        if 'index' not in kwargs:
            # Pass select input to all particle objects
            new_particles = []
            for p in self.p:
                p_selected, _ = p.select(**kwargs)
                new_particles.append(p_selected)
            self.p = new_particles
        else:
            # Index to grouped particles
            index = kwargs['index']

            # Create particle index for each face
            ipt = []
            for i, p in enumerate(self.p):
                ipt.extend([i] * p.nfaces)
            ipt = np.array(ipt)

            # Point index (global face index)
            ind_global = []
            for p in self.p:
                ind_global.extend(range(p.nfaces))
            ind_global = np.array(ind_global)

            # Get selected indices
            ind_selected = ind_global[index]
            ipt_selected = ipt[index]

            # Loop over all particles
            new_particles = []
            for i in range(len(self.p)):
                # Get local indices for this particle
                mask = (ipt_selected == i)
                if mask.any():
                    local_ind = ind_selected[mask]
                    p_selected, _ = self.p[i].select(index=local_ind)
                    new_particles.append(p_selected)

            self.p = new_particles

        # Keep only non-empty particles
        non_empty = [i for i, p in enumerate(self.p) if p.nfaces > 0]
        self.p = [self.p[i] for i in non_empty]
        self.inout = self.inout[non_empty, :]

        # Reset closed arguments
        self.closed = [None] * len(self.p)

        # Update mask
        self._mask = list(range(len(self.p)))

        # Update compound particle
        self._norm()

        return self

    # ==================== Wrapper methods ====================

    def clean(self, *args, **kwargs):
        """
        Apply particle.clean() to all particles.

        MATLAB: clean(obj, varargin)
        """
        self.p = [p.clean(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def flip(self, *args, **kwargs):
        """
        Apply particle.flip() to all particles.

        MATLAB: flip(obj, varargin)
        """
        self.p = [p.flip(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def flipfaces(self, *args, **kwargs):
        """
        Apply particle.flipfaces() to all particles.

        MATLAB: flipfaces(obj, varargin)
        """
        self.p = [p.flipfaces(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def rot(self, *args, **kwargs):
        """
        Apply particle.rot() to all particles.

        MATLAB: rot(obj, varargin)
        """
        self.p = [p.rot(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def scale(self, *args, **kwargs):
        """
        Apply particle.scale() to all particles.

        MATLAB: scale(obj, varargin)
        """
        self.p = [p.scale(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def shift(self, *args, **kwargs):
        """
        Apply particle.shift() to all particles.

        MATLAB: shift(obj, varargin)
        """
        self.p = [p.shift(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    # ==================== Delegation to pc ====================

    def deriv(self, v):
        """
        Tangential derivative of function defined on surface.

        MATLAB: [v1, v2, t1, t2] = deriv(obj, v)
        """
        return self.pc.deriv(v)

    def interp_values(self, v, method='area'):
        """
        Interpolate values from faces to vertices or vice versa.

        MATLAB: [vi, mat] = interp(obj, v, key)
        """
        return self.pc.interp_values(v, method)

    def curvature(self):
        """
        Curvature of particle.

        MATLAB: curv = curvature(obj, varargin)
        """
        return self.pc.curvature()

    def quad_integration(self, ind=None):
        """
        Integration over boundary elements.

        MATLAB: [pos, w, iface] = quad(obj, ind)
        """
        return self.pc.quad_integration(ind)

    def quadpol(self, ind=None):
        """
        Integration over boundary elements using polar coordinates.

        MATLAB: [pos, weight, row] = quadpol(obj, ind)
        """
        return self.pc.quadpol(ind)

    def vertices(self, ind, close=False):
        """
        Vertices of indexed face.

        MATLAB: v = vertices(obj, ind, 'close')

        Parameters
        ----------
        ind : int
            Global face index
        close : bool
            If True, close the face indices

        Returns
        -------
        v : ndarray
            Vertices of the face
        """
        # Find which particle this face belongs to
        ip, local_ind = self._ipart(ind)
        return self.p[ip].vertices(local_ind, close)

    def _ipart(self, ind):
        """
        Return particle and face index for global face index.

        MATLAB: [ip, ind] = ipart(obj, ind)

        Parameters
        ----------
        ind : int
            Global face index

        Returns
        -------
        ip : int
            Particle index (0-indexed)
        local_ind : int
            Local face index within that particle
        """
        offset = 0
        for i, p in enumerate(self.p):
            if ind < offset + p.nfaces:
                return i, ind - offset
            offset += p.nfaces
        raise IndexError("Face index {} out of range".format(ind))

    def plot(self, val=None, **kwargs):
        """
        Plot discretized particle surface.

        MATLAB: plot(obj, val, 'PropertyName', PropertyValue, ...)
        """
        return self.pc.plot(val, **kwargs)

    def plot2(self, val=None, **kwargs):
        """
        Advanced plot of discretized particle surface.

        MATLAB: plot2(obj, val, 'PropertyName', PropertyValue, ...)
        """
        return self.pc.plot2(val, **kwargs)

    # ==================== Dielectric functions ====================

    def eps1(self, enei):
        """
        Get inside dielectric constants at given wavelength.

        Parameters
        ----------
        enei : float or array
            Wavelength in nm

        Returns
        -------
        eps : ndarray
            Inside dielectric constants for each face
        """
        eps_vals = np.zeros(self.nfaces, dtype=complex)
        offset = 0

        for i, part in enumerate(self.p):
            eps_idx = int(self.inout[i, 0]) - 1  # Convert to 0-indexed
            eps_mat = self.eps[eps_idx]
            eps_val, _ = eps_mat(enei)

            # Broadcast to all faces of this particle
            eps_vals[offset:offset + part.nfaces] = complex(np.asarray(eps_val).flat[0])

            offset += part.nfaces

        return eps_vals

    def eps2(self, enei):
        """
        Get outside dielectric constants at given wavelength.

        Parameters
        ----------
        enei : float or array
            Wavelength in nm

        Returns
        -------
        eps : ndarray
            Outside dielectric constants for each face
        """
        eps_vals = np.zeros(self.nfaces, dtype=complex)
        offset = 0

        for i, part in enumerate(self.p):
            eps_idx = int(self.inout[i, 1]) - 1  # Convert to 0-indexed
            eps_mat = self.eps[eps_idx]
            eps_val, _ = eps_mat(enei)

            # Broadcast to all faces of this particle
            eps_vals[offset:offset + part.nfaces] = complex(np.asarray(eps_val).flat[0])

            offset += part.nfaces

        return eps_vals

    # ==================== Properties ====================

    @property
    def pos(self):
        """Centroid positions of all faces."""
        return self.pc.pos if hasattr(self, 'pc') else np.vstack([part.pos for part in self.p])

    @property
    def nvec(self):
        """Normal vectors of all faces."""
        return self.pc.nvec if hasattr(self, 'pc') else np.vstack([part.nvec for part in self.p])

    @property
    def area(self):
        """Areas of all faces."""
        return self.pc.area if hasattr(self, 'pc') else np.hstack([part.area for part in self.p])

    @property
    def verts(self):
        """All vertices (concatenated)."""
        return self.pc.verts if hasattr(self, 'pc') else np.vstack([part.verts for part in self.p])

    @property
    def faces(self):
        """All faces (concatenated with vertex offset)."""
        return self.pc.faces if hasattr(self, 'pc') else self._concat_faces()

    @property
    def inout_faces(self):
        """
        Get inside/outside material indices for each face.

        Returns array of shape (nfaces, 2) where:
        - Column 0: inside material index (1-indexed like MATLAB)
        - Column 1: outside material index (1-indexed like MATLAB)
        """
        inout_arr = np.zeros((self.nfaces, 2), dtype=int)
        offset = 0

        for i, part in enumerate(self.p):
            inout_arr[offset:offset + part.nfaces, 0] = self.inout[i, 0]
            inout_arr[offset:offset + part.nfaces, 1] = self.inout[i, 1]
            offset += part.nfaces

        return inout_arr

    def get_face_indices(self, medium, side='outside'):
        """
        Get indices of faces where the specified medium is on the given side.

        Parameters
        ----------
        medium : int
            Material index (1-indexed like MATLAB)
        side : str
            'inside' (column 0) or 'outside' (column 1)

        Returns
        -------
        indices : ndarray
            Array of face indices where the medium is on the specified side
        """
        col = 0 if side == 'inside' else 1
        return np.where(self.inout_faces[:, col] == medium)[0]

    @property
    def n(self):
        """Number of positions/faces (alias for nfaces, MATLAB compatibility)."""
        return self.nfaces

    @property
    def mask(self):
        """
        Mask array indicating which particles are active.

        MATLAB: obj.mask
        """
        mask_arr = np.zeros(len(self.p), dtype=bool)
        mask_arr[self._mask] = True
        return mask_arr

    def set_mask(self, ind):
        """
        Mask out particles indicated by ind.

        MATLAB: obj = mask(obj, ind)

        Parameters
        ----------
        ind : array_like
            Indices of particles to keep (1-indexed like MATLAB)
        """
        if np.isscalar(ind):
            ind = [ind]
        self._mask = [i - 1 for i in ind]  # Convert to 0-indexed
        self._norm()
        return self

    def index_func(self, particle_indices):
        """
        Get face indices for given particle indices.

        MATLAB: p.index(i) returns indices of faces belonging to particle i.

        Parameters
        ----------
        particle_indices : int or array
            Particle indices (1-indexed like MATLAB)

        Returns
        -------
        face_indices : ndarray
            Face indices corresponding to the particles
        """
        if np.isscalar(particle_indices):
            particle_indices = [particle_indices]

        face_indices = []
        offset = 0

        for i, part in enumerate(self.p):
            if (i + 1) in particle_indices:  # Convert to 1-indexed
                face_indices.extend(range(offset, offset + part.nfaces))
            offset += part.nfaces

        return np.array(face_indices, dtype=int)

    def bradius(self):
        """
        Minimal radius for spheres enclosing boundary elements.

        MATLAB: Particle/@particle/bradius.m (called on each particle)

        Returns
        -------
        r : ndarray
            Minimal radius for spheres enclosing each boundary element
            Concatenated from all particles in the composite

        Notes
        -----
        For composite particles, this concatenates the bradius values
        from all constituent particles.
        """
        # Concatenate bradius from all particles
        r_list = [part.bradius() for part in self.p]
        return np.hstack(r_list)

    # ==================== MATLAB @compound compatibility ====================

    def __eq__(self, other):
        """
        Test for equality between two compound objects (compare positions).

        MATLAB: @compound/eq.m
        """
        if not hasattr(other, 'pc') or not hasattr(other.pc, 'pos'):
            return NotImplemented
        if self.pc.pos.size != other.pc.pos.size:
            return False
        return np.all(self.pc.pos.ravel() == other.pc.pos.ravel())

    def __ne__(self, other):
        """
        Test for inequality between two compound objects.

        MATLAB: @compound/ne.m
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self):
        return id(self)

    @staticmethod
    def connect(*args):
        """
        Connectivity between compound points or particles.

        MATLAB: @compound/connect.m

        Usage
        -----
        con = ComParticle.connect(p1)
        con = ComParticle.connect(p1, p2)
        con = ComParticle.connect(p1, ind)
        con = ComParticle.connect(p1, p2, ind)

        Parameters
        ----------
        p1 : ComParticle
            First compound object
        p2 : ComParticle, optional
            Second compound object
        ind : ndarray, optional
            Replace dielectric materials according to ind

        Returns
        -------
        con : list of list of ndarray
            Cell array (n1 x n2) of arrays indicating whether compound
            points/particles can see each other. Non-zero entries indicate
            the shared medium index.
        """
        # Parse arguments
        if len(args) == 1:
            # Single particle
            p1 = args[0]
            inout1 = p1.inout[p1._mask, :]
            inout_list = [inout1]
        elif len(args) == 2:
            if isinstance(args[1], np.ndarray) and args[1].ndim == 1:
                # Single particle + index replacement
                p1 = args[0]
                ind = args[1]
                inout1 = ind[p1.inout[p1._mask, :] - 1]  # MATLAB is 1-indexed
                inout_list = [inout1]
            else:
                # Two particles
                p1, p2 = args[0], args[1]
                inout1 = p1.inout[p1._mask, :]
                inout2 = p2.inout[p2._mask, :]
                inout_list = [inout1, inout2]
        elif len(args) == 3:
            # Two particles + index replacement
            p1, p2, ind = args[0], args[1], args[2]
            inout1 = ind[p1.inout[p1._mask, :] - 1]
            inout2 = ind[p2.inout[p2._mask, :] - 1]
            inout_list = [inout1, inout2]
        else:
            raise ValueError('[error] Invalid number of arguments for connect()')

        # Compute connectivity matrix
        n1_cols = inout_list[0].shape[1]
        n2_cols = inout_list[-1].shape[1]
        con = [[None for _ in range(n2_cols)] for _ in range(n1_cols)]

        for i in range(n1_cols):
            for j in range(n2_cols):
                io1 = inout_list[0][:, i]
                io2 = inout_list[-1][:, j]
                c1 = np.tile(io1[:, np.newaxis], (1, len(io2)))
                c2 = np.tile(io2[np.newaxis, :], (len(io1), 1))
                result = np.zeros_like(c1)
                match = c1 == c2
                result[match] = c1[match]
                con[i][j] = result

        return con

    def dielectric(self, enei, inout):
        """
        Dielectric function at in- or outside.

        MATLAB: @compound/dielectric.m

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum
        inout : int
            Inside (1) or outside (2). Uses 1-indexed column.

        Returns
        -------
        eps_vals : list
            Dielectric function values for each masked particle
        """
        eps_table = [eps_fn(enei) for eps_fn in self.eps]

        if self.inout.size != len(self.p):
            # inout has 2 columns: pick the relevant column
            col = inout - 1  # Convert to 0-indexed column
            indices = self.inout[self._mask, col]
            return [eps_table[idx - 1] for idx in indices]  # 1-indexed to 0-indexed
        else:
            return eps_table

    def expand(self, val):
        """
        Expand cell array for all point or particle positions.

        MATLAB: @compound/expand.m (private)

        Parameters
        ----------
        val : scalar, ndarray, or list
            Value(s) to expand. If list, one entry per masked particle.

        Returns
        -------
        full : ndarray
            Expanded values, one per face of all masked particles
        """
        sizes = [self.p[i].nfaces for i in self._mask]

        if not isinstance(val, (list, tuple)):
            # Scalar or array: replicate for all faces
            total = sum(sizes)
            if np.isscalar(val):
                return np.full(total, val)
            else:
                return np.tile(np.asarray(val), (total, 1)) if np.ndim(val) > 0 else np.full(total, val)
        else:
            # List: replicate each value for the corresponding particle's faces
            parts = []
            for i, s in enumerate(sizes):
                v = np.asarray(val[i])
                if v.ndim == 0:
                    parts.append(np.full(s, v.item()))
                else:
                    parts.append(np.tile(v, (s, 1)) if v.ndim > 0 else np.full(s, v))

            # Pre-allocate and fill (avoid np.concatenate per CONVENTIONS)
            if len(parts) > 0 and parts[0].ndim > 1:
                total = sum(p.shape[0] for p in parts)
                result = np.empty((total, parts[0].shape[1]), dtype = parts[0].dtype)
                offset = 0
                for p in parts:
                    result[offset:offset + p.shape[0]] = p
                    offset += p.shape[0]
                return result
            else:
                total = sum(len(p) for p in parts)
                result = np.empty(total, dtype = parts[0].dtype if len(parts) > 0 else float)
                offset = 0
                for p in parts:
                    result[offset:offset + len(p)] = p
                    offset += len(p)
                return result

    def ipart(self, ind):
        """
        Find particle number and corresponding index for given global face index.

        MATLAB: @compound/ipart.m

        Parameters
        ----------
        ind : int or array_like
            Global face index (0-indexed)

        Returns
        -------
        particle_idx : ndarray
            Particle index (0-indexed) for each input index
        local_ind : ndarray
            Local face index within the particle
        """
        ind = np.atleast_1d(ind)
        sizes = np.array([self.p[i].nfaces for i in self._mask])
        cumulative = np.zeros(len(sizes) + 1, dtype = int)
        cumulative[1:] = np.cumsum(sizes)

        particle_idx = np.empty(len(ind), dtype = int)
        local_ind = np.empty(len(ind), dtype = int)

        for k, idx in enumerate(ind):
            # Find which particle this index belongs to
            p_idx = np.searchsorted(cumulative[1:], idx, side = 'right')
            particle_idx[k] = p_idx
            local_ind[k] = idx - cumulative[p_idx]

        return particle_idx, local_ind

    def set(self, **kwargs):
        """
        Set properties of the compound particle (on pc).

        MATLAB: @compound/set.m

        Parameters
        ----------
        **kwargs : dict
            Property name-value pairs to set on the concatenated particle
        """
        for key, value in kwargs.items():
            setattr(self.pc, key, value)
        return self

    @property
    def size(self):
        """
        Number of faces for each masked particle.

        MATLAB: compound.size property
        """
        return np.array([self.p[i].nfaces for i in self._mask])

    def __repr__(self):
        return (
            "ComParticle(nparticles = {}, "
            "nverts = {}, nfaces = {})".format(len(self.p), self.nverts, self.nfaces)
        )

    def __str__(self):
        parts_info = "\n".join(
            "  Particle {}: {} verts, {} faces".format(i + 1, p.nverts, p.nfaces)
            for i, p in enumerate(self.p)
        )
        return (
            "Compound Particle:\n"
            "  Materials: {}\n"
            "  Particles: {}\n"
            "{}\n"
            "  Total vertices: {}\n"
            "  Total faces: {}".format(len(self.eps), len(self.p), parts_info, self.nverts, self.nfaces)
        )
