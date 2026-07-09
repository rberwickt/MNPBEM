import os
import sys
import numpy as np
from typing import Optional, Tuple, Any, List, Callable


class ClusterTree(object):

    # MATLAB: @clustertree
    # Builds cluster tree through recursive bisection.
    # See S. Boerm et al., Eng. Analysis with Bound. Elem. 27, 405 (2003).

    def __init__(self,
            pos: np.ndarray,
            cleaf: int = 32,
            ipart_arr: Optional[np.ndarray] = None):

        # pos: (n, d) array of face/vertex positions
        # cleaf: threshold for bisection leaf size
        # ipart_arr: (n,) array of particle indices per face (0-based),
        #            None if single particle

        self.pos = np.asarray(pos, dtype = np.float64)
        self.n = self.pos.shape[0]

        # ipart_arr: maps each face index to its particle index
        if ipart_arr is None:
            self._ipart_arr = np.zeros(self.n, dtype = np.int64)
        else:
            self._ipart_arr = np.asarray(ipart_arr, dtype = np.int64)

        # Build tree
        self._build(cleaf)

    def _build(self, cleaf: int) -> None:

        n = self.n
        pos = self.pos

        # ind: mapping from cluster-ordered index to original face index
        # Working arrays during construction
        ind = np.arange(n, dtype = np.int64)
        cind = np.arange(n, dtype = np.int64)

        # Bounding box for root
        box_min = pos.min(axis = 0)
        box_max = pos.max(axis = 0)
        root_mid = 0.5 * (box_min + box_max)
        root_rad = 0.5 * np.linalg.norm(box_max - box_min)

        # Tree nodes stored as dicts in a list
        # Each node: {cind_start, cind_end, mid, rad, son1, son2}
        # son1/son2 = -1 means leaf
        tree = []
        tree.append({
            'cind_start': 0,
            'cind_end': n - 1,
            'mid': root_mid,
            'rad': root_rad,
            'son1': -1,
            'son2': -1,
            'leaf_ind': None,
        })

        self._bisection(tree, 0, ind, cind, cleaf)

        # Extract tree structure
        num_nodes = len(tree)

        # son array: (num_nodes, 2), 0 means no son (leaf)
        # Using 0-based indexing internally, but store -1 as "no child"
        # and convert: son[i] = [son1_idx, son2_idx], -1 means leaf
        son = np.full((num_nodes, 2), -1, dtype = np.int64)
        cind_arr = np.empty((num_nodes, 2), dtype = np.int64)
        mid_arr = np.empty((num_nodes, pos.shape[1]), dtype = np.float64)
        rad_arr = np.empty(num_nodes, dtype = np.float64)

        for i in range(num_nodes):
            node = tree[i]
            son[i, 0] = node['son1']
            son[i, 1] = node['son2']
            cind_arr[i, 0] = node['cind_start']
            cind_arr[i, 1] = node['cind_end']
            mid_arr[i] = node['mid']
            rad_arr[i] = node['rad']

        self.son = son
        self.cind = cind_arr
        self.mid = mid_arr
        self.rad = rad_arr

        # Build ind mapping: (n, 2) array
        # ind[:, 0] = cluster_to_part: for cluster index c, ind[c, 0] = particle face index
        # ind[:, 1] = part_to_cluster: for particle face index p, ind[p, 1] = cluster index
        # Collect leaf indices
        leaf_inds = []
        leaf_cinds = []
        for i in range(num_nodes):
            node = tree[i]
            if node['leaf_ind'] is not None:
                leaf_inds.append(node['leaf_ind'])
                cs = node['cind_start']
                ce = node['cind_end']
                leaf_cinds.append(np.arange(cs, ce + 1, dtype = np.int64))

        # face indices in cluster order
        total_leaf_ind = np.empty(n, dtype = np.int64)
        total_leaf_cind = np.empty(n, dtype = np.int64)
        offset = 0
        for li, lc in zip(leaf_inds, leaf_cinds):
            sz = len(li)
            total_leaf_ind[offset:offset + sz] = li
            total_leaf_cind[offset:offset + sz] = lc
            offset += sz

        # Sort by cluster index to create mapping arrays
        sort_by_cind = np.argsort(total_leaf_cind)
        # ind_array[cluster_idx] = face_idx
        cluster_to_part = total_leaf_ind[sort_by_cind]

        sort_by_ind = np.argsort(total_leaf_ind)
        # part_to_cluster[face_idx] gives cluster position
        part_to_cluster = total_leaf_cind[sort_by_ind]

        # Store as (n, 2): col 0 = cluster_to_part, col 1 = part_to_cluster
        # MATLAB: ind(:,1) maps cluster->particle, ind(:,2) maps particle->cluster
        # For MATLAB: v(obj.ind(:,1), :) reorders from cluster to particle order
        # Python: ind[c, 0] = face index for cluster position c
        #         ind[p, 1] = cluster position for face index p
        self.ind = np.empty((n, 2), dtype = np.int64)
        self.ind[:, 0] = cluster_to_part
        self.ind[:, 1] = part_to_cluster

        # Particle index per tree node (0-based, -1 for composite)
        ipart_node = np.empty(num_nodes, dtype = np.int64)
        for i in range(num_nodes):
            cs = cind_arr[i, 0]
            ce = cind_arr[i, 1]
            # Get face indices for start and end of this cluster
            face_start = self.ind[cs, 0]
            face_end = self.ind[ce, 0]
            ip_start = self._ipart_arr[face_start]
            ip_end = self._ipart_arr[face_end]
            if ip_start == ip_end:
                ipart_node[i] = ip_start
            else:
                ipart_node[i] = -1  # composite
        self.ipart = ipart_node

    def _bisection(self,
            tree: List,
            ic: int,
            ind: np.ndarray,
            cind: np.ndarray,
            cleaf: int) -> None:

        siz = len(tree)
        # Add two sons
        tree[ic]['son1'] = siz
        tree[ic]['son2'] = siz + 1

        # Split
        ind1, ind2, cind1, cind2, psplit, son1_info, son2_info = self._split(ind, cind)

        tree.append(son1_info)
        tree.append(son2_info)

        # Further splitting of cluster 1?
        if len(ind1) > cleaf or psplit:
            self._bisection(tree, siz, ind1, cind1, cleaf)
        else:
            tree[siz]['leaf_ind'] = ind1

        # Further splitting of cluster 2?
        if len(ind2) > cleaf or psplit:
            self._bisection(tree, siz + 1, ind2, cind2, cleaf)
        else:
            tree[siz + 1]['leaf_ind'] = ind2

    def _split(self,
            ind: np.ndarray,
            cind: np.ndarray) -> Tuple:

        # Try particle split first
        ind1, ind2, psplit = self._partsplit(ind)

        if not psplit:
            # Bisection split
            ind1, ind2 = self._bisplit(ind)

        n1 = len(ind1)
        cind1 = cind[:n1]
        cind2 = cind[n1:]

        # Build son info
        son1_info = {
            'cind_start': int(cind1[0]),
            'cind_end': int(cind1[-1]),
            'mid': None,
            'rad': None,
            'son1': -1,
            'son2': -1,
            'leaf_ind': None,
        }
        m1, r1 = self._sph_boundary(ind1)
        son1_info['mid'] = m1
        son1_info['rad'] = r1

        son2_info = {
            'cind_start': int(cind2[0]),
            'cind_end': int(cind2[-1]),
            'mid': None,
            'rad': None,
            'son1': -1,
            'son2': -1,
            'leaf_ind': None,
        }
        m2, r2 = self._sph_boundary(ind2)
        son2_info['mid'] = m2
        son2_info['rad'] = r2

        return ind1, ind2, cind1, cind2, psplit, son1_info, son2_info

    def _partsplit(self, ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:

        ip = self._ipart_arr[ind]
        unique_parts = np.unique(ip)

        if len(unique_parts) == 1:
            return np.array([], dtype = np.int64), np.array([], dtype = np.int64), False

        # Split: first particle vs rest
        mask = ip == ip[0]
        ind1 = ind[mask]
        ind2 = ind[~mask]
        return ind1, ind2, True

    def _bisplit(self, ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        pos = self.pos[ind]
        box_min = pos.min(axis = 0)
        box_max = pos.max(axis = 0)

        # Split along longest dimension
        extent = box_max - box_min
        k = np.argmax(extent)

        mid_val = box_min[k] + 0.5 * extent[k]

        mask = self.pos[ind, k] < mid_val
        ind1 = ind[mask]
        ind2 = ind[~mask]

        # Handle edge case: all points on one side
        if len(ind1) == 0:
            ind1 = ind[:1]
            ind2 = ind[1:]
        elif len(ind2) == 0:
            ind1 = ind[:-1]
            ind2 = ind[-1:]

        return ind1, ind2

    def _sph_boundary(self, ind: np.ndarray) -> Tuple[np.ndarray, float]:

        pos = self.pos[ind]
        box_min = pos.min(axis = 0)
        box_max = pos.max(axis = 0)
        mid = 0.5 * (box_min + box_max)
        rad = 0.5 * np.linalg.norm(box_max - box_min)
        return mid, rad

    def matsize(self, other: 'ClusterTree') -> Tuple[int, int]:

        # MATLAB: matsize
        return (self.n, other.n)

    def matindex(self,
            other: 'ClusterTree',
            i1: int,
            i2: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:

        # MATLAB: matindex
        # Returns row_indices, col_indices, sub_matrix_size
        rows = self.ind[self.cind[i1, 0]:self.cind[i1, 1] + 1, 0]
        cols = other.ind[other.cind[i2, 0]:other.cind[i2, 1] + 1, 0]
        siz = (len(rows), len(cols))
        return rows, cols, siz

    def part2cluster(self, v: np.ndarray) -> np.ndarray:

        # MATLAB: part2cluster
        # Reorders v from particle ordering to cluster ordering
        # v[face_idx] -> result[cluster_idx]
        return v[self.ind[:, 0]]

    def cluster2part(self, v: np.ndarray) -> np.ndarray:

        # MATLAB: cluster2part
        # Reorders v from cluster ordering to particle ordering
        # v[cluster_idx] -> result[face_idx]
        return v[self.ind[:, 1]]

    def admissibility(self,
            other: 'ClusterTree',
            fadmiss: Optional[Callable] = None) -> np.ndarray:

        # MATLAB: admissibility
        # Returns sparse-like matrix: 0 = not admissible (recurse), 1 = admissible (low-rank), 2 = leaf (dense)
        # Using a dictionary to store sparse entries

        if fadmiss is None:
            fadmiss = lambda rad1, rad2, dist: 2.5 * min(rad1, rad2) < dist

        n1 = self.son.shape[0]
        n2 = other.son.shape[0]

        # Store as dict of (i1, i2) -> value
        entries = {}
        self._blocktree(entries, other, 0, 0, fadmiss)

        return entries

    def _admissible(self,
            other: 'ClusterTree',
            i1: int,
            i2: int,
            fadmiss: Callable) -> int:

        # MATLAB: admissible (nested in admissibility.m)
        if self.son[i1, 0] == -1 and other.son[i2, 0] == -1:
            # Both are leaves -> dense block
            return 2

        # Check particle indices for the cluster range
        ip1_start = self.ipart[i1]
        ip2_start = other.ipart[i2]

        # If either cluster spans multiple particles, cannot check admissibility -> recurse
        if ip1_start == -1 or ip2_start == -1:
            return 0

        # Check admissibility condition
        dist = np.linalg.norm(self.mid[i1] - other.mid[i2])
        if fadmiss(self.rad[i1], other.rad[i2], dist):
            return 1
        else:
            return 0

    def _index(self, i: int) -> List[int]:

        # MATLAB: index (nested in admissibility.m)
        # Returns children or self for leaves
        if self.son[i, 0] == -1:
            return [i]
        else:
            return [self.son[i, 0], self.son[i, 1]]

    def _blocktree(self,
            entries: dict,
            other: 'ClusterTree',
            i1: int,
            i2: int,
            fadmiss: Callable) -> None:

        # MATLAB: blocktree (nested in admissibility.m)
        ad = self._admissible(other, i1, i2, fadmiss)

        if ad > 0:
            entries[(i1, i2)] = ad
        else:
            ind1 = self._index(i1)
            ind2 = other._index(i2)
            for ii1 in ind1:
                for ii2 in ind2:
                    self._blocktree(entries, other, ii1, ii2, fadmiss)

    def cluster_size(self, i: int) -> int:

        return self.cind[i, 1] - self.cind[i, 0] + 1
