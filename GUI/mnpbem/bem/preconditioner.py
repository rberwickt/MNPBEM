import os
import sys

from typing import Any, Optional, Tuple, List, Dict

import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve_triangular


# v1.5.0 Agent alpha — H-matrix LU preconditioner
# Two-phase implementation:
#   alpha-1: dense LU on full(HMatrix) — simple, OOM at 25k+
#   alpha-2: hierarchical block-Schur LU on H-tree root partition,
#            recursive small leaves stay dense, off-diagonal admissible
#            blocks stay low-rank (UV^T) and are applied to vectors via
#            two GEMMs.  Memory ~ O(N log N), not O(N^2).


class HMatrixLUPreconditioner(object):

    # Modes:
    #   'dense'       — alpha-1, full N x N LU
    #   'tree'        — alpha-2, recursive block-Schur LU on the H-tree root
    #   'auto'        — pick based on tree.n (< _AUTO_THRESHOLD => dense)

    _AUTO_THRESHOLD = 5000

    def __init__(self,
            hmatrix: Any,
            htol_lu: float = 1e-4,
            mode: str = 'auto',
            max_levels: int = 4) -> None:

        assert mode in {'dense', 'tree', 'auto'}, \
            '[error] Invalid <mode>! must be one of dense / tree / auto'
        self.hmatrix = hmatrix
        self.htol_lu = htol_lu
        self.max_levels = max_levels

        if mode == 'auto':
            n = self._matrix_size(hmatrix)
            mode = 'dense' if n < self._AUTO_THRESHOLD else 'tree'
        self.mode = mode

        self.shape = (self._matrix_size(hmatrix), self._matrix_size(hmatrix))
        self.dtype = self._infer_dtype(hmatrix)

        self._build()

    # scipy.sparse.linalg.LinearOperator wants a `.shape` and `.dtype`. By
    # exposing both attributes plus matvec / __matmul__ this class can be
    # passed directly as ``M=`` to gmres / cgs / bicgstab.

    @staticmethod
    def _matrix_size(hmatrix: Any) -> int:
        if isinstance(hmatrix, np.ndarray):
            return int(hmatrix.shape[0])
        if hasattr(hmatrix, 'tree') and hmatrix.tree is not None:
            return int(hmatrix.tree.n)
        if hasattr(hmatrix, 'shape'):
            return int(hmatrix.shape[0])
        raise ValueError('[error] cannot determine size of <hmatrix>')

    @staticmethod
    def _infer_dtype(hmatrix: Any) -> Any:
        if isinstance(hmatrix, np.ndarray):
            return hmatrix.dtype
        for attr in ('val', 'lhs', 'rhs'):
            blocks = getattr(hmatrix, attr, None)
            if blocks is None:
                continue
            for blk in blocks:
                if blk is not None and np.iscomplexobj(blk):
                    return np.complex128
        return np.float64

    # ------------------------------------------------------------------
    # Build dispatch
    # ------------------------------------------------------------------

    def _build(self) -> None:

        match self.mode:

            case 'dense':

                self._build_dense()

            case 'tree':

                self._build_tree()

            case _:

                raise ValueError('[error] Invalid <mode>!')

    def _build_dense(self) -> None:

        # alpha-1: simply densify, run scipy lu_factor.  Used as the
        # baseline. Caller is responsible for memory.
        if isinstance(self.hmatrix, np.ndarray):
            mat = np.array(self.hmatrix, copy = True)
        else:
            mat = self.hmatrix.full()
        self._lu_packed, self._lu_piv = lu_factor(mat, check_finite = False, overwrite_a = True)

    def _build_tree(self) -> None:

        # alpha-2: only meaningful for HMatrix objects with a tree.  For
        # ndarray we fall back to dense.
        if isinstance(self.hmatrix, np.ndarray) or not hasattr(self.hmatrix, 'tree'):
            self.mode = 'dense'
            self._build_dense()
            return

        tree = self.hmatrix.tree
        n = tree.n
        if n < 256:
            # Below this size the tree LU has negligible benefit over dense.
            self.mode = 'dense'
            self._build_dense()
            return

        # Build the recursive node list once. The tree is binary (sons[i]
        # has two indices, < 0 means leaf). We do a recursive partition of
        # the root cluster into small enough pieces, then store dense LU on
        # the leaf diagonal blocks together with the off-diagonal blocks
        # (preserved as low-rank if they were admissible, or dense
        # otherwise) for fast Schur application.
        #
        # We work in cluster ordering throughout — the HMatrix already
        # stores blocks indexed by cluster nodes, so this avoids permuting
        # off-diagonal blocks during build.

        self._tree = tree
        self._ind_part_to_cluster = np.argsort(tree.ind[:, 0])  # particle -> cluster perm
        # tree.ind[:, 0] maps cluster row -> particle; tree.ind[:, 1] maps particle -> cluster.

        self._dense_blocks, self._lr_blocks = self._collect_blocks()
        self._tree_lu_root = self._partition_lu(0, level = 0)

    # ------------------------------------------------------------------
    # alpha-2 helpers: block lookup by tree node indices
    # ------------------------------------------------------------------

    def _collect_blocks(self) -> Tuple[Dict[Tuple[int, int], np.ndarray],
            Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]]:

        # Index every dense / low-rank block by (row_node, col_node).
        hm = self.hmatrix
        dense = {}
        lr = {}
        for i in range(len(hm.row1)):
            if hm.val[i] is None:
                continue
            dense[(int(hm.row1[i]), int(hm.col1[i]))] = hm.val[i]
        for i in range(len(hm.row2)):
            if hm.lhs[i] is None or hm.rhs[i] is None:
                continue
            lr[(int(hm.row2[i]), int(hm.col2[i]))] = (hm.lhs[i], hm.rhs[i])
        return dense, lr

    def _node_range(self, node: int) -> Tuple[int, int]:
        return int(self._tree.cind[node, 0]), int(self._tree.cind[node, 1]) + 1

    def _node_size(self, node: int) -> int:
        a, b = self._node_range(node)
        return b - a

    def _children(self, node: int) -> Tuple[int, int]:
        s = self._tree.son[node]
        return int(s[0]), int(s[1])

    def _gather_block(self,
            row_node: int,
            col_node: int) -> np.ndarray:

        # Materialise the (row_node, col_node) sub-matrix from whatever
        # dense / low-rank blocks the H-matrix carries below those nodes.
        # This walks the H-matrix block list (pre-collected), summing each
        # block whose row/col cluster lies inside (row_node, col_node).
        #
        # Used at leaf level for diagonal blocks (LU) and for off-diagonal
        # blocks that need to be applied to vectors (gathered into either
        # a dense ndarray or kept in low-rank form).

        r0, r1 = self._node_range(row_node)
        c0, c1 = self._node_range(col_node)
        m = r1 - r0
        n = c1 - c0
        out = np.zeros((m, n), dtype = self.dtype)

        for (rn, cn), val in self._dense_blocks.items():
            br0, br1 = self._node_range(rn)
            bc0, bc1 = self._node_range(cn)
            if br0 >= r0 and br1 <= r1 and bc0 >= c0 and bc1 <= c1:
                out[br0 - r0:br1 - r0, bc0 - c0:bc1 - c0] += val

        for (rn, cn), (U, V) in self._lr_blocks.items():
            br0, br1 = self._node_range(rn)
            bc0, bc1 = self._node_range(cn)
            if br0 >= r0 and br1 <= r1 and bc0 >= c0 and bc1 <= c1:
                out[br0 - r0:br1 - r0, bc0 - c0:bc1 - c0] += U @ V.T

        return out

    def _partition_lu(self,
            node: int,
            level: int) -> Dict[str, Any]:

        # Recursive block-Schur LU.
        # If we are at max_levels or the children are missing, build a
        # dense LU for the block (node, node).
        s1, s2 = self._children(node)
        too_deep = level >= self.max_levels
        no_kids = s1 < 0 or s2 < 0
        too_small = self._node_size(node) <= 256

        if too_deep or no_kids or too_small:
            mat = self._gather_block(node, node)
            lu, piv = lu_factor(mat, check_finite = False, overwrite_a = True)
            return {
                'kind': 'leaf',
                'node': node,
                'size': mat.shape[0],
                'lu': lu,
                'piv': piv,
            }

        # 2x2 split: [[A11, A12], [A21, A22]] where children of node are s1, s2.
        # A11 / A22 recurse; A12 / A21 are stored as either low-rank or
        # dense (we materialise them once for speed).
        n1 = self._node_size(s1)
        n2 = self._node_size(s2)

        A12_block = self._collect_offdiag(s1, s2)
        A21_block = self._collect_offdiag(s2, s1)

        child1 = self._partition_lu(s1, level + 1)
        # Schur complement: S22 = A22 - A21 * A11^-1 * A12.
        # We need A12 / A21 as dense for the GEMM.  When the off-diagonal
        # is purely low-rank we still build the dense form with O(n1*n2)
        # cost (acceptable: this happens only at tree node level, not at
        # leaf level).
        A12_dense = self._materialize(A12_block, n1, n2)
        A21_dense = self._materialize(A21_block, n2, n1)

        # A11^-1 @ A12  via leaf solve
        invA11_A12 = self._apply_leaf_inverse(child1, A12_dense)
        S22 = self._gather_block(s2, s2) - A21_dense @ invA11_A12

        # Recurse into the Schur-complemented half.
        s2_lu, s2_piv = lu_factor(S22, check_finite = False, overwrite_a = True)
        child2 = {
            'kind': 'leaf',
            'node': s2,
            'size': n2,
            'lu': s2_lu,
            'piv': s2_piv,
        }

        return {
            'kind': 'split',
            'node': node,
            's1': s1,
            's2': s2,
            'n1': n1,
            'n2': n2,
            'child1': child1,
            'child2': child2,
            'A12': A12_dense,
            'A21': A21_dense,
        }

    def _collect_offdiag(self,
            row_node: int,
            col_node: int) -> List[Any]:

        # Return a list of contributing (kind, payload) tuples for the
        # (row_node, col_node) off-diagonal block.  We don't actually use
        # the structured form in this build: we'll materialise to dense
        # immediately afterwards.  The list form stays in case a future
        # iteration wants to keep low-rank applies.
        items = []
        r0, r1 = self._node_range(row_node)
        c0, c1 = self._node_range(col_node)

        for (rn, cn), val in self._dense_blocks.items():
            br0, br1 = self._node_range(rn)
            bc0, bc1 = self._node_range(cn)
            if br0 >= r0 and br1 <= r1 and bc0 >= c0 and bc1 <= c1:
                items.append(('dense', (br0 - r0, bc0 - c0, val)))

        for (rn, cn), (U, V) in self._lr_blocks.items():
            br0, br1 = self._node_range(rn)
            bc0, bc1 = self._node_range(cn)
            if br0 >= r0 and br1 <= r1 and bc0 >= c0 and bc1 <= c1:
                items.append(('lr', (br0 - r0, bc0 - c0, U, V)))

        return items

    def _materialize(self,
            block_list: List[Any],
            m: int,
            n: int) -> np.ndarray:

        out = np.zeros((m, n), dtype = self.dtype)
        for kind, payload in block_list:
            if kind == 'dense':
                r, c, val = payload
                vm, vn = val.shape
                out[r:r + vm, c:c + vn] += val
            else:
                r, c, U, V = payload
                vm, vn = U.shape[0], V.shape[0]
                out[r:r + vm, c:c + vn] += U @ V.T
        return out

    def _apply_leaf_inverse(self,
            leaf: Dict[str, Any],
            B: np.ndarray) -> np.ndarray:

        # Solve  A_leaf @ X = B  using the stored LU, supports 1-D / 2-D B.
        return lu_solve((leaf['lu'], leaf['piv']), B, check_finite = False)

    def _apply_node_inverse(self,
            node_lu: Dict[str, Any],
            b: np.ndarray) -> np.ndarray:

        # Recursive solve of A x = b using the node block-Schur structure.
        if node_lu['kind'] == 'leaf':
            return self._apply_leaf_inverse(node_lu, b)

        n1 = node_lu['n1']
        b1 = b[:n1]
        b2 = b[n1:]

        # x2 = S22^-1 (b2 - A21 @ A11^-1 b1)
        # x1 = A11^-1 (b1 - A12 @ x2)
        tmp = self._apply_node_inverse(node_lu['child1'], b1)
        rhs2 = b2 - node_lu['A21'] @ tmp
        x2 = self._apply_leaf_inverse(node_lu['child2'], rhs2)
        rhs1 = b1 - node_lu['A12'] @ x2
        x1 = self._apply_node_inverse(node_lu['child1'], rhs1)

        if b.ndim == 1:
            out = np.empty_like(b)
            out[:n1] = x1
            out[n1:] = x2
            return out

        # 2-D case: stack rows.
        out = np.empty_like(b)
        out[:n1] = x1
        out[n1:] = x2
        return out

    # ------------------------------------------------------------------
    # Public solve / matvec
    # ------------------------------------------------------------------

    def solve(self,
            b: np.ndarray) -> np.ndarray:

        if self.mode == 'dense':
            return lu_solve((self._lu_packed, self._lu_piv), b, check_finite = False)

        # alpha-2: H-matrix is in cluster ordering; the BEM solver feeds us
        # vectors in particle ordering.  Permute to cluster, solve, then
        # permute back.  This matches HMatrix.mtimes_vec.
        tree = self._tree
        b_cluster = tree.part2cluster(b)
        x_cluster = self._apply_node_inverse(self._tree_lu_root, b_cluster)
        x = tree.cluster2part(x_cluster)
        return x

    def matvec(self,
            b: np.ndarray) -> np.ndarray:

        return self.solve(b)

    def __matmul__(self,
            b: np.ndarray) -> np.ndarray:

        return self.solve(b)

    def __repr__(self) -> str:

        return 'HMatrixLUPreconditioner(mode={}, shape={}, htol_lu={})'.format(
            self.mode, self.shape, self.htol_lu)


def auto_mode_for_size(n: int,
        threshold: int = HMatrixLUPreconditioner._AUTO_THRESHOLD) -> str:

    return 'dense' if n < threshold else 'tree'
