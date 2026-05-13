import os
import sys
import time

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import gmres, cgs, bicgstab, LinearOperator


class BEMIter(object):

    # MATLAB: @bemiter properties
    SOLVER_MAP = {
        'gmres': gmres,
        'cgs': cgs,
        'bicgstab': bicgstab,
    }

    def __init__(self,
            solver: str = 'gmres',
            tol: float = 1e-4,
            maxit: int = 200,
            restart: Optional[int] = None,
            precond: Optional[str] = 'hmat',
            output: int = 0,
            **kwargs: Any) -> None:

        # MATLAB: @bemiter properties
        self.solver = solver
        self.tol = tol
        self.maxit = maxit
        self.restart = restart
        self.precond = precond
        self.output = output

        # MATLAB: @bemiter properties (Access = protected)
        self._flag = None
        self._relres = None
        self._iter = None
        self._enei_sav = []  # previously computed wavelengths
        self._stat = None
        self._timer = None

    def _iter_solve(self,
            x0: Optional[np.ndarray],
            b: np.ndarray,
            afun: Callable,
            mfun: Optional[Callable]) -> Tuple[np.ndarray, 'BEMIter']:

        # MATLAB: bemiter/solve.m
        # use only preconditioner?
        if self.maxit == 0 or self.solver is None:
            assert self.precond == 'hmat', '[error] preconditioner must be hmat when maxit=0'
            x = mfun(b)
            return x, self

        n = b.shape[0]

        # LinearOperator for afun
        a_op = LinearOperator((n, n), matvec = afun, dtype = b.dtype)

        # LinearOperator for preconditioner (mfun)
        m_op = None
        if mfun is not None:
            m_op = LinearOperator((n, n), matvec = mfun, dtype = b.dtype)

        # iterative solution through scipy
        if self.solver == 'cgs':
            x, flag = cgs(a_op, b, x0 = x0, rtol = self.tol, maxiter = self.maxit, M = m_op)
            relres = np.linalg.norm(a_op @ x - b) / np.linalg.norm(b) if np.linalg.norm(b) > 0 else 0.0
            iter_info = np.array([self.maxit, 0])

        elif self.solver == 'bicgstab':
            x, flag = bicgstab(a_op, b, x0 = x0, rtol = self.tol, maxiter = self.maxit, M = m_op)
            relres = np.linalg.norm(a_op @ x - b) / np.linalg.norm(b) if np.linalg.norm(b) > 0 else 0.0
            iter_info = np.array([self.maxit, 0])

        elif self.solver == 'gmres':
            # MATLAB MNPBEM parity: 'restart', [] -> no restart (full Krylov up to maxit).
            # scipy gmres requires int, so use maxit (full subspace).
            # Previous fallback min(n, 20) forced GMRES(20), causing slow / failed
            # convergence on ill-conditioned matrices (touching dimer, charge-transfer
            # plasmon). Setting restart=maxit eliminates restart cycles entirely.
            restart = self.restart if self.restart is not None else min(n, self.maxit)
            x, flag = gmres(a_op, b, x0 = x0, rtol = self.tol, maxiter = self.maxit,
                restart = restart, M = m_op)
            relres = np.linalg.norm(a_op @ x - b) / np.linalg.norm(b) if np.linalg.norm(b) > 0 else 0.0
            iter_info = np.array([self.maxit, 0])

        else:
            raise ValueError('[error] iterative solver not known: <{}>'.format(self.solver))

        # save statistics
        self._set_iter(flag, relres, iter_info)

        # print statistics
        if self.output:
            self._print_stat(flag, relres, iter_info)

        return x, self

    def _set_iter(self,
            flag: int,
            relres: float,
            iter_info: np.ndarray) -> None:

        # MATLAB: bemiter/setiter.m
        if self._flag is None:
            self._flag = []
            self._relres = []
            self._iter = []

        self._flag.append(flag)
        self._relres.append(relres)
        self._iter.append(iter_info.copy())

    def _set_stat(self,
            name: str,
            hmat: Any) -> None:

        # MATLAB: bemiter/setstat.m
        if self._stat is None:
            self._stat = {'compression': {}}

        stat = self._stat

        # compression for H-matrix
        if name not in stat['compression']:
            stat['compression'][name] = []
        if hasattr(hmat, 'compression'):
            stat['compression'][name].append(hmat.compression())

    def tocout(self,
            key: str,
            *args: Any) -> 'BEMIter':

        # MATLAB: bemiter/tocout.m
        # Intermediate timing/progress output for iterative BEM solvers.
        if not self.output or self.precond is None:
            return self

        timer = self._timer

        if key == 'init':
            # Initialize timer structure
            names = args[0] if args else []
            if timer is None:
                timer = {'names': list(names), 'toc': []}
            timer['toc'].append([0.0] * len(timer['names']))
            timer['id'] = time.perf_counter()

        elif key == 'close':
            # Save total elapsed time for last step
            elapsed = time.perf_counter() - timer['id']
            timer['toc'][-1][-1] = elapsed
            # Print final timing summary
            row = timer['toc'][-1]
            parts = ['  {}: {:.4f}s'.format(n, t) for n, t in zip(timer['names'], row)]
            print('BEM timing:\n' + '\n'.join(parts))

        else:
            # Save elapsed time for named step and restart timer
            elapsed = time.perf_counter() - timer['id']
            if key in timer['names']:
                idx = timer['names'].index(key)
                timer['toc'][-1][idx] = elapsed
            timer['id'] = time.perf_counter()

        self._timer = timer
        return self

    def _print_stat(self,
            flag: int,
            relres: float,
            iter_info: np.ndarray) -> None:

        # MATLAB: bemiter/printstat.m
        if self.solver == 'cgs':
            print('cgs({}), it={:3d}, res={:10.4g}, flag={}'.format(
                self.maxit, int(iter_info[0]), relres, flag))

        elif self.solver == 'bicgstab':
            print('bicgstab({}), it={:5.1f}, res={:10.4g}, flag={}'.format(
                self.maxit, iter_info[0], relres, flag))

        elif self.solver == 'gmres':
            print('gmres({}), it={:3d}({}), res={:10.4g}, flag={}'.format(
                self.maxit, int(iter_info[1]), int(iter_info[0]), relres, flag))

    def info(self) -> Tuple[List[int], List[float], List[np.ndarray]]:

        # MATLAB: bemiter/info.m
        return self._flag, self._relres, self._iter

    def hinfo(self) -> None:

        # MATLAB: bemiter/hinfo.m
        if self._stat is None:
            return

        stat = self._stat
        if 'compression' not in stat:
            return

        eta1 = []
        eta2 = []
        for name, vals in stat['compression'].items():
            if name in ('G', 'F', 'G1', 'H1', 'G2', 'H2'):
                eta1.extend(vals)
            else:
                eta2.extend(vals)

        if eta1:
            print('\nCompression Green functions        :  {:8.6f}'.format(np.mean(eta1)))
        if eta2:
            print('Compression auxiliary matrices     :  {:8.6f}\n'.format(np.mean(eta2)))

    @staticmethod
    def options(**kwargs: Any) -> Dict[str, Any]:

        # MATLAB: bemiter.options
        op = {
            'solver': 'gmres',
            'tol': 1e-6,
            'maxit': 100,
            'restart': None,
            'precond': 'hmat',
            'output': 0,
            'cleaf': 200,
            'htol': 1e-6,
            'kmax': [4, 100],
            'fadmiss': lambda rad1, rad2, dist: 2.5 * min(rad1, rad2) < dist,
        }

        for key, val in kwargs.items():
            if key in op:
                op[key] = val

        return op

    def __repr__(self) -> str:
        return 'BEMIter(solver={}, tol={}, maxit={}, precond={})'.format(
            self.solver, self.tol, self.maxit, self.precond)
