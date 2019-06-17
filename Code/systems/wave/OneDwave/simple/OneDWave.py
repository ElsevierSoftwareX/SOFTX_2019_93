import numpy as np

from coffee.tslices import tslices
from coffee.system import System
from coffee.diffop.sbp import sbp

class OneDwave(System):

    def __init__(self):
        self.D = sbp.D43_2_CNG()

    def timestep(self, u):
        return 0.4 * u.domain.step_sizes[0]

    def initial_data(self, t0, grid):
        axis = grid.axes[0]
        rv = 0.5 * np.exp(-10 * (axis - axis[int(axis.shape[0] / 2)])**2)
        return tslices.TimeSlice([rv, np.zeros_like(rv)], grid, t0)

    def evaluate(self, t, Psi):
        f0, Dtf0 = Psi.data
        DxDxf = np.real(self.D(f0, Psi.domain.step_sizes[0]))
        DtDtf = DxDxf
        DtDtf[-1] = 0.0
        DtDtf[0] =  0.0
        return tslices.TimeSlice([Dtf0, DtDtf], Psi.domain, time=t)
