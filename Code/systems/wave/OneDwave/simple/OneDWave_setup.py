from coffee import ibvp, solvers, grid
from coffee.actions import gp_plotter

from OneDWave import OneDwave

system = OneDwave()
solver = solvers.RungeKutta4(system)
grid = grid.UniformCart((200,), [(0, 4)])

plotter = gp_plotter.Plotter1D(
    system,
    'set terminal qt',
    'set yrange [-1:1]',
    'set style data lines',
)

problem = ibvp.IBVP(solver, system, grid=grid, action=[plotter])
problem.run(0, 3)
