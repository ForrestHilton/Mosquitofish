"""
This file builds upon https://www.cantorsparadise.com/the-julia-set-e03c29bed3d0 ,
in order to provide a visualization of the seed space for a given model.
The black pixels are the  that stay bounded or 
"""
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleModel
from functions import model


def seed_space(
    model: SimpleModel, iterations=50, pixels_wide=1000, bounds=np.array([-0, 2, -0, 2])
):
    # Limits of the grid.
    x0 = bounds[0]
    x1 = bounds[1]
    y0 = bounds[2]
    y1 = bounds[3]
    x, y = np.meshgrid(
        np.linspace(x0, x1, pixels_wide), np.linspace(y0, y1, pixels_wide) * 1j
    )
    state = SimpleModel.State(juveniles=x, adults=y)
    # F keeps track of which grid points are bounded
    # even after many iterations
    F = np.zeros([pixels_wide, pixels_wide])
    # Iterate through the operation
    for j in range(iterations):
        state = model.run_one_time_step(state)
        index = state.juveniles + state.adults < np.inf
        F[index] = F[index] + 1  # count how many iterations it takes to get to inf
    return np.linspace(x0, x1, pixels_wide), np.linspace(y0, y1, pixels_wide), F


def graph_seed_space(model: SimpleModel, juviniles_max, adults_max):
    x, y, F = seed_space(
        model,
        iterations=20,
        pixels_wide=1000,
        bounds=np.array([0, juviniles_max, 0, adults_max]),
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.pcolormesh(x, y, F, cmap="binary")
    ax.set_xlabel('Juveniles')
    ax.set_ylabel('Adults')
    # ax.set_title('Number of iterations without reaching infinity for initial value')
    fig.colorbar(im, ax=ax)
    plt.show()


model.strength_cannibalism = 5
# model = SimpleModel(0.005, 1, 1, 1)
# graph_seed_space(model, 100, 100)
# model = SimpleModel(1, 1, 1, 1)
graph_seed_space(model, 100, 100)
