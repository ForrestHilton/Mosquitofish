from typing import List
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleModel
import functions as fn


def ordinary_plot_over_time(model: SimpleModel, steps: int):
    # Run the model for 20 time steps starting with 10 juveniles and 20 adults
    juveniles, adults = model.run(SimpleModel.State(100, 100), 20)

    plt.plot(range(len(juveniles)), list(juveniles), label="Juveniles")
    plt.plot(range(len(adults)), list(adults), label="Adults")
    plt.xlabel("Time step")
    plt.ylabel("Population")
    plt.legend()
    plt.title(model.description())
    plt.show()


def show_interactive_2d_seedspace(model, juviniles_max, adults_max, iterations: int):
    def run(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        return model.run(SimpleModel.State(x, y), iterations)

    # Generate 20 points using the run function
    x, y = run(1, 1)

    # Initialize the plot
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, y, "-o")
    arrow = ax.quiver(
        x[:-1],
        y[:-1],
        x[1:] - x[:-1],
        y[1:] - y[:-1],
        scale_units="xy",
        angles="xy",
        scale=1,
    )
    # Set the limits of the plot
    ax.set_xlim(0, juviniles_max)
    ax.set_ylim(0, adults_max)
    ax.set_title(model.description())

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # Define the function to update the plot based on the new first point
    def update_plot(event):
        if event.inaxes == ax:
            x, y = run(event.xdata, event.ydata)
            line.set_data(x, y)
            arrow.set_offsets(np.c_[x[:-1], y[:-1]])
            arrow.set_UVC(x[1:] - x[:-1], y[1:] - y[:-1])
            fig.canvas.draw()

    # Connect the function to the mouse click event
    cid = fig.canvas.mpl_connect("button_press_event", update_plot)

    # Set the title and axis labels
    # ax.set_title('')
    ax.set_xlabel("Juveniles")
    ax.set_ylabel("Adults")

    plt.show()


def sensitivity_run(
    models: List[SimpleModel], initial=SimpleModel.State(20, 20), iterations=40
):
    list_adults = []
    for model in models:
        _, adults = model.run(initial, iterations)
        plt.plot(range(iterations), list(adults), label=model.description() + " Adult Population")
        list_adults.append(adults)

    plt.xlabel("Time step")
    plt.ylabel("Population")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    show_interactive_2d_seedspace(fn.linear_model, 1000, 500, 100)
