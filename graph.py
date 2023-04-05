import numpy as np
import matplotlib.pyplot as plt
from model import SimpleModel


def ordinary_plot_over_time(juveniles, adults):
    plt.plot(range(len(juveniles)), list(juveniles), label="Juveniles")
    plt.plot(range(len(adults)), list(adults), label="Adults")
    plt.xlabel("Time step")
    plt.ylabel("Population")
    plt.legend()
    plt.show()

def show_interactive_2d_seedspace(model, juviniles_max, adults_max):
    def run(x: float, y: float, iterations: int) -> tuple[np.ndarray, np.ndarray]:
        return model.run(SimpleModel.State(x, y), iterations)

    # Generate 20 points using the run function
    x, y = run(0, 0, 20)

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
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # Define the function to update the plot based on the new first point
    def update_plot(event):
        if event.inaxes == ax:
            x, y = run(event.xdata, event.ydata, 20)
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
