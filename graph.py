from typing import List
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleModel
import functions as fn
from matplotlib.widgets import Slider, Button
from frymodel import fry_model, FryModel
import inspect


def ordinary_plot_over_time(model: SimpleModel, steps: int):
    # Run the model for 20 time steps starting with 10 juveniles and 20 adults
    juveniles, adults = model.run(SimpleModel.State(100, 100), steps)

    plt.plot(range(steps), list(juveniles), label="Juveniles")
    plt.plot(range(steps), list(adults), label="Adults")
    plt.xlabel("Time step")
    plt.ylabel("Population")
    plt.legend()
    plt.title(model.description())
    plt.show()


def show_interactive_2d_seedspace(model, juviniles_max, adults_max, iterations: int):
    parameter_sliders = {}

    for i, key in enumerate(model.__dict__):
        val = model.__dict__[key]
        parameter_sliders[key] = Slider(
            plt.axes([0.35, 0.95 - i * 0.05, 0.55, 0.03]), key, 0.0, val * 5, val
        )

    def run(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        return model.run(SimpleModel.State(x, y), iterations)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, "Reset", color="gold", hovercolor="skyblue")
    # Generate 20 points using the run function

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # Initialize the plot
    initial = SimpleModel.State(1, 1)

    def draw_figure():
        ax.clear()
        juviniles_x, audults_y = model.run(initial, iterations)
        ax.plot(juviniles_x, audults_y, "-o")
        ax.quiver(
            juviniles_x[:-1],
            audults_y[:-1],
            juviniles_x[1:] - juviniles_x[:-1],
            audults_y[1:] - audults_y[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
        )
        # Set the limits of the plot
        ax.set_xlim(0, juviniles_max)
        ax.set_ylim(0, adults_max)
        ax.set_title(model.description())
        ax.set_xlabel("Juveniles")
        ax.set_ylabel("Adults")
        fig.canvas.draw()

    # Define the function to update the plot based on the new first point
    def respond_initial_changed(event):
        if event.inaxes == ax and fig.canvas.toolbar.mode == "":
            initial.juveniles = event.xdata
            initial.adults = event.ydata
            draw_figure()

    # Connect the function to the mouse click event
    fig.canvas.mpl_connect("button_press_event", respond_initial_changed)

    draw_figure()

    def respond_param_changed(_):
        for key in model.__dict__:
            model.__dict__[key] = parameter_sliders[key].val
        draw_figure()

    for slider in list(parameter_sliders.values()):
        slider.on_changed(respond_param_changed)

    # Create a function resetSlider to set slider to
    # initial values when Reset button is clicked

    def resetSlider(event):
        for slider in list(parameter_sliders.values()):
            slider.reset()

    # Call resetSlider function when clicked on reset button
    button.on_clicked(resetSlider)
    plt.show()


def show_interactive_3d_seedspace(
    model: FryModel, fry_max, juviniles_max, adults_max, iterations: int
):
    parameter_sliders = {}

    for i, key in enumerate(model.__dict__):
        val = model.__dict__[key]
        parameter_sliders[key] = Slider(
            plt.axes([0.35, 0.95 - i * 0.05, 0.55, 0.03]), key, 0.0, val * 5, val
        )

    fry_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), "Fry", 0.0, fry_max, 1)
    juvinililes_slider = Slider(
        plt.axes([0.25, 0.15, 0.65, 0.03]), "Juviniles", 0.0, juviniles_max, 1
    )
    adults_slider = Slider(
        plt.axes([0.25, 0.1, 0.65, 0.03]), "Adults", 0.0, adults_max, 1
    )

    # Create axes for reset button and create button
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, "Reset", color="gold", hovercolor="skyblue")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Create function to be called when slider value is changed
    # Set the limits of the plot

    def update(val):
        ax.clear()
        initial = FryModel.State(
            fry_slider.val, juvinililes_slider.val, adults_slider.val
        )
        for key in model.__dict__:
            model.__dict__[key] = parameter_sliders[key].val

        fry_x, juviniles_y, adults_z = model.run(initial, iterations)
        ax.scatter(fry_x, juviniles_y, adults_z, c="b", marker="o")

        # Connect consecutive points with lines
        # Add arrows to indicate direction
        for i in range(len(fry_x) - 1):
            dx = fry_x[i + 1] - fry_x[i]
            dy = juviniles_y[i + 1] - juviniles_y[i]
            dz = adults_z[i + 1] - adults_z[i]
            ax.quiver(fry_x[i], juviniles_y[i], adults_z[i], dx, dy, dz, color="black")

        ax.set_xlim(0, fry_max)
        ax.set_ylim(0, juviniles_max)
        ax.set_zlim(0, adults_max)
        ax.set_title(model.description())

        ax.set_xlabel("Fry")
        ax.set_ylabel("Juveniles")
        ax.set_zlabel("Adults")
        fig.canvas.draw()

    update(None)

    for slider in list(parameter_sliders.values()) + [
        fry_slider,
        juvinililes_slider,
        adults_slider,
    ]:
        slider.on_changed(update)

    # Create a function resetSlider to set slider to
    # initial values when Reset button is clicked

    def resetSlider(event):
        for slider in list(parameter_sliders.values()) + [
            fry_slider,
            juvinililes_slider,
            adults_slider,
        ]:
            slider.reset()

    # Call resetSlider function when clicked on reset button
    button.on_clicked(resetSlider)
    plt.show()


def sensitivity_run(
    models: List[SimpleModel], initial=SimpleModel.State(20, 20), iterations=40
):
    list_adults = []
    for model in models:
        _, adults = model.run(initial, iterations)
        plt.plot(
            range(iterations),
            list(adults),
            label=model.description() + " Adult Population",
        )
        list_adults.append(adults)

    plt.xlabel("Time step")
    plt.ylabel("Population")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # show_interactive_3d_seedspace(fry_model, 800, 100, 400, 20)
    show_interactive_2d_seedspace(fn.linear_model, 1000, 500, 20)
