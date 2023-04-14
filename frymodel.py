import numpy as np
import matplotlib.pyplot as plt
import functions

time_step_weeks = 4

# Mosquitofish survive about 2 years on average (104 weeks)
avg_lifespan_weeks = 104
# This can be used to find probability of dying in a given
# time interval:
adult_die_probability = time_step_weeks / avg_lifespan_weeks
# this can then be used to calculate chance of survival
# in given time interval
adult_survival_probability = 1 - adult_die_probability

# in simple model, we assume only death from old age and
# canabilism; we can assume probability of juvenile surviving
# to adulthood IN THE ABSENSE of canibilism should be
# very high (we can assume 1)

juvenile_survival_probability = 1

# Females can give birth to up to 30 at once (assume 15 average)
avg_births_at_once = 15
# Each female can produce three to four broods in her lifetime
# https://www.shelbytnhealth.com/487/Mosquito-Fish-for-Ponds
broods_per_lifetime = 3.5
# Thus, to calculate fecundity (number of ones born per year
# on average) do:
fecundity = (
    avg_births_at_once * broods_per_lifetime * time_step_weeks
) / avg_lifespan_weeks


strength_cannibalism = 1


class FryModel(functions.Ricker):
    class State:
        def __init__(self, fry: float, juveniles: float, adults: float):
            self.fry = fry
            self.juveniles = juveniles
            self.adults = adults

    def run_one_time_step(self, state: "FryModel.State") -> "FryModel.State":
        "takes a State object and returns a new State object representing the next time step"
        # this function has been changed heavily from parent
        fry, juveniles, adults = (
            state.fry,
            state.juveniles,
            state.adults,
        )

        births = self.fecundity * adults
        new_fry = births

        cannibalism_mult = self.cannibalism_multiplier(state)
        new_juveniles = fry * self.juvenile_survive_probability * cannibalism_mult

        maturations = juveniles * self.juvenile_survive_probability
        new_adults = adults * self.adult_survival_probability + maturations

        return FryModel.State(new_fry, new_juveniles, new_adults)

    def run(
        self, initial: "FryModel.State", time_steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        "takes a initial condition and produces a tuple of numpy arrays (fry, juveniles, adults)"
        # this is mostly the same, but with an extra variable

        current_state = initial
        fry_array = np.array([initial.fry])
        juveniles_array = np.array([initial.juveniles])
        adults_array = np.array([initial.adults])

        for _ in range(time_steps - 1):
            current_state = self.run_one_time_step(current_state)
            fry_array = np.append(fry_array, current_state.fry)
            juveniles_array = np.append(juveniles_array, current_state.juveniles)
            adults_array = np.append(adults_array, current_state.adults)

        return fry_array, juveniles_array, adults_array


def ordinary_plot_over_time(model: FryModel, steps: int):
    # Run the model for 20 time steps starting with 10 juveniles and 20 adults
    fry, juveniles, adults = model.run(FryModel.State(100, 100, 100), steps)

    plt.plot(range(steps), list(fry), label="Fry")
    plt.plot(range(steps), list(juveniles), label="Juveniles")
    plt.plot(range(steps), list(adults), label="Adults")
    plt.xlabel("Time step")
    plt.ylabel("Population")
    plt.legend()
    plt.title(model.description())
    plt.show()


def show_interactive_3d_seedspace(
    model: FryModel, fry_max, juviniles_max, adults_max, iterations: int
):
    def run(x: float, y: float, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return model.run(FryModel.State(x, y, z), iterations)

    # Generate 20 points using the run function
    x, y = run(1, 1, 1)

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


strength_can_ricker = 0.01

fry_model = FryModel(
    fecundity,
    juvenile_survival_probability,
    adult_survival_probability,
    strength_can_ricker,
)


ordinary_plot_over_time(fry_model, 40)
# show_interactive_3d_seedspace(fry_model, 100, 400, 800)
