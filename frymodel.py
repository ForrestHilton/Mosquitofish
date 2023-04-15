import numpy as np
import matplotlib.pyplot as plt
import functions
import math


class FryModel(functions.Ricker):
    def __init__(self):
        super().__init__()
        # adjustments for halving time step
        # please refer to model.py for further documentation

        time_step_weeks = 4
        avg_lifespan_weeks = 104
        adult_die_probability = time_step_weeks / avg_lifespan_weeks
        avg_births_at_once = 15
        broods_per_lifetime = 3.5
        self.iterations = math.floor(avg_lifespan_weeks / time_step_weeks * 2)
        self.fecundity = (
            avg_births_at_once * broods_per_lifetime * time_step_weeks
        ) / avg_lifespan_weeks
        self.adult_survival_probability = 1 - adult_die_probability

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
        new_juveniles = fry * self.juvenile_survival_probability * cannibalism_mult

        maturations = juveniles * self.juvenile_survival_probability
        new_adults = adults * self.adult_survival_probability + maturations

        return FryModel.State(new_fry, new_juveniles, new_adults)

    def run(
        self, initial: "FryModel.State"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        "takes a initial condition and produces a tuple of numpy arrays (fry, juveniles, adults)"
        # this is mostly the same, but with an extra variable
        self.iterations = math.floor(self.iterations)

        current_state = initial
        fry_array = np.array([initial.fry])
        juveniles_array = np.array([initial.juveniles])
        adults_array = np.array([initial.adults])

        for _ in range(self.iterations - 1):
            current_state = self.run_one_time_step(current_state)
            fry_array = np.append(fry_array, current_state.fry)
            juveniles_array = np.append(juveniles_array, current_state.juveniles)
            adults_array = np.append(adults_array, current_state.adults)

        return fry_array, juveniles_array, adults_array


def ordinary3d_plot_over_time(model: FryModel):
    # Run the model for 20 time steps starting with 10 juveniles and 20 adults
    fry, juveniles, adults = model.run(FryModel.State(100, 100, 100))

    plt.plot(range(model.iterations), list(fry), label="Fry")
    plt.plot(range(model.iterations), list(juveniles), label="Juveniles")
    plt.plot(range(model.iterations), list(adults), label="Adults")
    plt.xlabel("Time step")
    plt.ylabel("Population")
    plt.legend()
    plt.title(model.description())
    plt.show()


fry_model = FryModel()
