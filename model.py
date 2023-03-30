import numpy as np


class SimpleModel:
    class State:
        def __init__(self, juveniles: float, adults: float):
            self.juveniles = juveniles
            self.adults = adults

    def __init__(
        self,
        fecundity: float,
        juvenile_survive_probability: float,
        adult_survival_probability: float,
        strength_cannibalism: float,
    ) -> None:
        self.fecundity = fecundity
        self.juvenile_survive_probability = juvenile_survive_probability
        self.adult_survival_probability = adult_survival_probability
        self.strength_cannibalism = strength_cannibalism

    def run_one_time_step(self, state: "SimpleModel.State") -> "SimpleModel.State":
        "takes a State object and returns a new State object representing the next time step"
        juveniles, adults = state.juveniles, state.adults

        births = self.fecundity * adults
        cannibalism = self.strength_cannibalism * adults * juveniles
        deaths = adults / self.adult_survival_probability  # ostensibly from old age
        maturations = juveniles * self.juvenile_survive_probability - cannibalism

        new_juveniles = births
        new_adults = adults + maturations - deaths

        return SimpleModel.State(new_juveniles, new_adults)

    def run(
        self, initial: "SimpleModel.State", time_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        "takes a initial condition and produces a tuple of numpy arrays (juveniles, adults)"

        current_state = initial
        juveniles_array = np.array([initial.juveniles])
        adults_array = np.array([initial.adults])

        for _ in range(time_steps - 1):
            current_state = self.run_one_time_step(current_state)
            juveniles_array = np.append(juveniles_array, current_state.juveniles)
            adults_array = np.append(adults_array, current_state.adults)

        return juveniles_array, adults_array
