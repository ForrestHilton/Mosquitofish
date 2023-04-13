import numpy as np
import functions as fn

"""A similar class to SimpleModel except that a competition
term is included. Ricker cannibalism is used."""


class CompetitionModel:

    # A class variable will be used to keep track of 
    # the constant associated with the competition
    K = 1000

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

    def description(self):
        return type(self).__name__

    def run_one_time_step(self, state: "CompetitionModel.State") -> "CompetitionModel.State":
        "takes a State object and returns a new State object representing the next time step"
        juveniles, adults = state.juveniles, state.adults

        births = self.fecundity * adults
        cannibalism_mult = self.cannibalism_multiplier(state)
        maturations = juveniles * self.juvenile_survive_probability * cannibalism_mult
        # 
        competition_deaths = state.adults*state.adults/self.K

        new_juveniles = np.maximum(births, 0)
        new_adults = np.maximum(
            adults * self.adult_survival_probability + maturations - competition_deaths, 0
        )

        return CompetitionModel.State(new_juveniles, new_adults)

    def run(
        self, initial: "CompetitionModel.State", time_steps: int
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

    def cannibalism_multiplier(self, state: State):
        # Ricker competition, as specified in the instructions
        return np.exp(-self.strength_cannibalism*state.adults)
    
# Object for use in main -- need to import functions for
competition_model = CompetitionModel(
    fn.fecundity,
    fn.juvenile_survival_probability,
    fn.adult_survival_probability,
    fn.strength_can_ricker,
)