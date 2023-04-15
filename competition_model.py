import numpy as np
from cannibalism_models import Ricker, SimpleModel

"""A similar class to SimpleModel except that a competition
term is included. Ricker cannibalism is used."""


class CompetitionModel(Ricker):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.competition_param = 1000

    def run_one_time_step(
        self, state: "SimpleModel.State"
    ) -> "SimpleModel.State":
        "takes a State object and returns a new State object representing the next time step"
        juveniles, adults = state.juveniles, state.adults

        births = self.fecundity * adults
        cannibalism_mult = self.cannibalism_multiplier(state)
        maturations = juveniles * self.juvenile_survival_probability * cannibalism_mult
        competition_deaths = state.adults * state.adults / self.competition_param

        new_juveniles = np.maximum(births, 0)
        new_adults = np.maximum(
            adults * self.adult_survival_probability + maturations - competition_deaths,
            0,
        )

        return CompetitionModel.State(new_juveniles, new_adults)
