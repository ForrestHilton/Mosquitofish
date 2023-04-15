import numpy as np
import functions as fn

"""A similar class to SimpleModel except that a competition
term is included. Ricker cannibalism is used."""


class CompetitionModel(fn.Ricker):
    def __init__(
        self,
        cm_competion_param=1000,
    ) -> None:
        super().__init__()
        self.competition_param = cm_competion_param

    def run_one_time_step(
        self, state: "fn.SimpleModel.State"
    ) -> "fn.SimpleModel.State":
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


# Object for use in main -- need to import functions for
competition_model = CompetitionModel()
