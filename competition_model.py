import numpy as np
from cannibalism_models import Ricker, SimpleModel

"""A similar class to SimpleModel except that a competition
term is included. Ricker cannibalism is used as specified
in the instructions."""


class CompetitionModel(Ricker):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        # The parameter was arbitrarily chosen to
        # be a large number and this choice of
        # constant appears to have a noticeable
        # effect on the population. It is referred
        # to as a "parameter" rather than "strength"
        # because a larger value actually implies
        # less competition.
        self.competition_param = 1000

    def run_one_time_step(self, state: "SimpleModel.State") -> "SimpleModel.State":
        "takes a State object and returns a new State object representing the next time step"
        # This method is very similar to the corresponding
        # method in the parent class (see that class for
        # more details). The only change is incorporating
        # extra adult deaths from competition.
        juveniles, adults = state.juveniles, state.adults

        births = self.fecundity * adults
        cannibalism_mult = self.cannibalism_multiplier(state)
        maturations = juveniles * self.juvenile_survival_probability * cannibalism_mult
        # Here we're calculating the number adults that died
        # from competition, presumably from starving to death due to
        # limited resources and some fish getting acess to food but
        # others not. The number of deaths is proportional to the
        # number of adults squared, inspired by the simple logistic
        # population model: dP/dt = kP(1-P/Pmax). Expanding this
        # expression yields dP/dt = kP - kP^2/Pmax; the births term
        # is represented by kP, and the competition term, which we're
        # interested in, is represented by - kP^2/Pmax. For the sake
        # of simplicity we assume only adults die from competition.
        competition_deaths = state.adults * state.adults / self.competition_param

        new_juveniles = np.maximum(births, 0)
        # Here, when calculating the new number of adults, we have
        # to subtract deaths from competition from the figure we
        # would have obtained using the parent class method.
        new_adults = np.maximum(
            adults * self.adult_survival_probability + maturations - competition_deaths,
            0,
        )

        return CompetitionModel.State(new_juveniles, new_adults)
