import numpy as np
from cannibalism_models import Ricker
import math


class FryModel(Ricker):
    def __init__(self):
        # adjustments for halving time step
        super().__init__(time_step_weeks=4)

    class State:

        """The State inner class, like the State inner class
        for the parent class, holds information about what
        would be the stocks in the Stella model. We have
        an additional 'stock' for the fry."""

        def __init__(self, fry: float, juveniles: float, adults: float):
            self.fry = fry
            self.juveniles = juveniles
            self.adults = adults

    def run_one_time_step(self, state: "FryModel.State") -> "FryModel.State":
        "takes a State object and returns a new State object representing the next time step"
        # this function has been changed heavily from parent

        # These variables represent the number of old fry,
        # juvelines, and adults in the last time step.
        fry, juveniles, adults = (
            state.fry,
            state.juveniles,
            state.adults,
        )

        # Births is like a flow (see parent class for more details).
        # The difference here is that we are no longer giving birth
        # to "juveniles", but now giving births to fry, and keeping
        # track of the new fry that will be born in a variable.
        births = self.fecundity * adults
        new_fry = births

        # Here we're calculating the number of fry which will
        # mature to juveniles; they do not directly mature to adults.
        # The fry have a certain probability of surviving on their own
        # which is the same as that for the older juveniles, and
        # they are also susceptible to cannibalism. The cannibalsim
        # multiplier reprsents the probability of suriviving from
        # cannibalism of the adults in this time step.
        cannibalism_mult = self.cannibalism_multiplier(state)
        new_juveniles = fry * self.juvenile_survival_probability * cannibalism_mult

        # Here, we're calculating the number of juveniles that survive
        # to adulthood and the number of new adults. Juveniles have a
        # certain survival probability on their own, but unlike the fry,
        # are not susceptible to cannibalism, and so we don't multiply
        # the expresssion for maturations by the cannibalism multiplier.
        # Adults from the last period stay adults unless they die of
        # old age (represented in the adult survival probability) and
        # there are new adults due to juveniles maturing.
        maturations = juveniles * self.juvenile_survival_probability
        new_adults = adults * self.adult_survival_probability + maturations

        return FryModel.State(new_fry, new_juveniles, new_adults)

    def run(
        self, initial: "FryModel.State"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        "takes a initial condition and produces a tuple of numpy arrays (fry, juveniles, adults)"
        # this is mostly the same, but with an extra variable: fry
        # For more information, see the "run" method in the
        # parent class
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
