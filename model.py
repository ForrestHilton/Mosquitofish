import numpy as np


class SimpleModel:
    class State:

        """State is an inner class of SimpleModel which holds information
        about what would be the stocks in a Stella model: number of 
        juveniles and number of adults"""

        def __init__(self, juveniles: float, adults: float):
            self.juveniles = juveniles
            self.adults = adults

    def __init__(
        # Constructor: initializes all the instance variables,
        # which are convertors and can be thought of as
        # parameters since they are constant. 
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
        # This is so functions in graphs.py can title graphs based 
        # on the type of model by class 
        return type(self).__name__

    def run_one_time_step(self, state: "SimpleModel.State") -> "SimpleModel.State":
        "takes a State object and returns a new State object representing the next time step"
        # Set up temporary varaibles for adults and juvelines
        juveniles, adults = state.juveniles, state.adults

        # Births is a flow. Fecundity is nummber of births per year
        # per adult, adults is the number of adult mosquitofish,
        # births is thus the number of juvelines born per year
        births = self.fecundity * adults
        cannibalism_mult = self.cannibalism_multiplier(state) # temporary variable
        # Maturations is a flow, and it represents how many juvelines
        # mature and thus become adults in the next time step. 
        # Juvelines have a certain survival probability on their own
        # in absense of cannibalism, so this probabiliy is multiplied
        # by the number of juvelines. Cannibalism_mult represents the
        # probability a juvenile survives if cannibalism is present,
        # ignoring the other survival probability, and get mutliplied
        # by juveniles and the survival probability to yield maturations.
        maturations = juveniles * self.juvenile_survive_probability * cannibalism_mult

        # Number of new juvelines each year is the number that
        # are born each year. Cannot have less than 0 juveniles,
        # hence numpy.maximum is used. 
        new_juveniles = np.maximum(births, 0)
        # Adults have a particular survival probability, so the
        # number of adults that keep surviving from the last time
        # step is adults * adult survival probability, and there
        # are also new adults from juvelines maturing. Also
        # can't have less than 0 adults. 
        new_adults = np.maximum(
            adults * self.adult_survival_probability + maturations, 0
        )

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

    def cannibalism_multiplier(self, state: State):
        # No cannibalism in default model -> juveline survival not affected
        return 1
