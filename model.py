import numpy as np
import math

# except as otherwise noted, all units are in Fish or unit less


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
        # Time step for initial model is 8 weeks, different for other models
        time_step_weeks=8,
    ) -> None:
        # simple model functions/parameters

        # Mosquitofish survive about 2 years on average (104 weeks)
        avg_lifespan_weeks = 104

        # duration is twice the average life span for starters
        self.iterations = math.floor(avg_lifespan_weeks / time_step_weeks * 2)
        # This can be used to find probability of dying in a given
        # time interval:
        adult_die_probability = time_step_weeks / avg_lifespan_weeks
        # this can then be used to calculate chance of survival
        # in given time interval
        self.adult_survival_probability = 1 - adult_die_probability

        # in simple model, we assume only death from old age and
        # canabilism; we can assume probability of juvenile surviving
        # to adulthood IN THE ABSENSE of canibilism should be
        # very high (we can assume 1)

        self.juvenile_survival_probability = 1

        # Females can give birth to up to 30 young per brood (assume 15 average)
        avg_births_per_brood = 15
        # Each female can produce three to four broods in her lifetime
        # https://www.shelbytnhealth.com/487/Mosquito-Fish-for-Ponds
        broods_per_lifetime = 3.5
        # Thus, to calculate fecundity (number of young born per adult per year
        # on average) do:
        self.fecundity = (
            avg_births_per_brood * broods_per_lifetime * time_step_weeks
        ) / avg_lifespan_weeks

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
        cannibalism_mult = self.cannibalism_multiplier(state)  # temporary variable
        # Maturations is a flow, and it represents how many juvelines
        # mature and thus become adults in the next time step.
        # Juvelines have a certain survival probability on their own
        # in absense of cannibalism, so this probabiliy is multiplied
        # by the number of juvelines. Cannibalism_mult represents the
        # probability of a juvenile surviving if cannibalism is present,
        # ignoring the other survival probability, and gets mutliplied
        # by juveniles and the survival probability to yield maturations.
        maturations = juveniles * self.juvenile_survival_probability * cannibalism_mult

        # Number of new juvelines each year is the number that
        # are born each year. Cannot have less than 0 juveniles,
        # hence numpy.maximum is used.
        new_juveniles = births
        # Adults have a particular survival probability, so the
        # number of adults that keep surviving from the last time
        # step is adults * adult survival probability, and there
        # are also new adults from juvelines maturing. Also
        # can't have less than 0 adults.
        new_adults = adults * self.adult_survival_probability + maturations

        return SimpleModel.State(new_juveniles, new_adults)

    def run(self, initial: "SimpleModel.State") -> tuple[np.ndarray, np.ndarray]:
        "takes a initial condition and produces a tuple of numpy arrays (juveniles, adults)"
        # We want to make sure the number of iterations is a
        # whole number, so we use the floor function
        self.iterations = math.floor(self.iterations)

        # This function is going to return arrays representing
        # the number of juvenlines and adults that exist at each
        # time step in the simulation so this data can be graphed.
        # Here, we're setting up the initial state of the simulation
        # and loading the initial state (number of juveniles and
        # adults) into the appropriate arrays.
        current_state = initial
        juveniles_array = np.array([initial.juveniles])
        adults_array = np.array([initial.adults])

        # In this loop, we run a series of time steps one after
        # the other, each depending on the one before, in a manner
        # similar to Euler's method. Each time we have a new state
        # we append the data to the appropriate arrays.
        for _ in range(self.iterations - 1):
            current_state = self.run_one_time_step(current_state)
            juveniles_array = np.append(juveniles_array, current_state.juveniles)
            adults_array = np.append(adults_array, current_state.adults)

        return juveniles_array, adults_array

    def cannibalism_multiplier(self, state: State):
        # No cannibalism in default model -> juvenile survival not affected
        return 1
