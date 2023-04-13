"""
This file contains all mathematical functions (for stocks, flows, convertors)
included in all of the possible models. This makes it easier to make variants
of models because we can just import this file and call each of the functions
in the models when we need them rather than having to redefine them each time
"""
from model import SimpleModel

import numpy as np # maybe don't need? But probably do

# Simple model functions/parameters

# Time step for initial model is 8 weeks, different for other models
time_step_weeks = 8

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

model = SimpleModel(
    fecundity,
    juvenile_survival_probability,
    adult_survival_probability,
    strength_cannibalism,
)

# Different cannibalism models

# IMPORTANT: I think we have been misinterpreting the cannibalism
# multipliers. Right now, we are interpreting them as the raw 
# number of mosquitofish juveniles who are dying before they
# reach the next generation. I think the correct interpretation
# is that the multiplier is a term we multiply the juvenile 
# probability survival with. In model.py, this would correspond
# to changing:
# maturations = juveniles * self.juvenile_survive_probability - cannibalism
# to:
# maturations = juveniles * self.juvenile_survive_probability * cannibalism  
# and cannibalism should probably be renamed cannibalism multiplier.
# These multipliers decrease as the number of adults increases, and thus
# it makes sense that the number of maturations would decrease
# with increasing number of adults, but the way we have it set up
# now, the maturations are increasing with the number of adults.
# I have not changed the implementation in model.py yet, I'm just
# putting in new canibalism models; we can change are implementation
# when we discuss it in class if necessary.

class BevertonHolt(SimpleModel):
    def cannibalism_multiplier(self, state: SimpleModel.State):
        return 1 / (1 + self.strength_cannibalism * state.adults)

    
class Allee(SimpleModel):

    def __init__(
        self,
        fecundity: float,
        juvenile_survive_probability: float,
        adult_survival_probability: float,
        strength_cannibalism: float,
        m_param: float
    ) -> None:
        self.fecundity = fecundity
        self.juvenile_survive_probability = juvenile_survive_probability
        self.adult_survival_probability = adult_survival_probability
        self.strength_cannibalism = strength_cannibalism
        self.m_param = m_param


    def cannibalism_multiplier(self, state: SimpleModel.State):
        return 1 / (1 + self.strength_cannibalism*((state.adults-self.m_param)**2))


# What we thought was the Ricker cannibalism we implemented
# as default cannibalsim in SimpleModel, but I think this is 
# the correct implementation
class Ricker(SimpleModel):

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return np.exp(-self.strength_cannibalism*state.adults)
    
class Linear(SimpleModel):

    # Another different parameter specific to this model
    def __init__(
        self,
        fecundity: float,
        juvenile_survive_probability: float,
        adult_survival_probability: float,
        big_m_param: float,
    ) -> None:
        self.fecundity = fecundity
        self.juvenile_survive_probability = juvenile_survive_probability
        self.adult_survival_probability = adult_survival_probability
        self.big_m_param = big_m_param

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return np.maximum(1 - state.adults / self.big_m_param, 0)
    

strength_can_bev_holt = 0.1

bev_holt_model = BevertonHolt(
    fecundity,
    juvenile_survival_probability,
    adult_survival_probability,
    strength_can_bev_holt,
)

strength_can_allee = 0.1

# We need an extra parameter m for the Allee model,
# which is >= 1 but close to 1.

allee_m_param = 1.01

allee_model = Allee(
    fecundity,
    juvenile_survival_probability,
    adult_survival_probability,
    strength_can_allee,   
    allee_m_param,
)

strength_can_ricker = 0.1

ricker_model = Ricker(
    fecundity,
    juvenile_survival_probability,
    adult_survival_probability,
    strength_can_ricker,  
)

big_m_linear = 100 

linear_model = Linear(
    fecundity,
    juvenile_survival_probability,
    adult_survival_probability,
    big_m_linear,
)
