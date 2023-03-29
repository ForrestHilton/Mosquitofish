"""
This file contains all mathematical functions (for stocks, flows, convertors)
included in all of the possible models. This makes it easier to make variants
of models because we can just import this file and call each of the functions
in the models when we need them rather than having to redefine them each time
"""

# Simple model functions/parameters

# Time step for initial model is 8 weeks, different for other models
time_step_weeks = 8 

# Mosquitofish survive about 2 years on average (104 weeks)
avg_lifespan_weeks = 104
# This can be used to find probability of dying in a given
# time interval:
adult_die_probability = time_step_weeks/avg_lifespan_weeks
# this can then be used to calculate chance of survival 
# in given time interval
adult_survive_probability = 1 - adult_die_probability

# in simple model, we assume only death from old age and
# canabilism; we can assume probability of juvenile surviving
# to adulthood IN THE ABSENSE of canibilism should be
# very high (we can assume 1)

juveline_survive_probability = 1

# Females can give birth to up to 30 at once (assume 15 average)
avg_births_at_once = 15
# Each female can produce three to four broods in her lifetime
# https://www.shelbytnhealth.com/487/Mosquito-Fish-for-Ponds
broods_per_lifetime = 3.5
# Thus, to calculate fecundity (number of ones born per year
# on average) do:
fecundity = (avg_births_at_once*broods_per_lifetime*time_step_weeks)/avg_lifespan_weeks


 