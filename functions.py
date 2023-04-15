"""
This file contains all mathematical functions (for stocks, flows, converters)
included in all of the possible models. This makes it easier to make variants
of models because we can just import this file and call each of the functions
in the models when we need them rather than having to redefine them each time
"""
from model import SimpleModel

import numpy as np


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
    def __init__(
        self,
        strength_can_bev_holt=0.1,
    ):
        super().__init__()
        self.strength_can_bev_holt = strength_can_bev_holt

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return 1 / (1 + self.strength_can_bev_holt * state.adults)


bev_holt_model = BevertonHolt()


class Allee(SimpleModel):
    def __init__(
        self,
        strength_can_allee=0.1,
        # We need an extra parameter m for the Allee model,
        # which is >= 1 but close to 1.
        allee_m_param=1.01,
    ) -> None:
        super().__init__()
        self.strength_cannibalism = strength_can_allee
        self.m_param = allee_m_param

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return 1 / (
            1 + self.strength_cannibalism * ((state.adults - self.m_param) ** 2)
        )


allee_model = Allee()


# What we thought was the Ricker cannibalism we implemented
# as default cannibalsim in SimpleModel, but I think this is
# the correct implementation
class Ricker(SimpleModel):
    def __init__(
        self,
        strength_can_ricker=0.1,
    ) -> None:
        super().__init__()
        self.strength_cannibalism = strength_can_ricker

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return np.exp(-self.strength_cannibalism * state.adults)


ricker_model = Ricker()


class Linear(SimpleModel):
    def __init__(
        self,
        big_m_linear=100,
    ) -> None:
        super().__init__()
        self.big_m_param = big_m_linear

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return np.maximum(1 - state.adults / self.big_m_param, 0)


linear_model = Linear()
