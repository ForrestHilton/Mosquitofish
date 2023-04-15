from model import SimpleModel

import numpy as np


# Different cannibalism models

# IMPORTANT: The correct interpretation of the cannibalism function
# is that the multiplier is a term we multiply the juvenile
# probability survival with.

# These multipliers decrease as the number of adults increases, and thus
# it makes sense that the number of maturations would decrease
# with increasing number of adults.


class BevertonHolt(SimpleModel):
    def __init__(
        self,
        strength_can_bev_holt=0.1,
    ):
        super().__init__()
        self.strength_can_bev_holt = strength_can_bev_holt

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return 1 / (1 + self.strength_can_bev_holt * state.adults)


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


class Ricker(SimpleModel):
    def __init__(
        self,
        strength_can_ricker=0.1,
    ) -> None:
        super().__init__()
        self.strength_cannibalism = strength_can_ricker

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return np.exp(-self.strength_cannibalism * state.adults)


class Linear(SimpleModel):
    def __init__(
        self,
        big_m_linear=100,
    ) -> None:
        super().__init__()
        self.big_m_param = big_m_linear

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return np.maximum(1 - state.adults / self.big_m_param, 0)
