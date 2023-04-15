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
    ):
        super().__init__()
        self.strength_can_bev_holt = 0.1

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return 1 / (1 + self.strength_can_bev_holt * state.adults)


class Allee(SimpleModel):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.strength_cannibalism = 0.1
        # We need an extra parameter m for the Allee model,
        # which is >= 1 but close to 1.
        self.m_param = 1.01

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return 1 / (
            1 + self.strength_cannibalism * ((state.adults - self.m_param) ** 2)
        )


class Ricker(SimpleModel):
    def __init__(
        self,
        time_step_weeks=8,
    ) -> None:
        super().__init__(time_step_weeks=time_step_weeks)
        self.strength_cannibalism = 0.1

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return np.exp(-self.strength_cannibalism * state.adults)


class Linear(SimpleModel):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.big_m_param = 100

    def cannibalism_multiplier(self, state: SimpleModel.State):
        return np.maximum(1 - state.adults / self.big_m_param, 0)
