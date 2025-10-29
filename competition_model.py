# competition_model.py
from __future__ import annotations
from dataclasses import dataclass
from .frymodel import FryModel
from .model import SimpleModel

@dataclass
class CompetitionModel(FryModel):
    """
    FryModel + quadratic crowding losses on the *combined* adult stock.

    adult_competition_beta scales a loss term ~ beta * (AF + AM)^2 each step.
    The loss is removed proportionally from AF and AM to preserve the
    current sex ratio.
    """
    adult_competition_beta: float = 0.0  # loss ~ beta * (AF + AM)^2

    def step(self, s: SimpleModel.State) -> SimpleModel.State:
        # First run the base dynamics (survival, cannibalism-gated F->J, J->A split, recruitment)
        s2 = super().step(s)

        if self.adult_competition_beta > 0.0:
            A = s2.adult_female + s2.adult_male
            if A > 0.0:
                comp = self.adult_competition_beta * (A ** 2)
                # Remove proportionally to preserve sex ratio this step
                af_loss = comp * (s2.adult_female / A)
                am_loss = comp * (s2.adult_male   / A)
                s2 = SimpleModel.State(
                    s2.fry,
                    s2.juveniles,
                    max(0.0, s2.adult_female - af_loss),
                    max(0.0, s2.adult_male - am_loss),
                )

        return s2