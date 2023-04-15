from cannibalism_models import BevertonHolt, Allee, Ricker, Linear
from competition_model import CompetitionModel
from frymodel import FryModel
from graph import (
    sensitivity_run,
    show_interactive_2d_seedspace,
    show_interactive_3d_seedspace,
    ordinary_plot_over_time,
    ordinary3d_plot_over_time,
)

# in a terminal run "from main import *" and you can execute any of this
if __name__ == "__main__":

    # different cannibalism models
    ordinary_plot_over_time(BevertonHolt())
    show_interactive_2d_seedspace(BevertonHolt(), 2000, 400)
    ordinary_plot_over_time(Allee())
    show_interactive_2d_seedspace(Allee(), 400, 100)
    ordinary_plot_over_time(Ricker())
    show_interactive_2d_seedspace(Ricker(), 400, 100)
    ordinary_plot_over_time(Linear())
    show_interactive_2d_seedspace(Linear(), 500, 200)

    # FryModel
    ordinary3d_plot_over_time(FryModel())
    show_interactive_3d_seedspace(FryModel(), 100, 10, 100)

    # competition
    sensitivity_run([CompetitionModel(), Ricker()])
    ordinary_plot_over_time(Linear())
