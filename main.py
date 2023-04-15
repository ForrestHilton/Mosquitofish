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

    # different cannibalism models. Ordinary plots show the 
    # number of adults and the number of juveniles over time 
    # on the same graph with fixed parameters and initial
    # numbers of adults and juveniles. Interactive seedspace
    # plots show dot plots with juveniles on the x-axis and 
    # adults on the y-axis. Each dot represents a different 
    # point in time and they're connected by arrows for 
    # clarity. The user can click on different places in the
    # graph representing different intial numbers of juveniles
    # and adults and see how different intial conditions evolve
    # over time. There are also sliders which control various
    # parameters in the model, such as fecundacy, juvenile 
    # survival probability, etc. 
    ordinary_plot_over_time(BevertonHolt())
    show_interactive_2d_seedspace(BevertonHolt(), 2000, 400)
    ordinary_plot_over_time(Allee())
    show_interactive_2d_seedspace(Allee(), 400, 100)
    ordinary_plot_over_time(Ricker())
    show_interactive_2d_seedspace(Ricker(), 400, 100)
    ordinary_plot_over_time(Linear())
    show_interactive_2d_seedspace(Linear(), 500, 200)

    # FryModel. The ordinary 3-D plot over time is like the
    # ordinary plots over time but shows the number of adults,
    # juveniles, AND fry (there are no fry in the other models)
    # on the same graph. The interactive 3-D seedspace plot is 
    # similar to the interactive 2-D seedspace plots but also 
    # includes the fry in addition to the number of adults and
    # juveniles. Sliders are used to control the initial 
    # conditions instead of clicking on the graph; sliders are
    # used to control parameters as before.  
    ordinary3d_plot_over_time(FryModel())
    show_interactive_3d_seedspace(FryModel(), 100, 10, 100)

    # competition. The sensitivity run compares the number
    # of adults present over time in the competition model
    # to the standard ricker cannibalism model. A linear 
    # cannibalism model is shown in an additional plot 
    # for additional comparison purposes. 
    sensitivity_run([CompetitionModel(), Ricker()])
    ordinary_plot_over_time(Linear())
