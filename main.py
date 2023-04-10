import functions as fn
from graph import ordinary_plot_over_time

if __name__ == "__main__":

    # Beverton Holt
    ordinary_plot_over_time(fn.bev_holt_model, 20)

    # Allee
    ordinary_plot_over_time(fn.allee_model, 20)

    # Ricker
    ordinary_plot_over_time(fn.ricker_model, 20)

    # Linear
    ordinary_plot_over_time(fn.linear_model, 20)
